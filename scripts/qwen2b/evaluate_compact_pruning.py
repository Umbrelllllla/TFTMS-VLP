"""Post-training LoRA pruning sensitivity study for the compact adapter."""
import json
import os
from pathlib import Path
import time
import traceback

import compact_format as compact
import prepare_supervision as sup
import run_parse_once as base
import train_parse_once_sft_v2 as v2
import repair_compact_sft as repair


ROOT = sup.ROOT
SOURCE = ROOT / "runs/qwen2b_compact_sft_v2"
GENERAL = ROOT / "runs/qwen2b_parse_once_sft_holdout_v1"
OUT = ROOT / "runs/qwen2b_compact_pruning_v1"
MODEL = base.MODEL


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def prune_tensor(tensor, method):
    import torch
    result = tensor.clone()
    if method == "dense": return result
    if method.startswith("unstructured_"):
        fraction = float(method.rsplit("_", 1)[1])
        count = int(result.numel() * fraction)
        if count:
            indices = result.abs().reshape(-1).topk(count, largest=False).indices
            result.reshape(-1)[indices] = 0
        return result
    if method == "structured_2of4":
        if result.ndim != 2 or result.shape[-1] % 4:
            raise ValueError(f"2:4 incompatible tensor shape: {tuple(result.shape)}")
        groups = result.reshape(-1, 4)
        keep = groups.abs().topk(2, dim=1).indices
        mask = torch.zeros_like(groups, dtype=torch.bool)
        mask.scatter_(1, keep, True)
        groups.mul_(mask)
        return result
    raise ValueError(method)


def generate(model, tokenizer, rows, batch_size=8):
    import torch
    indexed = sorted(enumerate(rows), key=lambda item:len(item[1]["description"]))
    predictions = [None] * len(rows); started_all=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for start in range(0,len(indexed),batch_size):
        batch=indexed[start:start+batch_size];prompts=[]
        for _,row in batch:
            messages=[{"role":"system","content":repair.SYSTEM_PROMPT},
                      {"role":"user","content":"Description: "+row["description"]}]
            prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,
                add_generation_prompt=True,enable_thinking=False))
        inputs=tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors="pt").to("cuda")
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            output=model.generate(**inputs,do_sample=False,max_new_tokens=48,
                                  pad_token_id=tokenizer.eos_token_id,use_cache=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        raws=tokenizer.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        for (original,row),raw in zip(batch,raws):
            raw=raw.strip();parsed,error=compact.parse(raw)
            predictions[original]={"description_id":row["description_id"],"raw":raw,"parsed":parsed,
                "parse_error":error,"seconds":seconds/len(batch),"batch_size":len(batch)}
        del inputs,output
    return predictions,{"wall_seconds":time.perf_counter()-started_all,
        "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2}


def main():
    assert sup.read(SOURCE/"complete.json")["status"]=="PASS"
    OUT.mkdir(parents=True,exist_ok=False)
    os.environ["HF_HOME"]=str(ROOT/".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer,BitsAndBytesConfig,Qwen3_5ForCausalLM
    from peft import PeftModel
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tokenizer.padding_side="left"
    model,loading=Qwen3_5ForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        device_map={"":0},attn_implementation="sdpa",output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs")
    model=PeftModel.from_pretrained(model,SOURCE/"adapter",is_trainable=False)
    model.eval();model.config.use_cache=True
    lora={name:param for name,param in model.named_parameters() if "lora_" in name}
    assert lora and all(param.ndim==2 and param.shape[-1]%4==0 for param in lora.values())
    original={name:param.detach().clone() for name,param in lora.items()}
    descriptions=sup.read(GENERAL/"descriptions.json");cases=sup.read(GENERAL/"validation.json")
    methods=("dense","unstructured_0.25","unstructured_0.5","structured_2of4")
    results={}
    for method in methods:
        zeros=total=0
        with torch.no_grad():
            for name,param in lora.items():
                value=prune_tensor(original[name],method);param.copy_(value)
                zeros+=int((value==0).sum().item());total+=value.numel()
        predictions,timing=generate(model,tokenizer,descriptions)
        score,extraction,scored=v2.score_general(descriptions,cases,predictions)
        result={"method":method,"zero_parameters":zeros,"total_lora_parameters":total,
                "sparsity":zeros/total,"score":score,"timing":timing}
        results[method]=result
        sup.write(OUT/f"{method}_predictions.json",predictions)
        sup.write(OUT/f"{method}_extraction.json",extraction)
        sup.write(OUT/f"{method}_scored.json",scored)
        atomic(OUT/"status.json",{"status":"EVALUATING","completed":method,"results":results})
        print(method,json.dumps(result,ensure_ascii=False),flush=True)
    with torch.no_grad():
        for name,param in lora.items():param.copy_(original[name])
    final={"status":"COMPLETE","scope":"LoRA tensors only; dense kernels retained",
           "speed_claim_policy":"Sparsity is not called acceleration unless measured wall time improves.",
           "methods":results}
    sup.write(OUT/"results.json",final);atomic(OUT/"status.json",{"status":"COMPLETE"})
    print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    try:main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic(OUT/"failure.json",{"error":str(error),"traceback":traceback.format_exc()})
        atomic(OUT/"status.json",{"status":"FAILED","error":str(error)})
        raise
