"""Continue V2 LoRA training with a compact output language, then evaluate in batches."""
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import compact_format as compact
import prepare_supervision as sup
import run_parse_once as base
import train_parse_once_sft_v2 as v2


ROOT = sup.ROOT
SOURCE = ROOT / "runs/qwen2b_parse_once_sft_v2"
GENERAL = ROOT / "runs/qwen2b_parse_once_sft_holdout_v1"
OUT = ROOT / "runs/qwen2b_compact_sft_v1"
MODEL = base.MODEL
SEED = 20261029


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def prepare_snapshot():
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ("train.json", "special_holdout.json"):
        shutil.copy2(SOURCE / name, OUT / name)
    manifest = {
        "status":"PREPARED", "seed":SEED, "parent":"qwen2b_parse_once_sft_v2",
        "training_samples":512, "special_holdout":120,
        "train_sha256":sup.digest(OUT / "train.json"),
        "holdout_sha256":sup.digest(OUT / "special_holdout.json"),
        "compact_prompt_sha256":compact.prompt_hash(),
        "parent_adapter_sha256":{p.name:sup.digest(p) for p in (SOURCE/"adapter").iterdir() if p.is_file()},
        "format":"C,K,P,F,O,D,L,W,H in canonical order; pipe-separated; ranges use min:max",
        "test_labels_used":False,
    }
    sup.write(OUT / "manifest.json", manifest)
    atomic(OUT / "status.json", {"status":"PREPARED"})
    return manifest


def batched_generate(model, tokenizer, rows, id_key, stage, batch_size=8, system_prompt=compact.SYSTEM_PROMPT):
    import torch
    indexed = sorted(enumerate(rows), key=lambda item:len(item[1]["description"]))
    predictions = [None] * len(rows)
    records = []
    torch.cuda.reset_peak_memory_stats(); started_all = time.perf_counter()
    for start in range(0, len(indexed), batch_size):
        batch = indexed[start:start+batch_size]
        prompts=[]
        for _, row in batch:
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":"Description: "+row["description"]}]
            prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,
                           add_generation_prompt=True,enable_thinking=False))
        inputs=tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors="pt").to("cuda")
        torch.cuda.synchronize(); started=time.perf_counter()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            output=model.generate(**inputs,do_sample=False,max_new_tokens=48,
                                  pad_token_id=tokenizer.eos_token_id,use_cache=True)
        torch.cuda.synchronize(); seconds=time.perf_counter()-started
        suffix=output[:,inputs.input_ids.shape[1]:]
        raws=tokenizer.batch_decode(suffix,skip_special_tokens=True)
        for (original,row),raw in zip(batch,raws):
            raw=raw.strip();parsed,error=compact.parse(raw)
            predictions[original]={id_key:row[id_key],"raw":raw,"parsed":parsed,"parse_error":error,
                "seconds":seconds/len(batch),"batch_size":len(batch)}
        records.append({"rows":len(batch),"seconds":seconds,
                        "prompt_tokens":int(inputs.attention_mask.sum().item())})
        atomic(OUT/"status.json",{"status":stage,"completed":min(start+len(batch),len(rows)),
               "total":len(rows),"invalid":sum(x is not None and x["parsed"] is None for x in predictions)})
        print(stage,min(start+len(batch),len(rows)),"/",len(rows),round(seconds,2),"s",flush=True)
        del inputs,output,suffix
    return predictions,{"wall_seconds":time.perf_counter()-started_all,
        "generate_seconds":sum(x["seconds"] for x in records),
        "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,"batch_size":batch_size,"batches":records}


def main():
    manifest=prepare_snapshot() if not OUT.exists() else sup.read(OUT/"manifest.json")
    assert manifest["train_sha256"]==sup.digest(OUT/"train.json")
    assert manifest["holdout_sha256"]==sup.digest(OUT/"special_holdout.json")
    assert manifest["compact_prompt_sha256"]==compact.prompt_hash()
    assert not (OUT/"adapter").exists(),"compact adapter already exists"
    for name,digest in manifest["parent_adapter_sha256"].items():
        assert sup.digest(SOURCE/"adapter"/name)==digest
    os.environ["HF_HOME"]=str(ROOT/".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,BitsAndBytesConfig,Qwen3_5ForCausalLM
    from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict
    from safetensors.torch import load_file
    torch.manual_seed(SEED)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tokenizer.padding_side="left"
    training=sup.read(OUT/"train.json");prepared=[]
    long_target_tokens=0;compact_target_tokens=0;prompt_tokens=[]
    for row in training:
        messages=[{"role":"system","content":compact.SYSTEM_PROMPT},
                  {"role":"user","content":"Description: "+row["description"]}]
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=tokenizer.encode(prompt,add_special_tokens=False)
        answer=compact.encode(row["target"]);suffix=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
        assert compact.parse(answer)==(row["target"],None)
        assert tokenizer.decode(suffix[:-1])==answer and len(prefix+suffix)<=1024
        prepared.append((row["sample_id"],prefix,suffix));prompt_tokens.append(len(prefix))
        compact_target_tokens+=len(suffix)-1
        long_target_tokens+=len(tokenizer.encode(json.dumps(row["target"],ensure_ascii=False,separators=(",",":")),add_special_tokens=False))
    config={"seed":SEED,"steps":len(prepared),"batch_size":1,"learning_rate":0.00005,
        "training_source":"V2 adapter plus the unchanged 512 V2 samples",
        "compact_target_tokens":compact_target_tokens,"long_json_target_tokens":long_target_tokens,
        "target_token_reduction":1-compact_target_tokens/long_target_tokens,
        "mean_prompt_tokens":sum(prompt_tokens)/len(prompt_tokens),"max_prompt_tokens":max(prompt_tokens),
        "eval_batch_size":8,"max_new_tokens":48,
        "versions":{p:importlib.metadata.version(p) for p in ("torch","transformers","peft","bitsandbytes")}}
    sup.write(OUT/"config.json",config);atomic(OUT/"status.json",{"status":"LOADING","total_steps":len(prepared)})
    model,loading=Qwen3_5ForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        device_map={"":0},attn_implementation="sdpa",output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs")
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False})
    model=PeftModel.from_pretrained(model,SOURCE/"adapter",is_trainable=True)
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
    assert trainable and all("lora_" in n for n in trainable)
    optimizer=torch.optim.AdamW(list(trainable.values()),lr=config["learning_rate"])
    model.train();model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    steps=[]
    with (OUT/"steps.jsonl").open("w",encoding="utf-8") as stream:
        for index,(sample_id,prefix,suffix) in enumerate(prepared,1):
            ids=torch.tensor([prefix+suffix],device="cuda")
            positions=torch.arange(len(prefix)-1,ids.shape[1]-1,device="cuda");labels=ids[:,len(prefix):]
            optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=positions).logits
                loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1))
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            assert all(p.grad is None and p._version==frozen[n] for n,p in model.named_parameters() if not p.requires_grad)
            rec={"step":index,"sample_id":sample_id,"loss":loss.item(),"gradient_norm":norm.item(),
                 "seconds":time.perf_counter()-started,"supervised_tokens":labels.numel(),
                 "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
                 "peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2}
            steps.append(rec);stream.write(json.dumps(rec)+"\n");stream.flush();elapsed=sum(x["seconds"] for x in steps)
            atomic(OUT/"status.json",{"status":"TRAINING","completed_steps":index,"total_steps":len(prepared),
                "last_loss":rec["loss"],"mean_seconds_per_step":elapsed/index,
                "remaining_minutes_estimate":(len(prepared)-index)*elapsed/index/60,
                "peak_allocated_mib":max(x["peak_allocated_mib"] for x in steps)})
            if index%8==0:print("compact train",index,"/",len(prepared),"loss",round(rec["loss"],4),flush=True)
            del ids,positions,labels,logits,loss
    changed=sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in trainable.items());assert changed>0
    model.save_pretrained(OUT/"adapter",safe_serialization=True)
    saved=load_file(OUT/"adapter/adapter_model.safetensors");live=get_peft_model_state_dict(model)
    assert set(saved)==set(live) and all(torch.equal(saved[k],live[k].detach().cpu()) for k in saved)
    training_seconds=sum(x["seconds"] for x in steps);training_peak=max(x["peak_allocated_mib"] for x in steps)
    model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True
    special_rows=sup.read(OUT/"special_holdout.json")
    special_predictions,special_timing=batched_generate(model,tokenizer,special_rows,"sample_id","SPECIAL_EVAL")
    special_result,special_details=v2.score_special(special_rows,special_predictions)
    general_descriptions=sup.read(GENERAL/"descriptions.json");general_cases=sup.read(GENERAL/"validation.json")
    general_predictions,general_timing=batched_generate(model,tokenizer,general_descriptions,"description_id","GENERAL_EVAL")
    general_result,general_extraction,general_scored=v2.score_general(general_descriptions,general_cases,general_predictions)
    sup.write(OUT/"special_predictions.json",special_predictions);sup.write(OUT/"special_details.json",special_details)
    sup.write(OUT/"general_predictions.json",general_predictions);sup.write(OUT/"general_extraction.json",general_extraction)
    sup.write(OUT/"general_scored.json",general_scored)
    result={"format":config,"training":{"steps":len(steps),"seconds":training_seconds,
            "peak_allocated_mib":training_peak,"changed_tensors":changed},
            "special_holdout":special_result,"special_timing":special_timing,
            "general_regression":general_result,"general_timing":general_timing}
    sup.write(OUT/"results.json",result)
    passed=special_result["exact_conditions"]>=119 and general_result["candidate_correct"]==240
    sup.write(OUT/"complete.json",{"status":"PASS" if passed else "FAIL",
        "adapter_save_verified":True,"source_frozen":True,"special_exact":special_result["exact_conditions"],
        "general_correct":general_result["candidate_correct"]})
    atomic(OUT/"status.json",{"status":"COMPLETE" if passed else "FAILED_VALIDATION",
        "special_exact":special_result["exact_conditions"],"general_correct":general_result["candidate_correct"]})
    if (OUT/"failure.json").exists():(OUT/"failure.json").unlink()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    try:main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic(OUT/"failure.json",{"error":str(error),"traceback":traceback.format_exc()})
        atomic(OUT/"status.json",{"status":"FAILED","error":str(error)})
        raise
