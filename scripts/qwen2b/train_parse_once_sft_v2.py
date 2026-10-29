"""Continue the flat-condition adapter on evidence-checked special phrasing."""
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

import prepare_supervision as sup
import run_parse_once as base


ROOT = sup.ROOT
V1 = ROOT / "runs/qwen2b_parse_once_sft_v1"
OUT = ROOT / "runs/qwen2b_parse_once_sft_v2"
GENERAL = ROOT / "runs/qwen2b_parse_once_sft_holdout_v1"
SEED = 20261022


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def generate(model, tokenizer, rows, path, stage, id_key):
    import torch
    predictions = []
    with path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            messages = [{"role": "system", "content": base.SYSTEM_PROMPT},
                        {"role": "user", "content": "Description: " + row["description"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(**inputs, do_sample=False, max_new_tokens=128,
                                        pad_token_id=tokenizer.eos_token_id, use_cache=True)
            torch.cuda.synchronize(); generated = output[0, inputs.input_ids.shape[1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True).strip(); parsed, error = base.parse_strict(raw)
            item = {id_key: row[id_key], "raw": raw, "parsed": parsed, "parse_error": error,
                    "seconds": time.perf_counter()-started, "generated_tokens": len(generated),
                    "hit_token_cap": len(generated)>=128 and generated[-1].item()!=tokenizer.eos_token_id}
            predictions.append(item); stream.write(json.dumps(item, ensure_ascii=False)+"\n"); stream.flush()
            elapsed = sum(x["seconds"] for x in predictions)
            atomic(OUT / "status.json", {"status": stage, "completed": index, "total": len(rows),
                   "invalid_outputs": sum(x["parsed"] is None for x in predictions),
                   "mean_seconds": elapsed/index, "remaining_minutes_estimate": (len(rows)-index)*elapsed/index/60})
            print(stage, index, "/", len(rows), round(item["seconds"],2), "s", "valid" if parsed else error, flush=True)
    return predictions


def score_special(rows, predictions):
    by_id = {x["sample_id"]: x for x in predictions}; details=[]; errors=collections.Counter()
    for row in rows:
        p=by_id[row["sample_id"]]
        if p["parsed"] is None:
            details.append({**row,"raw":p["raw"],"parsed":None,"parse_error":p["parse_error"],"exact":False})
            errors["invalid"]+=1; continue
        missing,extra,wrong=base.condition_diff(p["parsed"],row["target"])
        for key in missing: errors[f"missing::{key}"]+=1
        for key in extra: errors[f"extra::{key}"]+=1
        for key in wrong: errors[f"wrong::{key}"]+=1
        details.append({**row,"raw":p["raw"],"parsed":p["parsed"],"missing_keys":missing,
                        "extra_keys":extra,"wrong_value_keys":wrong,"exact":not(missing or extra or wrong)})
    return {"valid":sum(x["parsed"] is not None for x in predictions),"total":len(rows),
            "exact_conditions":sum(x["exact"] for x in details),"field_errors":dict(sorted(errors.items())),
            "inference_seconds":sum(x["seconds"] for x in predictions)},details


def score_general(descriptions, cases, predictions):
    by_pred={x["description_id"]:x for x in predictions}; by_source={x["source_key"]:x for x in descriptions}
    extraction=[]; field_errors=collections.Counter()
    for d in descriptions:
        p=by_pred[d["description_id"]]
        if p["parsed"] is None:
            extraction.append({**d,"raw":p["raw"],"parsed":None,"parse_error":p["parse_error"],"exact":False}); field_errors["invalid"]+=1; continue
        missing,extra,wrong=base.condition_diff(p["parsed"],d["gold_conditions"])
        for key in missing: field_errors[f"missing::{key}"]+=1
        for key in extra: field_errors[f"extra::{key}"]+=1
        for key in wrong: field_errors[f"wrong::{key}"]+=1
        extraction.append({**d,"raw":p["raw"],"parsed":p["parsed"],"missing_keys":missing,
                           "extra_keys":extra,"wrong_value_keys":wrong,"exact":not(missing or extra or wrong)})
    counts=collections.Counter(); scored=[]
    for case in cases:
        d=by_source[base.description_key(case)]; p=by_pred[d["description_id"]]
        if p["parsed"] is None:
            pred=None; checks=None; correct=False; counts["unresolved"]+=1
            if case["target"]: counts["fn"]+=1
        else:
            value,checks=base.match_candidate(p["parsed"],case["input"]["candidate"]); pred=int(value); correct=pred==case["target"]
            if pred and case["target"]:counts["tp"]+=1
            elif pred:counts["fp"]+=1
            elif case["target"]:counts["fn"]+=1
            else:counts["tn"]+=1
        scored.append({"pair_id":case["input"]["pair_id"],"description_id":d["description_id"],
                       "gold":case["target"],"prediction":pred,"correct":correct,"checks":checks})
    correct=sum(x["correct"] for x in scored)
    return {"valid":sum(x["parsed"] is not None for x in predictions),"exact_conditions":sum(x["exact"] for x in extraction),
            "descriptions":len(descriptions),"candidate_correct":correct,"candidate_total":len(cases),
            "tp":counts["tp"],"fp":counts["fp"],"fn":counts["fn"],"tn":counts["tn"],
            "unresolved":counts["unresolved"],"field_errors":dict(sorted(field_errors.items())),
            "inference_seconds":sum(x["seconds"] for x in predictions)},extraction,scored


def main():
    manifest=sup.read(OUT/"manifest.json")
    assert manifest["train_sha256"]==sup.digest(OUT/"train.json") and manifest["holdout_sha256"]==sup.digest(OUT/"special_holdout.json")
    for name,digest in manifest["parent_adapter_sha256"].items():assert sup.digest(V1/"adapter"/name)==digest
    assert not (OUT/"adapter").exists(),"V2 adapter already exists"
    os.environ["HF_HOME"]=str(ROOT/".cache/huggingface");os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,BitsAndBytesConfig,Qwen3_5ForCausalLM
    from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict
    from safetensors.torch import load_file
    torch.manual_seed(SEED)
    tokenizer=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
    training=sup.read(OUT/"train.json");prepared=[]
    for row in training:
        messages=[{"role":"system","content":base.SYSTEM_PROMPT},{"role":"user","content":"Description: "+row["description"]}]
        prefix=tokenizer.encode(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        answer=json.dumps(row["target"],ensure_ascii=False,separators=(",",":"));suffix=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
        assert tokenizer.decode(suffix[:-1])==answer and len(prefix+suffix)<=1024
        prepared.append((row["sample_id"],prefix,suffix))
    config={"seed":SEED,"parent":"qwen2b_parse_once_sft_v1","steps":len(prepared),"special":256,"replay":256,
            "batch_size":1,"learning_rate":0.00005,"max_new_tokens":128,
            "versions":{p:importlib.metadata.version(p) for p in ("torch","transformers","peft","bitsandbytes")}}
    sup.write(OUT/"config.json",config);atomic(OUT/"status.json",{"status":"LOADING","completed_steps":0,"total_steps":len(prepared)})
    model,loading=Qwen3_5ForCausalLM.from_pretrained(base.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map={"":0},
        attn_implementation="sdpa",output_loading_info=True,quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs")
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
    model=PeftModel.from_pretrained(model,V1/"adapter",is_trainable=True)
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad};frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    initial={n:p.detach().cpu().clone() for n,p in trainable.items()};assert trainable and all("lora_" in n for n in trainable)
    optimizer=torch.optim.AdamW(list(trainable.values()),lr=config["learning_rate"])
    model.train();model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    steps=[]
    with (OUT/"steps.jsonl").open("w",encoding="utf-8") as stream:
        for index,(sample_id,prefix,suffix) in enumerate(prepared,1):
            ids=torch.tensor([prefix+suffix],device="cuda");positions=torch.arange(len(prefix)-1,ids.shape[1]-1,device="cuda");labels=ids[:,len(prefix):]
            optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=positions).logits
                loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1))
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
            assert all(p.grad is None and p._version==frozen[n] for n,p in model.named_parameters() if not p.requires_grad)
            rec={"step":index,"sample_id":sample_id,"loss":loss.item(),"gradient_norm":norm.item(),"seconds":time.perf_counter()-started,
                 "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,"peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2}
            steps.append(rec);stream.write(json.dumps(rec)+"\n");stream.flush();elapsed=sum(x["seconds"] for x in steps)
            atomic(OUT/"status.json",{"status":"TRAINING","completed_steps":index,"total_steps":len(prepared),"last_loss":rec["loss"],
                   "mean_seconds_per_step":elapsed/index,"remaining_minutes_estimate":(len(prepared)-index)*elapsed/index/60,
                   "peak_allocated_mib":max(x["peak_allocated_mib"] for x in steps)})
            if index%8==0:print("train",index,"/",len(prepared),"loss",round(rec["loss"],4),flush=True)
            del ids,positions,labels,logits,loss
    changed=sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in trainable.items());assert changed>0
    model.save_pretrained(OUT/"adapter",safe_serialization=True);saved=load_file(OUT/"adapter/adapter_model.safetensors");live=get_peft_model_state_dict(model)
    assert set(saved)==set(live) and all(torch.equal(saved[k],live[k].detach().cpu()) for k in saved)
    training_seconds=sum(x["seconds"] for x in steps);peak=max(x["peak_allocated_mib"] for x in steps)
    model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True
    special_rows=sup.read(OUT/"special_holdout.json")
    special_predictions=generate(model,tokenizer,special_rows,OUT/"special_predictions.jsonl","SPECIAL_EVAL","sample_id")
    special_result,special_details=score_special(special_rows,special_predictions);sup.write(OUT/"special_details.json",special_details)
    general_descriptions=sup.read(GENERAL/"descriptions.json");general_cases=sup.read(GENERAL/"validation.json")
    general_predictions=generate(model,tokenizer,general_descriptions,OUT/"general_predictions.jsonl","GENERAL_REGRESSION","description_id")
    general_result,general_extraction,general_scored=score_general(general_descriptions,general_cases,general_predictions)
    sup.write(OUT/"general_extraction.json",general_extraction);sup.write(OUT/"general_scored.json",general_scored)
    result={"training":{"steps":len(steps),"seconds":training_seconds,"peak_allocated_mib":peak,"changed_tensors":changed},
            "special_holdout":special_result,"general_regression":general_result,
            "parent_reference":{"special_training_added":256,"replay":256,"v1_general_correct":240}}
    sup.write(OUT/"results.json",result);sup.write(OUT/"complete.json",{"status":"PASS","adapter_save_verified":True,
              "source_frozen":True,"special_exact":special_result["exact_conditions"],"general_correct":general_result["candidate_correct"]})
    atomic(OUT/"status.json",{"status":"COMPLETE","special_exact":special_result["exact_conditions"],
           "special_total":120,"general_correct":general_result["candidate_correct"],"general_total":240})
    if (OUT/"failure.json").exists():(OUT/"failure.json").unlink()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    try:main()
    except Exception as error:
        if OUT.exists():atomic(OUT/"failure.json",{"error":str(error),"traceback":traceback.format_exc()});atomic(OUT/"status.json",{"status":"FAILED","error":str(error)})
        raise
