"""Targeted compact-format repair for hyphenated colors, using train split only."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback

import compact_format as compact
import prepare_parse_once_sft_v2 as prepare_v2
import prepare_supervision as sup
import run_parse_once as base
import train_compact_sft as compact_train
import train_parse_once_sft_v2 as v2


ROOT=sup.ROOT
SOURCE=ROOT/"runs/qwen2b_compact_sft_v1"
GENERAL=ROOT/"runs/qwen2b_parse_once_sft_holdout_v1"
OUT=ROOT/"runs/qwen2b_compact_sft_v2"
MODEL=base.MODEL
SEED=20261030
SYSTEM_PROMPT=compact.SYSTEM_PROMPT+"\nCopy K color spelling exactly, including every hyphen."


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8");tmp.replace(path)


def prepare_rows():
    split=sup.read(ROOT/"runs/20261015_qwen2b_input_v1/split_manifest.json");train_files=set(split["train"])
    records=sup.read(ROOT/"runs/20261015_qwen2b_data_audit/training_records.json")
    candidates=[]
    for record in records:
        if record["file"] not in train_files:continue
        for index,annotation in enumerate(record["annotations"]):
            try:target=prepare_v2.evidence_target(annotation)
            except ValueError:continue
            color=target.get("color","")
            if "-" not in color:continue
            candidates.append({"source_file":record["file"],"camera":record["camera"],
                "annotation_index":index,"description":annotation["text"],"target":target,
                "source":"hyphen_color"})
    rng=random.Random(SEED);rng.shuffle(candidates)
    candidates.sort(key=lambda row:(row["target"].get("color")!="dark-brown",row["source_file"]))
    selected=[];used_files=set()
    for row in candidates:
        if row["source_file"] in used_files:continue
        selected.append(row);used_files.add(row["source_file"])
        if len(selected)==32:break
    assert len(selected)==32 and any(x["target"].get("color")=="dark-brown" for x in selected)
    replay=[x for x in sup.read(SOURCE/"train.json") if x["source_file"] not in used_files]
    rng.shuffle(replay);replay=replay[:32]
    rows=selected+[{**x,"source":"replay"} for x in replay];rng.shuffle(rows)
    for i,row in enumerate(rows,1):row["sample_id"]=f"compact-repair-{i:03d}"
    return rows,candidates


def main():
    assert sup.read(SOURCE/"complete.json")["adapter_save_verified"]
    OUT.mkdir(parents=True,exist_ok=False)
    rows,candidates=prepare_rows();sup.write(OUT/"train.json",rows)
    manifest={"status":"PREPARED","seed":SEED,"train_split_only":True,
        "hyphen_candidates":len(candidates),"repair_samples":32,"replay_samples":32,
        "colors":dict(__import__("collections").Counter(x["target"]["color"] for x in rows if x["source"]=="hyphen_color")),
        "train_sha256":sup.digest(OUT/"train.json"),
        "system_prompt_sha256":hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "parent_adapter_sha256":{p.name:sup.digest(p) for p in (SOURCE/"adapter").iterdir() if p.is_file()}}
    sup.write(OUT/"manifest.json",manifest);atomic(OUT/"status.json",{"status":"LOADING"})
    os.environ["HF_HOME"]=str(ROOT/".cache/huggingface");os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,BitsAndBytesConfig,Qwen3_5ForCausalLM
    from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict
    from safetensors.torch import load_file
    torch.manual_seed(SEED);tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tokenizer.padding_side="left"
    prepared=[]
    for row in rows:
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"Description: "+row["description"]}]
        prefix=tokenizer.encode(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        answer=compact.encode(row["target"]);suffix=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
        assert compact.parse(answer)==(row["target"],None);prepared.append((row["sample_id"],prefix,suffix))
    model,loading=Qwen3_5ForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map={"":0},
        attn_implementation="sdpa",output_loading_info=True,quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs")
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
    model=PeftModel.from_pretrained(model,SOURCE/"adapter",is_trainable=True)
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad};initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
    optimizer=torch.optim.AdamW(list(trainable.values()),lr=2e-5);model.train();model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False});steps=[]
    with (OUT/"steps.jsonl").open("w",encoding="utf-8") as stream:
        for index,(sample_id,prefix,suffix) in enumerate(prepared,1):
            ids=torch.tensor([prefix+suffix],device="cuda");positions=torch.arange(len(prefix)-1,ids.shape[1]-1,device="cuda");labels=ids[:,len(prefix):]
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=positions).logits
                loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1))
            loss.backward();torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
            rec={"step":index,"sample_id":sample_id,"loss":loss.item(),"seconds":time.perf_counter()-started}
            steps.append(rec);stream.write(json.dumps(rec)+"\n");stream.flush()
            atomic(OUT/"status.json",{"status":"TRAINING","completed":index,"total":len(prepared)})
            if index%8==0:print("repair",index,"/",len(prepared),round(rec["loss"],4),flush=True)
    changed=sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in trainable.items());assert changed>0
    model.save_pretrained(OUT/"adapter",safe_serialization=True);saved=load_file(OUT/"adapter/adapter_model.safetensors");live=get_peft_model_state_dict(model)
    assert set(saved)==set(live) and all(torch.equal(saved[k],live[k].detach().cpu()) for k in saved)
    model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True
    special_rows=sup.read(SOURCE/"special_holdout.json")
    special_predictions,special_timing=compact_train.batched_generate(model,tokenizer,special_rows,"sample_id","SPECIAL_EVAL",system_prompt=SYSTEM_PROMPT)
    special_result,special_details=v2.score_special(special_rows,special_predictions)
    descriptions=sup.read(GENERAL/"descriptions.json");cases=sup.read(GENERAL/"validation.json")
    general_predictions,general_timing=compact_train.batched_generate(model,tokenizer,descriptions,"description_id","GENERAL_EVAL",system_prompt=SYSTEM_PROMPT)
    general_result,extraction,scored=v2.score_general(descriptions,cases,general_predictions)
    for name,value in (("special_predictions.json",special_predictions),("special_details.json",special_details),
        ("general_predictions.json",general_predictions),("general_extraction.json",extraction),("general_scored.json",scored)):sup.write(OUT/name,value)
    result={"training":{"steps":64,"seconds":sum(x["seconds"] for x in steps),"changed_tensors":changed},
        "special_holdout":special_result,"special_timing":special_timing,"general_regression":general_result,
        "general_timing":general_timing,"repair_manifest":manifest}
    sup.write(OUT/"results.json",result);passed=special_result["exact_conditions"]>=119 and general_result["candidate_correct"]==240
    sup.write(OUT/"complete.json",{"status":"PASS" if passed else "FAIL","adapter_save_verified":True,
        "special_exact":special_result["exact_conditions"],"general_correct":general_result["candidate_correct"]})
    atomic(OUT/"status.json",{"status":"COMPLETE" if passed else "FAILED_VALIDATION"})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    try:main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);atomic(OUT/"failure.json",{"error":str(error),"traceback":traceback.format_exc()});atomic(OUT/"status.json",{"status":"FAILED","error":str(error)});raise
