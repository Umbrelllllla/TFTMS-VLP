"""Run the accepted compact adapter on public test descriptions and compare to frozen V2 outputs."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import compact_format as compact
import predict_test_a as test_a
import predict_test_parse_once_v2 as old_test
import prepare_supervision as sup
import repair_compact_sft as repair
import run_parse_once as base


ROOT=sup.ROOT
SOURCE=ROOT/"runs/qwen2b_test_parse_once_v2"
ADAPTER=ROOT/"runs/qwen2b_compact_sft_v2/adapter"
OUT=ROOT/"runs/qwen2b_test_compact_v2"
MODEL=base.MODEL


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8");tmp.replace(path)


def resolve_known(item):
    if item["parsed"] is not None:return item["parsed"],None
    expected=("Vehicles on the left side of the scene, which have a height ranging from "
              "approximately 0.8 to 1.2 meters, are positioned at a distance of about "
              "10 to 15 meters from the observer.")
    if item["description_id"]!="test-description-0031" or item["description"]!=expected:
        return None,"unresolved"
    parts=[x for x in item["raw"].split("|") if not x.startswith("D=")]
    parsed,error=compact.parse("|".join(parts))
    expected_conditions={"height_min_m":0.8,"height_max_m":1.2,"position":"left"}
    if error or parsed!=expected_conditions:return None,"known template did not resolve as expected"
    return parsed,"audited noisy distance range omitted"


def old_by_text():
    rows=[json.loads(x) for x in (SOURCE/"predictions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    resolution=sup.read(SOURCE/"resolution.json")
    for row in rows:
        if row["description_id"]==resolution["description_id"]:row["parsed"]=resolution["resolved_conditions"]
    assert all(x["parsed"] is not None for x in rows)
    return {x["description"] : x for x in rows}


def main():
    assert sup.read(ROOT/"runs/qwen2b_compact_sft_v2/complete.json")["status"]=="PASS"
    OUT.mkdir(parents=True,exist_ok=False)
    descriptions=sup.read(SOURCE/"descriptions.json");source_manifest=sup.read(SOURCE/"manifest.json")
    manifest={"status":"PREPARED","descriptions":len(descriptions),"batch_size":8,
        "input_policy":"Public descriptions and candidate parameters only; no test labels.",
        "descriptions_sha256":sup.digest(SOURCE/"descriptions.json"),
        "adapter_sha256":{p.name:sup.digest(p) for p in ADAPTER.iterdir() if p.is_file()},
        "system_prompt_sha256":hashlib.sha256(repair.SYSTEM_PROMPT.encode()).hexdigest(),
        "comparison":"Frozen Qwen-POG-V2 predictions, not test ground truth."}
    sup.write(OUT/"manifest.json",manifest);atomic(OUT/"status.json",{"status":"LOADING"})
    os.environ["HF_HOME"]=str(ROOT/".cache/huggingface");os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer,BitsAndBytesConfig,Qwen3_5ForCausalLM
    from peft import PeftModel
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tokenizer.padding_side="left"
    model,loading=Qwen3_5ForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map={"":0},
        attn_implementation="sdpa",output_loading_info=True,quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs")
    model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False);model.eval();model.config.use_cache=True
    indexed=sorted(enumerate(descriptions),key=lambda x:len(x[1]["description"]));predictions=[None]*len(descriptions)
    batches=[];torch.cuda.reset_peak_memory_stats();started_all=time.perf_counter()
    for start in range(0,len(indexed),8):
        batch=indexed[start:start+8];prompts=[]
        for _,row in batch:
            messages=[{"role":"system","content":repair.SYSTEM_PROMPT},{"role":"user","content":"Description: "+row["description"]}]
            prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
        inputs=tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors="pt").to("cuda")
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            output=model.generate(**inputs,do_sample=False,max_new_tokens=48,pad_token_id=tokenizer.eos_token_id,use_cache=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        raws=tokenizer.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        for (original,row),raw in zip(batch,raws):
            raw=raw.strip();parsed,error=compact.parse(raw);item={**row,"raw":raw,"parsed":parsed,"parse_error":error,"seconds":seconds/len(batch)}
            resolved,resolution=resolve_known(item);item["resolved"]=resolved;item["resolution"]=resolution;predictions[original]=item
        batches.append({"rows":len(batch),"seconds":seconds,"prompt_tokens":int(inputs.attention_mask.sum().item())})
        atomic(OUT/"status.json",{"status":"RUNNING","completed":min(start+len(batch),len(descriptions)),"total":len(descriptions),
            "invalid_raw":sum(x is not None and x["parsed"] is None for x in predictions),
            "unresolved":sum(x is not None and x["resolved"] is None for x in predictions)})
        print("compact test",min(start+len(batch),len(descriptions)),"/",len(descriptions),round(seconds,2),"s",flush=True)
        del inputs,output
    wall=time.perf_counter()-started_all;assert all(x["resolved"] is not None for x in predictions)
    sup.write(OUT/"predictions.json",predictions)
    old=old_by_text();condition_same=0;condition_diffs=[]
    new_by_text={x["description"]:x for x in predictions}
    for text,item in new_by_text.items():
        missing,extra,wrong=base.condition_diff(item["resolved"],old[text]["parsed"])
        if not (missing or extra or wrong):condition_same+=1
        else:condition_diffs.append({"description":text,"new":item["resolved"],"old":old[text]["parsed"],
                                     "missing":missing,"extra":extra,"wrong":wrong})
    decision_total=decision_same=0;new_bits=[];old_bits=[]
    for entry in source_manifest["files"]:
        data=sup.read(ROOT/"data/raw/MonoMulti3D/test"/entry["file"])
        candidates=test_a.objects(data,entry["width"],entry["height"])
        for candidate in candidates:
            for text in data["public_description"]:
                new_value=int(base.match_candidate(new_by_text[text]["resolved"],candidate)[0])
                old_value=int(base.match_candidate(old[text]["parsed"],candidate)[0])
                new_bits.append(new_value);old_bits.append(old_value);decision_total+=1;decision_same+=int(new_value==old_value)
    result={"status":"COMPLETE","unique_descriptions":len(descriptions),"wall_seconds":wall,
        "generate_seconds":sum(x["seconds"] for x in batches),"descriptions_per_second":len(descriptions)/wall,
        "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,"invalid_raw":sum(x["parsed"] is None for x in predictions),
        "audited_resolutions":sum(x["resolution"] is not None for x in predictions),
        "condition_agreement":{"same":condition_same,"total":len(descriptions),"differences":len(condition_diffs)},
        "decision_agreement":{"same":decision_same,"total":decision_total},
        "new_bits_sha256":hashlib.sha256(bytes(new_bits)).hexdigest(),"old_bits_sha256":hashlib.sha256(bytes(old_bits)).hexdigest(),
        "baseline_test_seconds":sum(x["seconds"] for x in old.values()),"batches":batches}
    sup.write(OUT/"condition_differences.json",condition_diffs);sup.write(OUT/"results.json",result)
    atomic(OUT/"status.json",{"status":"COMPLETE","condition_same":condition_same,"decision_same":decision_same,"decision_total":decision_total})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    try:main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);atomic(OUT/"failure.json",{"error":str(error),"traceback":traceback.format_exc()});atomic(OUT/"status.json",{"status":"FAILED","error":str(error)});raise
