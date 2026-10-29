"""Freeze and evaluate a second untouched 120-image holdout for the flat-JSON adapter."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import time
import traceback

import prepare_ab
import prepare_supervision as sup
import run_parse_once as base


ROOT = sup.ROOT
OUT = ROOT / "runs/qwen2b_parse_once_sft_holdout_v1"
SFT = ROOT / "runs/qwen2b_parse_once_sft_v1"
SEED = 20261021


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    split = sup.read(ROOT / "runs/20261015_qwen2b_input_v1/split_manifest.json")
    validation_files = set(split["validation"])
    excluded = {x["source_file"] for x in sup.read(ROOT / "runs/qwen2b_supervision_ab_expanded_v1/validation.json")}
    excluded |= {x["source_file"] for x in sup.read(ROOT / "runs/qwen2b_supervision_ab_v1/validation.json")}
    train_sft_files = {x["source_file"] for x in sup.read(SFT / "train.json")}
    spec = importlib.util.spec_from_file_location("feature_prep", ROOT / "runs/20261015_qwen2b_input_v1/prepare.py")
    feature = importlib.util.module_from_spec(spec); spec.loader.exec_module(feature)
    pools = collections.defaultdict(list); rejected = collections.Counter()
    for record in sup.read(ROOT / "runs/20261015_qwen2b_data_audit/training_records.json"):
        if record["file"] not in validation_files or record["file"] in excluded:
            continue
        for annotation_index, annotation in enumerate(record["annotations"]):
            try:
                if prepare_ab.extract(annotation["text"]) != prepare_ab.normalized(annotation["properties"]):
                    raise ValueError("text_property_disagreement")
                positives = set(annotation["positive_indices"])
                negatives = set(range(len(record["objects"]))) - positives
                if not positives or not negatives:
                    raise ValueError("no_positive_or_negative")
                cases = []
                for object_index, obj in enumerate(record["objects"]):
                    case = {"input": {"description": annotation["text"],
                                      "candidate": feature.features(obj, record["width"], record["height"])},
                            "target": int(object_index in positives), "properties_for_audit_only": annotation["properties"],
                            "source_file": record["file"], "camera": record["camera"],
                            "annotation_index": annotation_index, "object_index": object_index}
                    case["structured_target"] = sup.target(case); cases.append(case)
                pools[record["file"]].append(cases)
            except (ValueError, AssertionError) as error:
                rejected[str(error) or "membership_disagreement"] += 1
    rng = random.Random(SEED)
    files = sorted(pools); rng.shuffle(files)
    assert len(files) >= 120
    selected = []
    for filename in files[:120]:
        cases = rng.choice(pools[filename])
        positive = rng.choice([x for x in cases if x["target"]])
        negatives = [x for x in cases if not x["target"]]; rng.shuffle(negatives)
        hard = min(negatives, key=lambda x: sum(a["satisfied"] is False for a in x["structured_target"]["attributes"].values()))
        selected.extend([positive, hard])
    rng.shuffle(selected)
    for index, case in enumerate(selected): case["input"]["pair_id"] = f"holdout-{index:03d}"
    assert len(selected) == 240 and sum(x["target"] for x in selected) == 120
    assert not {x["source_file"] for x in selected} & excluded
    assert not {x["source_file"] for x in selected} & train_sft_files
    sup.write(OUT / "validation.json", selected)
    grouped = collections.defaultdict(list)
    for case in selected: grouped[base.description_key(case)].append(case)
    assert len(grouped) == 120
    descriptions = []
    for index, (key, rows) in enumerate(sorted(grouped.items()), 1):
        assert len(rows) == 2 and {x["target"] for x in rows} == {0, 1}
        gold = base.gold_flat(rows[0]); assert all(base.gold_flat(x) == gold for x in rows)
        descriptions.append({"description_id": f"holdout-description-{index:03d}", "source_key": key,
                             "description": rows[0]["input"]["description"], "gold_conditions": gold,
                             "pair_ids": [x["input"]["pair_id"] for x in rows]})
    sup.write(OUT / "descriptions.json", descriptions)
    train_texts = {x["description"] for x in sup.read(SFT / "train.json")}
    normalize = lambda text: re.sub(r"\d+(?:\.\d+)?", "<NUM>", text.lower())
    train_templates = {normalize(x) for x in train_texts}
    manifest = {"status": "PREPARED", "seed": SEED, "images": 120, "descriptions": 120,
                "candidate_pairs": 240, "positive_pairs": 120, "eligible_remaining_images": len(files),
                "excluded_prior_validation_images": len(excluded), "train_file_overlap": 0,
                "prior_validation_file_overlap": 0,
                "exact_training_text_overlap": sum(x["description"] in train_texts for x in descriptions),
                "number_normalized_template_overlap": sum(normalize(x["description"]) in train_templates for x in descriptions),
                "validation_sha256": sup.digest(OUT / "validation.json"),
                "descriptions_sha256": sup.digest(OUT / "descriptions.json"),
                "adapter_sha256": {p.name: sup.digest(p) for p in (SFT / "adapter").iterdir() if p.is_file()},
                "system_prompt_sha256": hashlib.sha256(base.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "limits": "New filtered development holdout; not official test. Same validation camera groups as prior development data."}
    sup.write(OUT / "manifest.json", manifest)
    atomic(OUT / "status.json", {"status": "PREPARED", "completed_descriptions": 0, "total_descriptions": 120})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def run():
    manifest = sup.read(OUT / "manifest.json")
    assert sup.digest(OUT / "validation.json") == manifest["validation_sha256"]
    assert sup.digest(OUT / "descriptions.json") == manifest["descriptions_sha256"]
    for name, digest in manifest["adapter_sha256"].items(): assert sup.digest(SFT / "adapter" / name) == digest
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from peft import PeftModel, prepare_model_for_kbit_training
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(base.MODEL, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, SFT / "adapter", is_trainable=False)
    model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache = True
    predictions = []
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(sup.read(OUT / "descriptions.json"), 1):
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
            item = {"description_id": row["description_id"], "raw": raw, "parsed": parsed, "parse_error": error,
                    "seconds": time.perf_counter() - started, "generated_tokens": len(generated),
                    "hit_token_cap": len(generated) >= 128 and generated[-1].item() != tokenizer.eos_token_id}
            predictions.append(item); stream.write(json.dumps(item, ensure_ascii=False) + "\n"); stream.flush()
            elapsed = sum(x["seconds"] for x in predictions)
            atomic(OUT / "status.json", {"status": "RUNNING", "completed_descriptions": index, "total_descriptions": 120,
                   "invalid_outputs": sum(x["parsed"] is None for x in predictions),
                   "mean_seconds_per_description": elapsed / index,
                   "remaining_minutes_estimate": (120-index)*elapsed/index/60})
            print(index, "/120", round(item["seconds"], 2), "s", "valid" if parsed else error, flush=True)
    atomic(OUT / "status.json", {"status": "INFERENCE_COMPLETE", "completed_descriptions": 120, "total_descriptions": 120})


def report():
    descriptions = sup.read(OUT / "descriptions.json"); cases = sup.read(OUT / "validation.json")
    predictions = [json.loads(x) for x in (OUT / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(descriptions) == len(predictions) == 120
    by_prediction = {x["description_id"]: x for x in predictions}
    by_source = {x["source_key"]: x for x in descriptions}
    extraction = []; field_errors = collections.Counter()
    for d in descriptions:
        p = by_prediction[d["description_id"]]
        if p["parsed"] is None:
            extraction.append({**d, "raw": p["raw"], "parsed": None, "parse_error": p["parse_error"], "exact": False})
            field_errors["invalid_descriptions"] += 1; continue
        missing, extra, wrong = base.condition_diff(p["parsed"], d["gold_conditions"])
        for key in missing: field_errors[f"missing::{key}"] += 1
        for key in extra: field_errors[f"extra::{key}"] += 1
        for key in wrong: field_errors[f"wrong::{key}"] += 1
        extraction.append({**d, "raw": p["raw"], "parsed": p["parsed"], "missing_keys": missing,
                           "extra_keys": extra, "wrong_value_keys": wrong, "exact": not (missing or extra or wrong)})
    exact = {x["description_id"]: x["exact"] for x in extraction}; counts = collections.Counter(); scored = []; oracle = 0
    for case in cases:
        d = by_source[base.description_key(case)]; p = by_prediction[d["description_id"]]
        gold_pred, _ = base.match_candidate(d["gold_conditions"], case["input"]["candidate"])
        oracle += int(int(gold_pred) == case["target"])
        if p["parsed"] is None:
            pred = None; checks = None; correct = False; counts["unresolved"] += 1
            if case["target"]: counts["fn"] += 1
        else:
            value, checks = base.match_candidate(p["parsed"], case["input"]["candidate"])
            pred = int(value); correct = pred == case["target"]
            if pred and case["target"]: counts["tp"] += 1
            elif pred: counts["fp"] += 1
            elif case["target"]: counts["fn"] += 1
            else: counts["tn"] += 1
        scored.append({"pair_id": case["input"]["pair_id"], "description_id": d["description_id"],
                       "gold": case["target"], "prediction": pred, "correct": correct, "checks": checks})
    assert oracle == 240
    correct = sum(x["correct"] for x in scored); precision = counts["tp"]/(counts["tp"]+counts["fp"]) if counts["tp"]+counts["fp"] else 0
    recall = counts["tp"]/(counts["tp"]+counts["fn"]) if counts["tp"]+counts["fn"] else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
    result = {"description_extraction": {"valid": sum(x["parsed"] is not None for x in predictions), "total": 120,
              "exact_conditions": sum(x["exact"] for x in extraction), "field_errors": dict(sorted(field_errors.items())),
              "generated_token_cap_hits": sum(x["hit_token_cap"] for x in predictions)},
              "candidate_decisions": {"correct": correct, "total": 240, "accuracy": correct/240, "tp": counts["tp"],
              "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"], "unresolved": counts["unresolved"],
              "precision": precision, "recall": recall, "f1": f1},
              "program_verification": {"gold_condition_oracle_correct": oracle, "total": 240},
              "timing": {"model_inference_seconds": sum(x["seconds"] for x in predictions),
                         "mean_seconds_per_description": sum(x["seconds"] for x in predictions)/120}}
    sup.write(OUT / "extraction_details.json", extraction); sup.write(OUT / "scored_pairs.json", scored)
    sup.write(OUT / "errors.json", [x for x in scored if not x["correct"]]); sup.write(OUT / "results.json", result)
    sup.write(OUT / "complete.json", {"status": "PASS", "final_correct": correct, "adapter_unchanged": True})
    atomic(OUT / "status.json", {"status": "COMPLETE", "final_correct": correct, "total": 240})
    if (OUT / "failure.json").exists(): (OUT / "failure.json").unlink()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["prepare", "run", "report", "pipeline"])
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "run": run()
    elif args.action == "report": report()
    else: prepare(); run(); report()


if __name__ == "__main__":
    try: main()
    except Exception as error:
        if OUT.exists():
            atomic(OUT / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
            atomic(OUT / "status.json", {"status": "FAILED", "error": str(error)})
        raise
