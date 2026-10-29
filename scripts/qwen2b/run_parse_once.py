"""Parse each description once, then match candidates with deterministic code.

The model receives the original description only.  Candidate features and gold
annotations are used after generation, exclusively by the evaluator.
"""
import argparse
import collections
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/qwen2b_supervision_ab_expanded_v1"
OUT = ROOT / "runs/qwen2b_parse_once_v1"
MODEL = ROOT / "weights/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc"
VALIDATION_SHA256 = "6845c5daa10d8e8e2e06dd9fc33110f70deb6d7d9c21ceb337a735044492df73"

TEXT_KEYS = {"category", "color", "position", "facing", "occlusion"}
NUMBER_KEYS = {
    "distance_max_m",
    "length_min_m", "length_max_m",
    "width_min_m", "width_max_m",
    "height_min_m", "height_max_m",
}
ALLOWED_KEYS = TEXT_KEYS | NUMBER_KEYS
VOCAB = {
    "category": {"car", "van", "truck", "bus"},
    "position": {"top left", "top", "top right", "left", "center", "right",
                 "bottom left", "bottom", "bottom right"},
    "facing": {"facing me", "facing away from me", "side facing me"},
    "occlusion": {"no occluded", "partially occluded"},
}

SYSTEM_PROMPT = """Extract only conditions explicitly stated in the English description.
Return exactly one JSON object on one line. Do not explain.

Allowed keys:
category, color, position, facing, occlusion,
distance_max_m, length_min_m, length_max_m, width_min_m, width_max_m,
height_min_m, height_max_m.

Rules:
1. Every returned key must have direct words or numbers in the description as evidence.
2. Omit every unstated key. Do not output null and do not add defaults.
3. Generic words "vehicle" and "vehicles" do not specify category. Never turn them into car.
4. category values: car, van, truck, bus.
5. position values: top left, top, top right, left, center, right, bottom left, bottom, bottom right.
6. facing values: facing me, facing away from me, side facing me. "Sideways towards me" means side facing me.
7. occlusion values: no occluded, partially occluded. "Not occluded" means no occluded.
8. "within D meters" becomes distance_max_m D.
9. A size range A to B uses both the corresponding min and max keys. Copy the exact numbers.
10. All conditions in the description must be returned together.

Examples:
Description: Vehicles are white and have a length of approximately 3.2 to 3.8 meters.
Output: {"color":"white","length_min_m":3.2,"length_max_m":3.8}
Description: Cars in the top right are not occluded.
Output: {"category":"car","position":"top right","occlusion":"no occluded"}"""


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def description_key(case):
    return f"{case['source_file']}::{case['annotation_index']}"


def gold_flat(case):
    result = {}
    for name, item in case["structured_target"]["attributes"].items():
        if not item["relevant"]:
            continue
        condition = item["condition"]
        if name in TEXT_KEYS:
            result[name] = condition["value"]
        elif name == "distance":
            result["distance_max_m"] = condition["value"]
        else:
            result[f"{name}_min_m"] = condition["min"]
            result[f"{name}_max_m"] = condition["max"]
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = read(SOURCE / "manifest.json")
    assert digest(SOURCE / "validation.json") == VALIDATION_SHA256
    assert source_manifest["validation_sha256"] == VALIDATION_SHA256
    cases = read(SOURCE / "validation.json")
    grouped = collections.defaultdict(list)
    for case in cases:
        grouped[description_key(case)].append(case)
    assert len(cases) == 240 and len(grouped) == 120
    descriptions = []
    for index, (key, rows) in enumerate(sorted(grouped.items()), 1):
        assert len(rows) == 2
        assert {r["target"] for r in rows} == {0, 1}
        assert len({r["input"]["description"] for r in rows}) == 1
        golds = [gold_flat(r) for r in rows]
        assert golds[0] == golds[1]
        descriptions.append({
            "description_id": f"description-{index:03d}",
            "source_key": key,
            "description": rows[0]["input"]["description"],
            "gold_conditions": golds[0],
            "pair_ids": [r["input"]["pair_id"] for r in rows],
        })
    write(OUT / "descriptions.json", descriptions)
    manifest = {
        "status": "PREPARED",
        "method": "base Qwen3.5-2B NF4 parses each description once; deterministic code matches candidates",
        "model_input": "description only",
        "descriptions": len(descriptions),
        "candidate_pairs": len(cases),
        "positive_pairs": sum(c["target"] for c in cases),
        "source_validation_sha256": VALIDATION_SHA256,
        "descriptions_sha256": digest(OUT / "descriptions.json"),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "max_new_tokens": 128,
        "decoding": "greedy; thinking disabled",
        "strict_output_policy": "No repair, code-fence removal, key inference, or value normalization",
        "limits": "Filtered 120-image development subset with one positive and one hard negative per description; not official test.",
    }
    write(OUT / "manifest.json", manifest)
    write(OUT / "status.json", {"status": "PREPARED", "completed_descriptions": 0, "total_descriptions": 120})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def reject_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate key: {key}")
        obj[key] = value
    return obj


def parse_strict(raw):
    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid constant: {x}")))
        if not isinstance(value, dict):
            raise ValueError("root is not an object")
        unknown = set(value) - ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unknown keys: {sorted(unknown)}")
        if not value:
            raise ValueError("empty conditions")
        for key, item in value.items():
            if key in TEXT_KEYS:
                if not isinstance(item, str) or not item:
                    raise ValueError(f"{key} must be a non-empty string")
                if key in VOCAB and item not in VOCAB[key]:
                    raise ValueError(f"invalid {key}: {item}")
            else:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError(f"{key} must be a number")
                number = Decimal(str(item))
                if not number.is_finite() or number < 0:
                    raise ValueError(f"invalid number for {key}: {item}")
        for dimension in ("length", "width", "height"):
            lo, hi = f"{dimension}_min_m", f"{dimension}_max_m"
            if (lo in value) != (hi in value):
                raise ValueError(f"{dimension} range needs both min and max")
            if lo in value and Decimal(str(value[lo])) > Decimal(str(value[hi])):
                raise ValueError(f"{dimension} min exceeds max")
        return value, None
    except (json.JSONDecodeError, ValueError, InvalidOperation) as error:
        return None, str(error)


def run():
    assert (OUT / "manifest.json").exists(), "Run prepare first"
    manifest = read(OUT / "manifest.json")
    assert digest(SOURCE / "validation.json") == manifest["source_validation_sha256"]
    assert digest(OUT / "descriptions.json") == manifest["descriptions_sha256"]
    assert hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest() == manifest["system_prompt_sha256"]
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from download import verify

    for entry in read(MODEL / "source_manifest.json")["siblings"]:
        if entry["rfilename"] != ".gitattributes":
            assert verify(MODEL / entry["rfilename"], entry)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", output_loading_info=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs")
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = True
    assert not any(p.requires_grad for p in model.parameters())

    descriptions = read(OUT / "descriptions.json")
    prediction_path = OUT / "predictions.jsonl"
    completed = []
    if prediction_path.exists():
        completed = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done_ids = {row["description_id"] for row in completed}
    assert len(done_ids) == len(completed)
    write(OUT / "status.json", {"status": "RUNNING", "completed_descriptions": len(completed),
                                 "total_descriptions": len(descriptions)})
    mode = "a" if completed else "w"
    with prediction_path.open(mode, encoding="utf-8") as stream:
        for item in descriptions:
            if item["description_id"] in done_ids:
                continue
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Description: " + item["description"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            assert inputs.input_ids.shape[1] + manifest["max_new_tokens"] <= 1536
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(**inputs, do_sample=False, max_new_tokens=manifest["max_new_tokens"],
                                        pad_token_id=tokenizer.eos_token_id, use_cache=True)
            torch.cuda.synchronize()
            generated = output[0, inputs.input_ids.shape[1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
            parsed, error = parse_strict(raw)
            row = {"description_id": item["description_id"], "raw": raw, "parsed": parsed,
                   "parse_error": error, "seconds": time.perf_counter() - started,
                   "generated_tokens": len(generated),
                   "hit_token_cap": len(generated) >= manifest["max_new_tokens"] and generated[-1].item() != tokenizer.eos_token_id}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            completed.append(row)
            elapsed = sum(x["seconds"] for x in completed)
            remaining = (len(descriptions) - len(completed)) * elapsed / len(completed)
            write(OUT / "status.json", {"status": "RUNNING", "completed_descriptions": len(completed),
                                         "total_descriptions": len(descriptions), "last_description_id": item["description_id"],
                                         "invalid_outputs": sum(x["parsed"] is None for x in completed),
                                         "mean_seconds_per_description": elapsed / len(completed),
                                         "remaining_minutes_estimate": remaining / 60})
            print(len(completed), "/", len(descriptions), round(row["seconds"], 2), "s", "valid" if parsed else error, flush=True)
    write(OUT / "status.json", {"status": "INFERENCE_COMPLETE", "completed_descriptions": len(completed),
                                 "total_descriptions": len(descriptions),
                                 "invalid_outputs": sum(x["parsed"] is None for x in completed),
                                 "inference_seconds": sum(x["seconds"] for x in completed)})


def decimal_equal(a, b):
    return Decimal(str(a)) == Decimal(str(b))


def condition_diff(predicted, gold):
    missing = sorted(set(gold) - set(predicted))
    extra = sorted(set(predicted) - set(gold))
    wrong = sorted(k for k in set(gold) & set(predicted)
                   if (not decimal_equal(predicted[k], gold[k]) if k in NUMBER_KEYS else predicted[k] != gold[k]))
    return missing, extra, wrong


def match_candidate(conditions, candidate):
    checks = {}
    paths = {
        "category": candidate["type"], "color": candidate["color"],
        "position": candidate["image_region"], "facing": candidate["facing_convention"],
        "occlusion": candidate["occlusion"],
    }
    for key, expected in conditions.items():
        if key in TEXT_KEYS:
            checks[key] = paths[key] == expected
        elif key == "distance_max_m":
            checks[key] = Decimal(str(candidate["distance_m"])) <= Decimal(str(expected))
        else:
            dimension, bound, _ = key.split("_", 2)
            actual = Decimal(str(candidate["dimensions_m"][dimension]))
            limit = Decimal(str(expected))
            checks[key] = actual >= limit if bound == "min" else actual <= limit
    return all(checks.values()), checks


def report():
    manifest = read(OUT / "manifest.json")
    assert digest(SOURCE / "validation.json") == manifest["source_validation_sha256"]
    descriptions = read(OUT / "descriptions.json")
    predictions = [json.loads(line) for line in (OUT / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(predictions) == len(descriptions) == 120
    by_id = {p["description_id"]: p for p in predictions}
    source_to_description = {d["source_key"]: d["description_id"] for d in descriptions}
    gold_by_id = {d["description_id"]: d["gold_conditions"] for d in descriptions}
    extraction_rows = []
    field_counts = collections.Counter()
    for description in descriptions:
        p = by_id[description["description_id"]]
        if p["parsed"] is None:
            extraction_rows.append({**description, "raw": p["raw"], "parsed": None,
                                    "parse_error": p["parse_error"], "exact": False})
            field_counts["invalid_descriptions"] += 1
            continue
        missing, extra, wrong = condition_diff(p["parsed"], description["gold_conditions"])
        for key in missing: field_counts[f"missing::{key}"] += 1
        for key in extra: field_counts[f"extra::{key}"] += 1
        for key in wrong: field_counts[f"wrong::{key}"] += 1
        extraction_rows.append({**description, "raw": p["raw"], "parsed": p["parsed"],
                                "parse_error": None, "missing_keys": missing, "extra_keys": extra,
                                "wrong_value_keys": wrong, "exact": not (missing or extra or wrong)})

    cases = read(SOURCE / "validation.json")
    scored = []
    oracle_correct = 0
    counts = collections.Counter()
    subgroup = collections.defaultdict(lambda: collections.Counter(total=0, correct=0))
    exact_by_id = {x["description_id"]: x["exact"] for x in extraction_rows}
    program_started = time.perf_counter()
    for case in cases:
        source_key = description_key(case)
        did = source_to_description[source_key]
        prediction = by_id[did]
        oracle_value, _ = match_candidate(gold_by_id[did], case["input"]["candidate"])
        oracle_correct += int(int(oracle_value) == case["target"])
        if prediction["parsed"] is None:
            group = "invalid_output"
            row = {"pair_id": case["input"]["pair_id"], "description_id": did, "gold": case["target"],
                   "prediction": None, "correct": False, "checks": None, "reason": "invalid model output"}
            counts["unresolved"] += 1
            if case["target"] == 1: counts["fn"] += 1
        else:
            group = "exact_conditions" if exact_by_id[did] else "valid_but_inexact_conditions"
            value, checks = match_candidate(prediction["parsed"], case["input"]["candidate"])
            pred = int(value)
            correct = pred == case["target"]
            row = {"pair_id": case["input"]["pair_id"], "description_id": did, "gold": case["target"],
                   "prediction": pred, "correct": correct, "checks": checks}
            if pred == 1 and case["target"] == 1: counts["tp"] += 1
            elif pred == 1: counts["fp"] += 1
            elif case["target"] == 1: counts["fn"] += 1
            else: counts["tn"] += 1
        scored.append(row)
        subgroup[group]["total"] += 1
        subgroup[group]["correct"] += int(row["correct"])
    program_seconds = time.perf_counter() - program_started
    correct = sum(x["correct"] for x in scored)
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    inference_seconds = sum(p["seconds"] for p in predictions)
    a_b = read(SOURCE / "results.json")
    result = {
        "method": "parse description once, then deterministic candidate matching",
        "description_extraction": {
            "valid": sum(p["parsed"] is not None for p in predictions), "total": len(predictions),
            "exact_conditions": sum(x["exact"] for x in extraction_rows),
            "field_errors": dict(sorted(field_counts.items())),
            "generated_token_cap_hits": sum(p["hit_token_cap"] for p in predictions),
        },
        "candidate_decisions": {"correct": correct, "total": len(scored), "accuracy": correct / len(scored),
                                "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"],
                                "unresolved": counts["unresolved"], "precision": precision, "recall": recall, "f1": f1},
        "program_verification": {"gold_condition_oracle_correct": oracle_correct, "total": len(cases),
                                 "subgroups": {k: dict(v) for k, v in subgroup.items()}},
        "timing": {"model_calls": len(predictions), "model_inference_seconds": inference_seconds,
                   "mean_seconds_per_description": inference_seconds / len(predictions),
                   "program_matching_seconds": program_seconds,
                   "total_pipeline_seconds": inference_seconds + program_seconds,
                   "A_direct_240_calls_seconds": a_b["A"]["inference_seconds"],
                   "B_structured_240_calls_seconds": a_b["B"]["inference_seconds"],
                   "speedup_vs_A": a_b["A"]["inference_seconds"] / (inference_seconds + program_seconds),
                   "speedup_vs_B": a_b["B"]["inference_seconds"] / (inference_seconds + program_seconds)},
        "strict_scoring_note": "Invalid model output leaves both candidate decisions unresolved and both count as incorrect; its positive is also counted as FN.",
    }
    assert oracle_correct == len(cases), "Deterministic matcher disagrees with gold labels"
    write(OUT / "extraction_details.json", extraction_rows)
    write(OUT / "scored_pairs.json", scored)
    write(OUT / "errors.json", [x for x in scored if not x["correct"]])
    write(OUT / "results.json", result)
    lines = [
        "# 描述只解析一次：正式验证结果", "",
        "模型只读取120条原始英文描述；候选参数、标注属性和答案没有进入模型输入。模型输出条件JSON后，固定程序判断240个候选。", "",
        "|指标|结果|", "|---|---:|",
        f"|合法JSON|{result['description_extraction']['valid']}/120|",
        f"|条件完全正确|{result['description_extraction']['exact_conditions']}/120|",
        f"|最终判断正确|{correct}/240 ({correct/240:.2%})|",
        f"|误选 FP|{counts['fp']}|", f"|漏选 FN|{counts['fn']}|",
        f"|无法判断|{counts['unresolved']}|", f"|F1|{f1:.4f}|",
        f"|标准条件送入同一程序|{oracle_correct}/240|",
        f"|模型推理时间|{inference_seconds:.1f}秒（120次）|",
        f"|程序比较时间|{program_seconds:.6f}秒（240次）|",
        f"|相对A总耗时|{result['timing']['speedup_vs_A']:.2f}倍|",
        f"|相对B总耗时|{result['timing']['speedup_vs_B']:.2f}倍|", "",
        "这里的“条件完全正确”要求键和值都与标注一致；多加、漏掉或写错任何条件都算错。程序比较使用原始浮点数和闭区间，不使用容差。", "",
        "验证集是既有的筛选开发子集：120张图片，每条描述配一个正候选和一个困难负候选。它不是官方test，不能代表完整排行榜成绩。",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(OUT / "status.json", {"status": "COMPLETE", "completed_descriptions": 120, "total_descriptions": 120})
    if (OUT / "failure.json").exists():
        (OUT / "failure.json").unlink()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "report", "pipeline"])
    args = parser.parse_args()
    try:
        if args.action == "prepare": prepare()
        elif args.action == "run": run()
        elif args.action == "report": report()
        else:
            if not (OUT / "manifest.json").exists(): prepare()
            run()
            report()
    except Exception as error:
        if OUT.exists():
            write(OUT / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
            write(OUT / "status.json", {"status": "FAILED", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
