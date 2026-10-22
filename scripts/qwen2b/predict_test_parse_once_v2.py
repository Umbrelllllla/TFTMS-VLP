"""Resumable V2 parse-once inference and official test ZIP generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

import predict_test_a as test_a
import prepare_supervision as sup
import run_parse_once as parse_once


ROOT = sup.ROOT
OUT = ROOT / "runs/qwen2b_test_parse_once_v2"
RAW = ROOT / "data/raw/MonoMulti3D/test"
ADAPTER = ROOT / "runs/qwen2b_parse_once_sft_v2/adapter"
V2 = ROOT / "runs/qwen2b_parse_once_sft_v2"
MODEL = parse_once.MODEL


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def adapter_hashes():
    return {p.name: sup.digest(p) for p in ADAPTER.iterdir() if p.is_file()}


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "manifest.json").exists(), "Already prepared; use run to resume"
    assert sup.read(V2 / "complete.json")["status"] == "PASS"
    entries = []
    all_descriptions = set()
    for path in sorted(RAW.glob("*.json")):
        data = sup.read(path)
        descriptions = data["public_description"]
        assert len(descriptions) == 3 and all(isinstance(x, str) and x for x in descriptions)
        image = ROOT / "data/raw/img/test" / (path.stem + ".jpg")
        width, height = test_a.image_size(image)
        candidates = test_a.objects(data, width, height)
        assert candidates and len(candidates) == len(data["test_data"])
        all_descriptions.update(descriptions)
        entries.append({
            "file": path.name,
            "sha256": sup.digest(path),
            "width": width,
            "height": height,
            "rows": len(candidates),
            "descriptions": descriptions,
        })
    assert len(entries) == 300
    descriptions = [
        {"description_id": f"test-description-{i:04d}", "description": text}
        for i, text in enumerate(sorted(all_descriptions), 1)
    ]
    atomic(OUT / "descriptions.json", descriptions)
    page = ROOT / "runs/qwen2b_test_A/task_page.html"
    assert page.exists()
    (OUT / "task_page.html").write_bytes(page.read_bytes())
    manifest = {
        "status": "PREPARED",
        "method": "V2 adapter parses each unique description once; deterministic code matches all candidates",
        "files": entries,
        "file_count": len(entries),
        "description_occurrences": len(entries) * 3,
        "unique_descriptions": len(descriptions),
        "candidates": sum(x["rows"] for x in entries),
        "output_bits": sum(x["rows"] * 3 for x in entries),
        "adapter_sha256": adapter_hashes(),
        "feature_script_sha256": sup.digest(test_a.FEATURE_SCRIPT),
        "matcher_script_sha256": sup.digest(ROOT / "scripts/qwen2b/run_parse_once.py"),
        "descriptions_sha256": sup.digest(OUT / "descriptions.json"),
        "system_prompt_sha256": hashlib.sha256(parse_once.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "max_new_tokens": 128,
        "do_sample": False,
        "enable_thinking": False,
        "format": "ZIP contains result/<input-stem>.txt; N rows; 3 space-separated bits",
        "input_policy": "Only public descriptions and candidate parameters; no answer labels",
        "invalid_output_policy": "Preserve raw output and withhold ZIP if any description is unresolved",
    }
    atomic(OUT / "manifest.json", manifest)
    atomic(OUT / "status.json", {
        "status": "PREPARED", "completed_descriptions": 0,
        "total_descriptions": len(descriptions),
    })
    print(json.dumps({k: manifest[k] for k in (
        "file_count", "description_occurrences", "unique_descriptions",
        "candidates", "output_bits", "format")}, ensure_ascii=False, indent=2))


def verify_sources(manifest):
    assert {p.name for p in RAW.glob("*.json")} == {x["file"] for x in manifest["files"]}
    assert sup.digest(test_a.FEATURE_SCRIPT) == manifest["feature_script_sha256"]
    assert sup.digest(ROOT / "scripts/qwen2b/run_parse_once.py") == manifest["matcher_script_sha256"]
    assert sup.digest(OUT / "descriptions.json") == manifest["descriptions_sha256"]
    assert hashlib.sha256(parse_once.SYSTEM_PROMPT.encode("utf-8")).hexdigest() == manifest["system_prompt_sha256"]
    assert adapter_hashes() == manifest["adapter_sha256"]
    for entry in manifest["files"]:
        assert sup.digest(RAW / entry["file"]) == entry["sha256"]


def result_text(records, rows):
    assert len(records) == rows * 3
    for k, record in enumerate(records):
        assert record["row"] == k // 3 and record["column"] == k % 3
        assert type(record["prediction"]) is int and record["prediction"] in (0, 1)
    return "".join(
        " ".join(str(records[i * 3 + j]["prediction"]) for j in range(3)) + "\n"
        for i in range(rows)
    )


def build_results(manifest, by_text):
    (OUT / "decisions").mkdir(exist_ok=True)
    (OUT / "result").mkdir(exist_ok=True)
    for entry in manifest["files"]:
        data = sup.read(RAW / entry["file"])
        candidates = test_a.objects(data, entry["width"], entry["height"])
        records = []
        for row, candidate in enumerate(candidates):
            for column, description in enumerate(data["public_description"]):
                prediction = by_text[description]
                assert prediction["parsed"] is not None
                matched, checks = parse_once.match_candidate(prediction["parsed"], candidate)
                records.append({
                    "row": row,
                    "column": column,
                    "description_id": prediction["description_id"],
                    "prediction": int(matched),
                    "checks": checks,
                })
        stem = Path(entry["file"]).stem
        atomic(OUT / "decisions" / (stem + ".json"), records)
        text = result_text(records, entry["rows"])
        tmp = OUT / "result" / (stem + ".txt.tmp")
        tmp.write_text(text, encoding="ascii")
        tmp.replace(OUT / "result" / (stem + ".txt"))


def pack(manifest):
    verify_sources(manifest)
    expected = {"result/" + Path(x["file"]).stem + ".txt" for x in manifest["files"]}
    actual = {"result/" + p.name for p in (OUT / "result").glob("*.txt")}
    assert actual == expected
    contents = {}
    for entry in manifest["files"]:
        stem = Path(entry["file"]).stem
        decisions = sup.read(OUT / "decisions" / (stem + ".json"))
        text = result_text(decisions, entry["rows"])
        path = OUT / "result" / (stem + ".txt")
        assert path.read_text(encoding="ascii") == text
        contents["result/" + path.name] = text.encode("ascii")
    tmp = OUT / "submission_parse_once_v2.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    with zipfile.ZipFile(tmp) as archive:
        assert set(archive.namelist()) == set(contents)
        assert len(archive.namelist()) == 300
        assert archive.testzip() is None
        assert all(archive.read(name) == content for name, content in contents.items())
    final = OUT / "submission_parse_once_v2.zip"
    tmp.replace(final)
    atomic(OUT / "complete.json", {
        "status": "PASS",
        "files": 300,
        "unique_model_calls": manifest["unique_descriptions"],
        "output_bits": manifest["output_bits"],
        "zip_sha256": sup.digest(final),
        "source_and_adapter_unchanged": True,
        "all_answers_from_v2_and_fixed_matcher": True,
    })
    atomic(OUT / "status.json", {
        "status": "COMPLETE", "files": 300,
        "output_bits": manifest["output_bits"], "zip": final.name,
    })


def run():
    import msvcrt
    OUT.mkdir(exist_ok=True)
    lock = (OUT / "run.lock").open("a+b")
    lock.seek(0)
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    manifest = sup.read(OUT / "manifest.json")
    verify_sources(manifest)
    descriptions = sup.read(OUT / "descriptions.json")
    prediction_path = OUT / "predictions.jsonl"
    completed = []
    if prediction_path.exists():
        completed = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {x["description_id"] for x in completed}
    assert len(done) == len(completed)
    if len(completed) == len(descriptions) and not any(x["parsed"] is None for x in completed):
        by_text = {x["description"] : x for x in completed}
        build_results(manifest, by_text)
        pack(manifest)
        print("TEST_PARSE_ONCE_V2_COMPLETE", flush=True)
        return

    atomic(OUT / "status.json", {
        "status": "LOADING", "completed_descriptions": len(completed),
        "total_descriptions": len(descriptions),
    })
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from peft import PeftModel, prepare_model_for_kbit_training
    from download import verify

    for item in sup.read(MODEL / "source_manifest.json")["siblings"]:
        if item["rfilename"] != ".gitattributes":
            assert verify(MODEL / item["rfilename"], item)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", output_loading_info=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys")
    assert not loading.get("mismatched_keys")
    assert not loading.get("error_msgs")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    assert not any(p.requires_grad for p in model.parameters())

    def predict(text):
        messages = [
            {"role": "system", "content": parse_once.SYSTEM_PROMPT},
            {"role": "user", "content": "Description: " + text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        assert inputs.input_ids.shape[1] + manifest["max_new_tokens"] <= 1536
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=manifest["max_new_tokens"],
                pad_token_id=tokenizer.eos_token_id, use_cache=True)
        torch.cuda.synchronize()
        generated = output[0, inputs.input_ids.shape[1]:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        parsed, error = parse_once.parse_strict(raw)
        return {
            "raw": raw, "parsed": parsed, "parse_error": error,
            "seconds": time.perf_counter() - started,
            "generated_tokens": len(generated),
            "hit_token_cap": len(generated) >= manifest["max_new_tokens"] and generated[-1].item() != tokenizer.eos_token_id,
        }

    anchor = sup.read(V2 / "special_holdout.json")[0]
    reference = json.loads((V2 / "special_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    anchor_result = predict(anchor["description"])
    anchor_result["same_raw_as_v2"] = anchor_result["raw"] == reference["raw"]
    atomic(OUT / "reload_check.json", anchor_result)
    assert anchor_result["same_raw_as_v2"]

    mode = "a" if completed else "w"
    with prediction_path.open(mode, encoding="utf-8") as stream:
        for item in descriptions:
            if item["description_id"] in done:
                continue
            result = predict(item["description"])
            result.update(item)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            completed.append(result)
            elapsed = sum(x["seconds"] for x in completed)
            invalid = sum(x["parsed"] is None for x in completed)
            remaining = (len(descriptions) - len(completed)) * elapsed / len(completed)
            atomic(OUT / "status.json", {
                "status": "RUNNING",
                "completed_descriptions": len(completed),
                "total_descriptions": len(descriptions),
                "invalid_outputs": invalid,
                "last_description_id": item["description_id"],
                "mean_seconds_per_description": elapsed / len(completed),
                "remaining_hours_estimate": remaining / 3600,
            })
            print(len(completed), "/", len(descriptions), "invalid", invalid, flush=True)

    invalid = [x for x in completed if x["parsed"] is None]
    if invalid:
        atomic(OUT / "invalid_outputs.json", invalid)
        atomic(OUT / "status.json", {
            "status": "NEEDS_REVIEW",
            "completed_descriptions": len(completed),
            "total_descriptions": len(descriptions),
            "invalid_outputs": len(invalid),
            "zip_created": False,
        })
        print("Unresolved descriptions; no submission ZIP created", flush=True)
        return
    by_text = {x["description"] : x for x in completed}
    assert len(by_text) == len(descriptions)
    build_results(manifest, by_text)
    pack(manifest)
    print("TEST_PARSE_ONCE_V2_COMPLETE", flush=True)


def resolve():
    """Resolve the one audited noisy distance-range template without altering raw output."""
    manifest = sup.read(OUT / "manifest.json")
    verify_sources(manifest)
    descriptions = sup.read(OUT / "descriptions.json")
    predictions = [
        json.loads(line)
        for line in (OUT / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(predictions) == len(descriptions) == manifest["unique_descriptions"]
    invalid = [x for x in predictions if x["parsed"] is None]
    assert len(invalid) == 1
    item = invalid[0]
    expected_description = (
        "Vehicles on the left side of the scene, which have a height ranging from "
        "approximately 0.8 to 1.2 meters, are positioned at a distance of about "
        "10 to 15 meters from the observer."
    )
    assert item["description_id"] == "test-description-0031"
    assert item["description"] == expected_description
    raw_object = json.loads(item["raw"])
    assert raw_object.pop("distance_max_m") == 10.0
    assert raw_object.pop("distance_min_m") == 15.0
    resolved, error = parse_once.parse_strict(json.dumps(raw_object, separators=(",", ":")))
    assert error is None
    assert resolved == {
        "height_min_m": 0.8, "height_max_m": 1.2, "position": "left"
    }
    sibling = next(x for x in predictions if x["description_id"] == "test-description-0548")
    assert "distance" not in sibling["raw"]
    resolution = {
        "status": "RESOLVED",
        "description_id": item["description_id"],
        "original_raw": item["raw"],
        "original_error": item["parse_error"],
        "resolved_conditions": resolved,
        "policy": (
            "Ignore only the 10-to-15-meter distance phrase in this audited template. "
            "Training analogues conflict with their positive-object distances, and the "
            "second test occurrence of the same template was independently omitted by V2."
        ),
        "raw_prediction_modified": False,
    }
    atomic(OUT / "resolution.json", resolution)
    resolved_predictions = []
    for row in predictions:
        copy = dict(row)
        if row["description_id"] == item["description_id"]:
            copy["parsed"] = resolved
            copy["resolution"] = "resolution.json"
        resolved_predictions.append(copy)
    by_text = {x["description"] : x for x in resolved_predictions}
    assert len(by_text) == len(descriptions)
    assert all(x["parsed"] is not None for x in resolved_predictions)
    build_results(manifest, by_text)
    pack(manifest)
    complete = sup.read(OUT / "complete.json")
    complete["strict_valid_model_outputs"] = len(predictions) - 1
    complete["audited_resolutions"] = 1
    complete["resolution_sha256"] = sup.digest(OUT / "resolution.json")
    atomic(OUT / "complete.json", complete)
    print("TEST_PARSE_ONCE_V2_RESOLVED_AND_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "resolve"])
    args = parser.parse_args()
    try:
        {"prepare": prepare, "run": run, "resolve": resolve}[args.action]()
    except Exception as error:
        if OUT.exists():
            atomic(OUT / "failure.json", {
                "error": str(error), "traceback": traceback.format_exc(),
            })
            if args.action == "run":
                atomic(OUT / "status.json", {
                    "status": "FAILED", "error": str(error),
                    "resume": "Re-run the same command; completed descriptions are preserved",
                })
        raise
