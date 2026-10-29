"""Benchmark batched V2 inference without modifying the frozen adapter."""
import collections
import json
import os
from pathlib import Path
import time
import traceback

import prepare_supervision as sup
import run_parse_once as base
import train_parse_once_sft_v2 as v2


ROOT = sup.ROOT
OUT = ROOT / "runs/qwen2b_batch_benchmark_v1"
MODEL = base.MODEL
ADAPTER = ROOT / "runs/qwen2b_parse_once_sft_v2/adapter"
GENERAL = ROOT / "runs/qwen2b_parse_once_sft_holdout_v1"
BATCH_SIZES = (1, 2, 4, 8)
BENCHMARK_ROWS = 48


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def condition_exact(predicted, gold):
    if predicted is None: return False
    return not any(base.condition_diff(predicted, gold))


def generate_batches(model, tokenizer, rows, batch_size, max_new_tokens=64):
    import torch
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: len(item[1]["description"]))
    outputs = [None] * len(rows)
    batch_records = []
    torch.cuda.reset_peak_memory_stats()
    started_all = time.perf_counter()
    for start in range(0, len(indexed), batch_size):
        batch = indexed[start:start + batch_size]
        prompts = []
        for _, row in batch:
            messages = [{"role":"system","content":base.SYSTEM_PROMPT},
                        {"role":"user","content":"Description: " + row["description"]}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False,
                           add_generation_prompt=True, enable_thinking=False))
        inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        torch.cuda.synchronize(); started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                       pad_token_id=tokenizer.eos_token_id, use_cache=True)
        torch.cuda.synchronize(); seconds = time.perf_counter() - started
        suffix = generated[:, inputs.input_ids.shape[1]:]
        raws = tokenizer.batch_decode(suffix, skip_special_tokens=True)
        for (original_index, row), raw in zip(batch, raws):
            raw = raw.strip(); parsed, error = base.parse_strict(raw)
            outputs[original_index] = {"description_id":row["description_id"], "raw":raw,
                "parsed":parsed, "parse_error":error, "batch_size":len(batch),
                "seconds":seconds / len(batch)}
        batch_records.append({"rows":len(batch), "seconds":seconds,
            "prompt_tokens":int(inputs.attention_mask.sum().item()),
            "padded_prompt_tokens":int(inputs.input_ids.numel())})
        del inputs, generated, suffix
    wall = time.perf_counter() - started_all
    return outputs, {"wall_seconds":wall, "generate_seconds":sum(x["seconds"] for x in batch_records),
        "descriptions_per_second":len(rows)/wall, "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
        "batches":batch_records}


def main():
    resume = OUT.exists()
    OUT.mkdir(parents=True, exist_ok=True)
    descriptions = sup.read(GENERAL / "descriptions.json")
    cases = sup.read(GENERAL / "validation.json")
    if not resume:
        atomic(OUT / "manifest.json", {"status":"PREPARED", "batch_sizes":BATCH_SIZES,
            "benchmark_descriptions":BENCHMARK_ROWS, "full_validation_descriptions":len(descriptions),
            "adapter_sha256":{p.name:sup.digest(p) for p in ADAPTER.iterdir() if p.is_file()},
            "policy":"Only exact benchmark configurations may enter full validation."})
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    model, loading = Qwen3_5ForCausalLM.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16, device_map={"":0}, attn_implementation="sdpa", output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs")
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval(); model.config.use_cache = True
    benchmark = descriptions[:BENCHMARK_ROWS]
    if resume and (OUT / "benchmark_resume.json").exists():
        results = sup.read(OUT / "benchmark_resume.json")
        for batch_size in BATCH_SIZES:
            saved = sup.read(OUT / f"batch_{batch_size}_predictions.json")
            assert sum(condition_exact(p["parsed"], row["gold_conditions"])
                       for p, row in zip(saved, benchmark)) == BENCHMARK_ROWS
        print("Resuming from verified benchmark predictions", flush=True)
    else:
        results = {}
        for batch_size in BATCH_SIZES:
            try:
                predictions, timing = generate_batches(model, tokenizer, benchmark, batch_size)
                exact = sum(condition_exact(p["parsed"], row["gold_conditions"])
                            for p, row in zip(predictions, benchmark))
                result = {"status":"PASS" if exact == len(benchmark) else "INEXACT",
                          "exact":exact, "total":len(benchmark), **timing}
                results[str(batch_size)] = result
                atomic(OUT / f"batch_{batch_size}_predictions.json", predictions)
                atomic(OUT / "status.json", {"status":"BENCHMARKING", "completed_batch_size":batch_size,
                       "results":results})
                print("batch", batch_size, result, flush=True)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                results[str(batch_size)] = {"status":"OOM", "error":str(error)}
                print("batch", batch_size, "OOM", flush=True)
    eligible = [(float(value["descriptions_per_second"]), int(key)) for key, value in results.items()
                if value["status"] == "PASS"]
    assert eligible, "No exact batch configuration"
    best_batch = max(eligible)[1]
    full_predictions, full_timing = generate_batches(model, tokenizer, descriptions, best_batch)
    sup.write(OUT / "full_predictions.json", full_predictions)
    general_result, extraction, scored = v2.score_general(descriptions, cases, full_predictions)
    sup.write(OUT / "full_extraction.json", extraction)
    sup.write(OUT / "full_scored.json", scored)
    final = {"benchmark":results, "best_batch_size":best_batch,
             "full_validation":general_result, "full_timing":full_timing,
             "reference":{"sequential_inference_seconds":sup.read(ROOT / "runs/qwen2b_parse_once_sft_v2/results.json")["general_regression"]["inference_seconds"]}}
    sup.write(OUT / "results.json", final)
    atomic(OUT / "status.json", {"status":"COMPLETE", "best_batch_size":best_batch,
           "full_correct":general_result["candidate_correct"], "full_total":general_result["candidate_total"]})
    if (OUT / "failure.json").exists(): (OUT / "failure.json").unlink()
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try: main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic(OUT / "failure.json", {"error":str(error), "traceback":traceback.format_exc()})
        atomic(OUT / "status.json", {"status":"FAILED", "error":str(error)})
        raise
