"""Train and evaluate the 512-description flat-condition LoRA pilot."""
import collections
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import traceback

import prepare_supervision as sup
import run_parse_once as base


ROOT = sup.ROOT
OUT = ROOT / "runs/qwen2b_parse_once_sft_v1"
MODEL = base.MODEL
EVAL_SOURCE = ROOT / "runs/qwen2b_parse_once_v1"
SEED = 20261021


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def verify_inputs():
    manifest = sup.read(OUT / "manifest.json")
    assert manifest["train_sha256"] == sup.digest(OUT / "train.json")
    assert manifest["formal_validation_sha256"] == sup.digest(ROOT / "runs/qwen2b_supervision_ab_expanded_v1/validation.json")
    assert manifest["system_prompt_sha256"] == hashlib.sha256(base.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert sup.digest(EVAL_SOURCE / "descriptions.json") == sup.read(EVAL_SOURCE / "manifest.json")["descriptions_sha256"]
    return manifest


def score(predictions, training_seconds, peak_mib):
    descriptions = sup.read(EVAL_SOURCE / "descriptions.json")
    cases = sup.read(ROOT / "runs/qwen2b_supervision_ab_expanded_v1/validation.json")
    by_prediction = {p["description_id"]: p for p in predictions}
    by_source = {d["source_key"]: d for d in descriptions}
    extraction = []
    fields = collections.Counter()
    for d in descriptions:
        p = by_prediction[d["description_id"]]
        if p["parsed"] is None:
            extraction.append({**d, "raw": p["raw"], "parsed": None, "parse_error": p["parse_error"], "exact": False})
            fields["invalid_descriptions"] += 1
            continue
        missing, extra, wrong = base.condition_diff(p["parsed"], d["gold_conditions"])
        for key in missing: fields[f"missing::{key}"] += 1
        for key in extra: fields[f"extra::{key}"] += 1
        for key in wrong: fields[f"wrong::{key}"] += 1
        extraction.append({**d, "raw": p["raw"], "parsed": p["parsed"], "parse_error": None,
                           "missing_keys": missing, "extra_keys": extra, "wrong_value_keys": wrong,
                           "exact": not (missing or extra or wrong)})
    exact = {x["description_id"]: x["exact"] for x in extraction}
    train_rows = sup.read(OUT / "train.json")
    train_texts = {x["description"] for x in train_rows}
    normalize_template = lambda text: re.sub(r"\d+(?:\.\d+)?", "<NUM>", text.lower())
    train_templates = {normalize_template(x["description"]) for x in train_rows}
    novelty = {}
    for d in descriptions:
        if d["description"] in train_texts:
            novelty[d["description_id"]] = "exact_text_seen"
        elif normalize_template(d["description"]) in train_templates:
            novelty[d["description_id"]] = "template_seen_new_values"
        else:
            novelty[d["description_id"]] = "template_unseen"
    counts = collections.Counter(); subgroups = collections.defaultdict(collections.Counter); scored = []
    novelty_groups = collections.defaultdict(collections.Counter)
    started = time.perf_counter(); oracle = 0
    for case in cases:
        d = by_source[base.description_key(case)]
        p = by_prediction[d["description_id"]]
        oracle_prediction, _ = base.match_candidate(d["gold_conditions"], case["input"]["candidate"])
        oracle += int(int(oracle_prediction) == case["target"])
        if p["parsed"] is None:
            group = "invalid_output"; prediction = None; checks = None; correct = False
            counts["unresolved"] += 1
            if case["target"]: counts["fn"] += 1
        else:
            group = "exact_conditions" if exact[d["description_id"]] else "valid_but_inexact_conditions"
            value, checks = base.match_candidate(p["parsed"], case["input"]["candidate"])
            prediction = int(value); correct = prediction == case["target"]
            if prediction and case["target"]: counts["tp"] += 1
            elif prediction: counts["fp"] += 1
            elif case["target"]: counts["fn"] += 1
            else: counts["tn"] += 1
        subgroups[group]["total"] += 1; subgroups[group]["correct"] += int(correct)
        novelty_groups[novelty[d["description_id"]]]["total"] += 1
        novelty_groups[novelty[d["description_id"]]]["correct"] += int(correct)
        scored.append({"pair_id": case["input"]["pair_id"], "description_id": d["description_id"],
                       "gold": case["target"], "prediction": prediction, "correct": correct, "checks": checks})
    program_seconds = time.perf_counter() - started
    assert oracle == 240
    correct = sum(x["correct"] for x in scored)
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    inference_seconds = sum(p["seconds"] for p in predictions)
    baseline = sup.read(EVAL_SOURCE / "results.json")
    ab = sup.read(ROOT / "runs/qwen2b_supervision_ab_expanded_v1/results.json")
    result = {
        "description_extraction": {"valid": sum(p["parsed"] is not None for p in predictions), "total": 120,
                                   "exact_conditions": sum(x["exact"] for x in extraction),
                                   "field_errors": dict(sorted(fields.items())),
                                   "generated_token_cap_hits": sum(p["hit_token_cap"] for p in predictions)},
        "candidate_decisions": {"correct": correct, "total": 240, "accuracy": correct / 240,
                                "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"],
                                "unresolved": counts["unresolved"], "precision": precision, "recall": recall, "f1": f1},
        "program_verification": {"gold_condition_oracle_correct": oracle, "total": 240,
                                 "subgroups": {k: dict(v) for k, v in subgroups.items()}},
        "novelty_analysis": {"description_groups": dict(collections.Counter(novelty.values())),
                             "candidate_decisions": {k: dict(v) for k, v in novelty_groups.items()},
                             "file_overlap": 0, "camera_group_overlap": 0,
                             "note": "Template normalization replaces every number with <NUM>; it is a diagnostic, not a semantic equivalence proof."},
        "timing": {"training_seconds": training_seconds, "training_peak_allocated_mib": peak_mib,
                   "model_inference_seconds": inference_seconds, "mean_seconds_per_description": inference_seconds / 120,
                   "program_matching_seconds": program_seconds},
        "comparison": {"base_parse_once_correct": baseline["candidate_decisions"]["correct"],
                       "A_correct": ab["A"]["correct"], "B_correct": ab["B"]["correct"]},
    }
    sup.write(OUT / "extraction_details.json", extraction)
    sup.write(OUT / "scored_pairs.json", scored)
    sup.write(OUT / "errors.json", [x for x in scored if not x["correct"]])
    sup.write(OUT / "results.json", result)
    return result


def main():
    manifest = verify_inputs()
    assert not (OUT / "adapter").exists(), "Adapter already exists; refusing to overwrite frozen pilot"
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict
    from safetensors.torch import load_file
    from download import verify

    for entry in sup.read(MODEL / "source_manifest.json")["siblings"]:
        if entry["rfilename"] != ".gitattributes": assert verify(MODEL / entry["rfilename"], entry)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    training = sup.read(OUT / "train.json")
    prepared = []
    for row in training:
        messages = [{"role": "system", "content": base.SYSTEM_PROMPT},
                    {"role": "user", "content": "Description: " + row["description"]}]
        prefix = tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                                enable_thinking=False), add_special_tokens=False)
        answer = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
        suffix = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        assert tokenizer.decode(suffix[:-1]) == answer and len(prefix + suffix) <= 1024
        prepared.append((row["sample_id"], prefix, suffix))
    config = {"seed": SEED, "samples": len(prepared), "epochs": 1, "batch_size": 1, "rank": 4,
              "alpha": 8, "dropout": 0.0, "target_modules": "all-linear", "learning_rate": 0.0001,
              "quantization": "NF4 double quantization, bfloat16 compute", "max_new_tokens": 128,
              "training_supervised_tokens": sum(len(x[2]) for x in prepared),
              "train_sha256": manifest["train_sha256"],
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes")}}
    sup.write(OUT / "config.json", config)
    atomic(OUT / "status.json", {"status": "LOADING", "completed_steps": 0, "total_steps": len(prepared)})
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        output_loading_info=True, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys") and not loading.get("error_msgs")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                           target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    assert trainable and all("lora_" in n for n in trainable)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=config["learning_rate"])
    model.train(); model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    steps = []
    with (OUT / "steps.jsonl").open("w", encoding="utf-8") as stream:
        for index, (sample_id, prefix, suffix) in enumerate(prepared, 1):
            ids = torch.tensor([prefix + suffix], device="cuda")
            positions = torch.arange(len(prefix) - 1, ids.shape[1] - 1, device="cuda")
            labels = ids[:, len(prefix):]
            optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                               logits_to_keep=positions).logits
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            assert all(p.grad is None and p._version == frozen[n] for n, p in model.named_parameters() if not p.requires_grad)
            record = {"step": index, "sample_id": sample_id, "loss": loss.item(), "gradient_norm": norm.item(),
                      "seconds": time.perf_counter() - started, "supervised_tokens": labels.numel(),
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
            steps.append(record); stream.write(json.dumps(record) + "\n"); stream.flush()
            elapsed = sum(x["seconds"] for x in steps)
            atomic(OUT / "status.json", {"status": "TRAINING", "completed_steps": index, "total_steps": len(prepared),
                   "last_loss": record["loss"], "mean_seconds_per_step": elapsed / index,
                   "remaining_minutes_estimate": (len(prepared) - index) * elapsed / index / 60,
                   "peak_allocated_mib": max(x["peak_allocated_mib"] for x in steps)})
            if index % 8 == 0: print("train", index, "/", len(prepared), "loss", round(record["loss"], 4), flush=True)
            del ids, positions, labels, logits, loss
    changed = sum(not torch.equal(initial[n], p.detach().cpu()) for n, p in trainable.items())
    assert changed > 0
    model.save_pretrained(OUT / "adapter", safe_serialization=True)
    saved = load_file(OUT / "adapter/adapter_model.safetensors"); live = get_peft_model_state_dict(model)
    assert set(saved) == set(live) and all(torch.equal(saved[k], live[k].detach().cpu()) for k in saved)
    training_seconds = sum(x["seconds"] for x in steps); peak_mib = max(x["peak_allocated_mib"] for x in steps)
    atomic(OUT / "status.json", {"status": "EVALUATING", "completed_descriptions": 0, "total_descriptions": 120})

    model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache = True
    predictions = []
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(sup.read(EVAL_SOURCE / "descriptions.json"), 1):
            messages = [{"role": "system", "content": base.SYSTEM_PROMPT},
                        {"role": "user", "content": "Description: " + row["description"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(**inputs, do_sample=False, max_new_tokens=config["max_new_tokens"],
                                        pad_token_id=tokenizer.eos_token_id, use_cache=True)
            torch.cuda.synchronize(); generated = output[0, inputs.input_ids.shape[1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True).strip(); parsed, error = base.parse_strict(raw)
            prediction = {"description_id": row["description_id"], "raw": raw, "parsed": parsed, "parse_error": error,
                          "seconds": time.perf_counter() - started, "generated_tokens": len(generated),
                          "hit_token_cap": len(generated) >= config["max_new_tokens"] and generated[-1].item() != tokenizer.eos_token_id}
            predictions.append(prediction); stream.write(json.dumps(prediction, ensure_ascii=False) + "\n"); stream.flush()
            elapsed = sum(x["seconds"] for x in predictions)
            atomic(OUT / "status.json", {"status": "EVALUATING", "completed_descriptions": index,
                   "total_descriptions": 120, "invalid_outputs": sum(x["parsed"] is None for x in predictions),
                   "mean_seconds_per_description": elapsed / index,
                   "remaining_minutes_estimate": (120 - index) * elapsed / index / 60})
            print("eval", index, "/120", round(prediction["seconds"], 2), "s", "valid" if parsed else error, flush=True)
    result = score(predictions, training_seconds, peak_mib)
    sup.write(OUT / "complete.json", {"status": "PASS", "steps": len(steps), "changed_tensors": changed,
              "source_frozen": True, "adapter_save_verified": True,
              "final_correct": result["candidate_decisions"]["correct"]})
    atomic(OUT / "status.json", {"status": "COMPLETE", "steps": len(steps),
           "final_correct": result["candidate_decisions"]["correct"], "total": 240})
    if (OUT / "failure.json").exists(): (OUT / "failure.json").unlink()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if OUT.exists():
            atomic(OUT / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
            atomic(OUT / "status.json", {"status": "FAILED", "error": str(error)})
        raise
