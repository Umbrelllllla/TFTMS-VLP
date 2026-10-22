"""Reload the V2 adapter and reproduce one frozen special-holdout output."""
import json
import os

import prepare_supervision as sup
import run_parse_once as base


ROOT = sup.ROOT
OUT = ROOT / "runs/qwen2b_parse_once_sft_v2"


def main():
    complete = sup.read(OUT / "complete.json")
    assert complete["status"] == "PASS"
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from peft import PeftModel, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        base.MODEL, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", output_loading_info=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    assert not loading.get("missing_keys")
    assert not loading.get("mismatched_keys")
    assert not loading.get("error_msgs")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, OUT / "adapter", is_trainable=False)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    assert not any(p.requires_grad for p in model.parameters())

    sample = sup.read(OUT / "special_holdout.json")[0]
    reference = json.loads((OUT / "special_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    messages = [
        {"role": "system", "content": base.SYSTEM_PROMPT},
        {"role": "user", "content": "Description: " + sample["description"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id, use_cache=True)
    raw = tokenizer.decode(
        output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    result = {
        "status": "PASS" if raw == reference["raw"] else "FAIL",
        "sample_id": sample["sample_id"],
        "reference_raw": reference["raw"],
        "reloaded_raw": raw,
        "exact_match": raw == reference["raw"],
        "adapter_sha256": {
            p.name: sup.digest(p) for p in (OUT / "adapter").iterdir() if p.is_file()
        },
    }
    sup.write(OUT / "reload_check.json", result)
    assert result["exact_match"]
    complete["reload_anchor_exact"] = True
    sup.write(OUT / "complete.json", complete)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
