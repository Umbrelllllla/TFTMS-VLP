"""Small training-side inference check; no training or validation-set scoring."""
import importlib.metadata
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def main():
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from download import DEST, REVISION, verify

    output = ROOT / 'runs' / ('qwen2b_smoke_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available(), 'CUDA not available; stop rather than silently use CPU.'
    source = json.loads((DEST / 'source_manifest.json').read_text(encoding='utf-8'))
    for entry in source['siblings']:
        if entry['rfilename'] != '.gitattributes':
            assert verify(DEST / entry['rfilename'], entry), entry['rfilename']
    environment = {
        'revision': REVISION, 'gpu': torch.cuda.get_device_name(),
        'cuda': torch.version.cuda,
        'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'accelerate', 'bitsandbytes')},
        'quantization': 'NF4 4bit with double quantization; bfloat16 compute',
        'generation': {'do_sample': False, 'max_new_tokens': 8, 'enable_thinking': False},
        'scope': 'first 12 training pairs, operational smoke check only',
    }
    (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print('ENVIRONMENT', json.dumps(environment), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(DEST, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        DEST, local_files_only=True, dtype=torch.bfloat16, device_map={'': 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        attn_implementation='sdpa', output_loading_info=True,
    )
    (output / 'loading_info.json').write_text(json.dumps(loading,
        default=lambda value: sorted(value) if isinstance(value, set) else str(value), indent=2), encoding='utf-8')
    assert not loading.get('missing_keys'), 'Missing language weights; stop.'
    assert not loading.get('mismatched_keys'), 'Mismatched weight shapes; stop.'
    assert not any('visual' in name for name, _ in model.named_modules()), 'Unexpected visual model.'
    model.eval()
    print(f'MODEL LOADED {time.perf_counter()-started:.1f}s', flush=True)
    data = ROOT / 'runs' / '20261015_qwen2b_input_v1'
    prompt = (data / 'system_prompt.txt').read_text(encoding='utf-8')
    from itertools import islice
    with (data / 'train.inputs.jsonl').open(encoding='utf-8') as f:
        items = [json.loads(line) for line in islice(f, 12)]
    results = []
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as f:
        for item in items:
            messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(
                {k: item[k] for k in ('description', 'candidate')}, ensure_ascii=False)}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')
            assert inputs.input_ids.shape[1] <= 2048, 'Unexpected prompt length; do not silently truncate.'
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True,
                                        pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            raw = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            result = {'pair_id': item['pair_id'], 'raw': raw,
                      'prediction': int(raw) if raw in ('0', '1') else None,
                      'input_tokens': inputs.input_ids.shape[1], 'seconds': time.perf_counter()-start}
            f.write(json.dumps(result) + '\n')
            f.flush()
            results.append(result)
            print(json.dumps(result), flush=True)
    # Only now read answers; they cannot influence inference.
    with (data / 'train.answers.jsonl').open(encoding='utf-8') as f:
        answers = {x['pair_id']: x['target'] for x in map(json.loads, islice(f, 12))}
    summary = {'count': len(results), 'valid_format': sum(r['prediction'] is not None for r in results),
               'correct': sum(r['prediction'] == answers[r['pair_id']] for r in results),
               'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
               'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
               'note': 'Training-side functionality check; not validation accuracy or competition score.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps(summary), flush=True)
    print('RESULT', output, flush=True)


if __name__ == '__main__':
    main()
