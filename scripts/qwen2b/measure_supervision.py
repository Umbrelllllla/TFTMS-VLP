"""Measure formatted training sequence lengths without loading model weights."""
import json
from pathlib import Path
from windows_runtime import prepare_runtime

prepare_runtime()
from transformers import AutoTokenizer
from download import DEST

root = Path(__file__).resolve().parents[2]
p = root / 'runs/qwen2b_intermediate_supervision_v1'
tokenizer = AutoTokenizer.from_pretrained(DEST, local_files_only=True)
rows = [json.loads(x) for x in (p/'sft_preview.jsonl').read_text(encoding='utf-8').splitlines()]
sizes = []
for r in rows:
    rendered = tokenizer.apply_chat_template(r['messages'], tokenize=False, enable_thinking=False)
    total = len(tokenizer.encode(rendered, add_special_tokens=False))
    answer = len(tokenizer.encode(r['messages'][-1]['content'], add_special_tokens=False))
    assert total > answer > 0
    sizes.append({'pair_id':r['pair_id'], 'total_tokens':total, 'answer_tokens':answer})
(p/'token_lengths.json').write_text(json.dumps(sizes,indent=2),encoding='utf-8')
print('total_token_range',min(x['total_tokens'] for x in sizes),max(x['total_tokens'] for x in sizes))
print('answer_token_range',min(x['answer_tokens'] for x in sizes),max(x['answer_tokens'] for x in sizes))
