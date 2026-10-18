"""Fixed training-side prompt ablation: prepare, run, report. No training."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'runs/20261015_qwen2b_input_v1'
AUDIT = ROOT / 'runs/20261015_qwen2b_data_audit'
OUT = ROOT / 'runs/qwen2b_prompt_diagnostic_v1'
SEED = 20261016
EXTRA = '''
Check every stated condition. ALL stated conditions must hold for a match.
Ignore attributes that the description does not mention.
Use dimensions_m.length for length, dimensions_m.width for width,
dimensions_m.height for height, and distance_m for distance from the camera.
Within D metres means distance_m <= D. Between A and B includes both endpoints.
Do not require the candidate value to equal an upper bound.
Use the actual numeric value, not the number of digits or the length of its text.
Return exactly 1 if all stated conditions hold; otherwise return exactly 0.
'''


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def flat(candidate):
    # Same values and original field names as JSON, flattened without rounding.
    rows = []
    units = {'dimensions_m': 'metres', 'camera_position_m': 'metres',
             'distance_m': 'metres', 'bbox_xyxy_px': 'pixels', 'image_size_px': 'pixels',
             'observation_angle_rad': 'radians', 'yaw_rad': 'radians'}
    def visit(d, prefix=''):
        for k, v in d.items():
            name = prefix + k
            if isinstance(v, dict):
                visit(v, name + '.')
            else:
                unit = units.get(name.split('.')[0], '')
                rows.append(f'{name}: {json.dumps(v, ensure_ascii=False)}' + (f' ({unit})' if unit else ''))
    visit(candidate)
    return '\n'.join(rows)


def user_content(item, version):
    if version == 'V0':
        return json.dumps({k: item[k] for k in ('description', 'candidate')}, ensure_ascii=False)
    return 'Description: ' + item['description'] + '\nCandidate:\n' + flat(item['candidate'])


def messages(item, version, examples):
    system = (DATA / 'system_prompt.txt').read_text(encoding='utf-8')
    result = [{'role': 'system', 'content': system + (EXTRA if version in ('V2', 'V3') else '')}]
    if version == 'V3':
        for e in examples:
            result.extend([{'role': 'user', 'content': user_content(e['input'], version)},
                           {'role': 'assistant', 'content': str(e['target'])}])
    result.append({'role': 'user', 'content': user_content(item, version)})
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'tasks.jsonl').exists():
        raise RuntimeError('Frozen tasks already exist; do not silently resample.')
    spec = importlib.util.spec_from_file_location('input_prepare', DATA / 'prepare.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    split = read(DATA / 'split_manifest.json')
    allowed = set(split['train'])
    records = [r for r in read(AUDIT / 'training_records.json') if r['file'] in allowed]
    rng = random.Random(SEED)
    rng.shuffle(records)
    groups = {'attribute': {'objectName', 'appearance'}, 'size': {'height', 'width', 'length'},
              'distance': {'distance'}, 'place': {'place'}, 'facing': {'facing_direction'}, 'occlusion': {'occlusion'}}

    def fails(o, r, props):
        f = mod.features(o, r['width'], r['height'])
        errors = []
        for k, v in props.items():
            if k == 'objectName': ok = o[0] == v
            elif k == 'appearance': ok = o[15] == v
            elif k == 'occlusion': ok = f['occlusion'] == v
            elif k == 'place': ok = f['image_region'] == v
            elif k == 'facing_direction': ok = f['facing_convention'] == v
            elif k == 'distance': ok = f['distance_m'] <= float(v.split()[0])
            else:
                lo, hi = map(float, re.findall(r'[\d.]+', v))
                ok = lo <= f['dimensions_m'][k] <= hi
            if not ok: errors.append(k)
        return errors

    def pair(r, ai):
        a = r['annotations'][ai]
        pos = a['positive_indices']
        neg = [i for i in range(len(r['objects'])) if i not in pos]
        if not pos or not neg: return None
        rng.shuffle(neg)
        ni = min(neg, key=lambda i: len(fails(r['objects'][i], r, a['properties'])))
        result = []
        for oi, target in [(rng.choice(pos), 1), (ni, 0)]:
            error_fields = fails(r['objects'][oi], r, a['properties'])
            assert bool(error_fields) == (target == 0)
            result.append({'input': {'pair_id': f'{r["file"]}:{ai}:{oi}', 'description': a['text'],
                          'candidate': mod.features(r['objects'][oi], r['width'], r['height'])},
                          'target': target, 'source_file': r['file'], 'camera': r['camera'],
                          'properties_for_audit_only': a['properties'], 'violated_fields_for_audit_only': error_fields})
        return result

    used = set()
    examples = []
    # Four demonstrations, two positive/two negative from two reserved images.
    for r in records:
        for ai in range(3):
            selected = pair(r, ai)
            if selected:
                examples.extend(selected)
                used.add(r['file'])
                break
        if len(examples) == 4: break
    cases = []
    for group, keys in groups.items():
        n = 0
        for r in records:
            if r['file'] in used: continue
            for ai, a in enumerate(r['annotations']):
                if not keys & set(a['properties']): continue
                selected = pair(r, ai)
                if selected:
                    for x in selected: x['group'] = group
                    cases.extend(selected)
                    used.add(r['file'])
                    n += 2
                    break
            if n == 20: break
        assert n == 20, group
    assert len(cases) == 120
    assert len({c['source_file'] for c in cases}) == 60
    assert not {c['source_file'] for c in cases} & {e['source_file'] for e in examples}
    assert all(c['source_file'] in allowed for c in cases + examples)
    rng.shuffle(cases)
    # IDs and answers live outside actual model messages.
    for i, c in enumerate(cases): c['input']['pair_id'] = f'diagnostic-{i:03d}'
    write(OUT / 'cases_answers.json', cases)
    write(OUT / 'demonstrations.json', examples)
    tasks = []
    for version in ('V0', 'V1', 'V2', 'V3'):
        for c in cases:
            tasks.append({'task_id': version + ':' + c['input']['pair_id'], 'version': version,
                          'pair_id': c['input']['pair_id'], 'messages': messages(c['input'], version, examples),
                          'max_new_tokens': 8})
    probes = []
    def probe(name, question, expected, kind='binary'):
        probes.append({'task_id': 'probe:' + name, 'version': 'probe', 'pair_id': name,
                       'messages': [{'role': 'system', 'content': 'Follow the requested output format exactly.'},
                                    {'role': 'user', 'content': question}],
                       'max_new_tokens': 80 if kind == 'json' else 16})
        probe_answers[name] = {'expected': expected, 'kind': kind}
    probe_answers = {}
    probe('length_yes', 'Is 4.231418 between 4.1 and 4.5, inclusive? Answer only 1 for yes or 0 for no.', 1)
    probe('length_no', 'Is 8.795015 between 4.1 and 4.5, inclusive? Answer only 1 for yes or 0 for no.', 0)
    probe('distance_yes', 'Is 15.519878769553257 <= 23.6? Answer only 1 for yes or 0 for no.', 1)
    probe('distance_no', 'Is 28.5 <= 23.6? Answer only 1 for yes or 0 for no.', 0)
    first = read(ROOT / 'runs/qwen2b_smoke_20261016_224244/review_cases.json')[0]['input']
    for field in ('length', 'width'):
        probe('lookup_' + field, json.dumps(first['candidate']) + f'\nRead dimensions_m.{field}. Output only its numeric value.',
              first['candidate']['dimensions_m'][field], 'number')
    for i, description in enumerate([first['description'], 'Select vehicles at most 23.6 metres away and between 4.1 and 4.5 metres long.']):
        probe('extract_' + str(i), description + '\nExtract constraints. Return only JSON with numeric keys distance_max_m, length_min_m, length_max_m.',
              {'distance_max_m': 23.6, 'length_min_m': 4.1, 'length_max_m': 4.5}, 'json')
    for name, length, target in [('combined_yes', 4.231418, 1), ('combined_no', 8.795015, 0)]:
        probe(name, first['description'] + f'\nCandidate: car; length {length} metres; distance 15.519878769553257 metres. '
              'Does the candidate satisfy every condition? Answer only 1 for yes or 0 for no.', target)
    write(OUT / 'probe_answers.json', probe_answers)
    with (OUT / 'tasks.jsonl').open('w', encoding='utf-8') as f:
        for task in probes + tasks: f.write(json.dumps(task, ensure_ascii=False) + '\n')
    # Check every flattened candidate can be reconstructed exactly.
    for c in cases + examples:
        original = c['input']['candidate']
        recovered = {}
        for line in flat(original).splitlines():
            k, value = line.split(': ', 1)
            value = re.sub(r' \((metres|pixels|radians)\)$', '', value)
            if '.' in k:
                a, b = k.split('.')
                recovered.setdefault(a, {})[b] = json.loads(value)
            else: recovered[k] = json.loads(value)
        assert original == recovered
    write(OUT / 'manifest.json', {'seed': SEED, 'cases': 120, 'images': 60, 'positive': 60, 'negative': 60,
        'demonstration_images': 2, 'versions': ['V0', 'V1', 'V2', 'V3'], 'probes': len(probes),
        'groups': dict(collections.Counter(c['group'] for c in cases)),
        'property_coverage': dict(collections.Counter(k for c in cases for k in c['properties_for_audit_only'])),
        'single_failure_negatives': sum(c['target'] == 0 and len(c['violated_fields_for_audit_only']) == 1 for c in cases),
        'single_condition_cases': sum(len(c['properties_for_audit_only']) == 1 for c in cases),
        'task_sha256': sha(OUT / 'tasks.jsonl'), 'split_sha256': sha(DATA / 'split_manifest.json'),
        'checks': 'PASS: training only; demonstration image exclusion; 60/60 balanced; all values preserved in V1',
        'limits': 'Diagnostic selection uses training properties; properties are not model input. Six groups may contain overlapping conditions. Not representative validation.'})
    print(json.dumps(read(OUT / 'manifest.json'), indent=2))


def run():
    os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from download import DEST, REVISION, verify
    import importlib.metadata
    assert sha(OUT / 'tasks.jsonl') == read(OUT / 'manifest.json')['task_sha256']
    tasks = [json.loads(x) for x in (OUT / 'tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    assert torch.cuda.is_available()
    for entry in read(DEST / 'source_manifest.json')['siblings']:
        if entry['rfilename'] != '.gitattributes': assert verify(DEST / entry['rfilename'], entry)
    tokenizer = AutoTokenizer.from_pretrained(DEST, local_files_only=True)
    # Check ALL prompts before model execution; do not truncate long few-shot input.
    for task in tasks:
        text = tokenizer.apply_chat_template(task['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        task['rendered'] = text
        task['input_tokens'] = len(tokenizer.encode(text, add_special_tokens=False))
        assert task['input_tokens'] <= 4096, (task['task_id'], task['input_tokens'])
    write(OUT / 'environment.json', {'revision': REVISION, 'gpu': torch.cuda.get_device_name(),
        'versions': {p: importlib.metadata.version(p) for p in ('torch','transformers','accelerate','bitsandbytes')},
        'quantization': 'NF4 double quantization, bfloat16 compute', 'do_sample': False,
        'enable_thinking': False, 'input_token_range': [min(t['input_tokens'] for t in tasks), max(t['input_tokens'] for t in tasks)],
        'model_task_tokens': 8, 'probe_json_tokens': 80, 'task_sha256': sha(OUT / 'tasks.jsonl')})
    print('LOADING_MODEL', flush=True)
    model, info = Qwen3_5ForCausalLM.from_pretrained(DEST, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa', output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    write(OUT / 'loading_info.json', {k: sorted(v) if isinstance(v,set) else v for k,v in info.items()})
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
    assert not any('visual' in n for n,_ in model.named_modules())
    model.eval()
    done = set()
    result = OUT / 'predictions.jsonl'
    if result.exists():
        for line in result.read_text(encoding='utf-8').splitlines(): done.add(json.loads(line)['task_id'])
    with result.open('a', encoding='utf-8') as f:
        for i, task in enumerate(tasks):
            if task['task_id'] in done: continue
            inputs = tokenizer(task['rendered'], add_special_tokens=False, return_tensors='pt').to('cuda')
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=task['max_new_tokens'], do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            raw = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            row = {'task_id': task['task_id'], 'version': task['version'], 'pair_id': task['pair_id'],
                'raw': raw, 'prediction': int(raw) if raw in ('0','1') else None,
                'input_tokens': task['input_tokens'], 'seconds': time.perf_counter()-start,
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
            f.write(json.dumps(row) + '\n'); f.flush()
            print(f'{i+1}/{len(tasks)} ' + json.dumps(row), flush=True)
    print('INFERENCE_COMPLETE', flush=True)


def report():
    cases = {c['input']['pair_id']: c for c in read(OUT / 'cases_answers.json')}
    rows = [json.loads(x) for x in (OUT / 'predictions.jsonl').read_text().splitlines()]
    expected_ids = {json.loads(x)['task_id'] for x in (OUT / 'tasks.jsonl').read_text(encoding='utf-8').splitlines()}
    assert len(rows) == len(expected_ids) == len({r['task_id'] for r in rows})
    assert {r['task_id'] for r in rows} == expected_ids
    stats, review = {}, []
    for v in ('V0','V1','V2','V3'):
        rr = [r for r in rows if r['version'] == v]
        tp = sum(r['prediction'] == 1 and cases[r['pair_id']]['target'] == 1 for r in rr)
        fp = sum(r['prediction'] == 1 and cases[r['pair_id']]['target'] == 0 for r in rr)
        fn = sum(r['prediction'] != 1 and cases[r['pair_id']]['target'] == 1 for r in rr)
        correct = sum(r['prediction'] == cases[r['pair_id']]['target'] for r in rr)
        invalid = sum(r['prediction'] is None for r in rr)
        stats[v] = {'correct': correct, 'total': len(rr), 'tp':tp,'fp':fp,'fn':fn,'invalid':invalid,
                    'precision': tp/(tp+fp) if tp+fp else 0, 'recall':tp/(tp+fn),
                    'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                    'median_seconds':statistics.median(r['seconds'] for r in rr),
                    'peak_allocated_mib':max(r['peak_allocated_mib'] for r in rr),
                    'groups': {g: sum(r['prediction']==cases[r['pair_id']]['target'] for r in rr if cases[r['pair_id']]['group']==g)
                               for g in read(OUT / 'manifest.json')['groups']}}
        for r in rr:
            review.append({'version':v,'case':cases[r['pair_id']], 'result':r,'correct':r['prediction']==cases[r['pair_id']]['target']})
    probe_answers = read(OUT / 'probe_answers.json')
    probe_results = []
    for r in rows:
        if r['version'] != 'probe': continue
        a = probe_answers[r['pair_id']]
        try:
            parsed = json.loads(r['raw'])
            if a['kind'] == 'binary':
                ok = r['raw'] in ('0', '1') and int(r['raw']) == a['expected']
            elif a['kind'] == 'number':
                ok = type(parsed) in (float, int) and parsed == a['expected']
            else:
                ok = isinstance(parsed, dict) and parsed == a['expected']
        except (ValueError, TypeError): ok = False
        probe_results.append({**r, **a, 'correct':ok})
    by_version = {v:{r['pair_id']:r['prediction']==cases[r['pair_id']]['target'] for r in rows if r['version']==v} for v in stats}
    transitions = {v:{'fixed_vs_V0':sum(not by_version['V0'][p] and by_version[v][p] for p in cases),
                      'regressed_vs_V0':sum(by_version['V0'][p] and not by_version[v][p] for p in cases)} for v in ('V1','V2','V3')}
    write(OUT / 'metrics.json', {'versions':stats,'changes':transitions,'probes':probe_results,
          'invalid_policy':'Invalid output counts as incorrect; for binary TP/FP/FN it is an unselected candidate.'})
    write(OUT / 'review_cases.json', review)
    lines = ['# 训练侧提示词对照实验 v1', '', '固定 60 张训练图片、120 个配对（60 正、60 负）；不代表完整验证或比赛成绩。',
             '四条例题来自另外两张训练图片。未训练；未使用开发验证或 test 选择题目。', '',
             '| 版本 | 正确/120 | 漏选 FN | 错选 FP | 非法格式 | F1 | 单题中位秒 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for v,s in stats.items(): lines.append(f'| {v} | {s["correct"]} | {s["fn"]} | {s["fp"]} | {s["invalid"]} | {s["f1"]:.3f} | {s["median_seconds"]:.3f} |')
    lines += ['', 'V0：原版 JSON；V1：完整字段逐行表达且不舍入；V2：加明确的匹配规则；V3：再加四条正反例。',
              '', '## 相对原版的逐题变化', '', '| 版本 | 改对 | 改错 |', '|---|---:|---:|']
    for v,t in transitions.items(): lines.append(f'| {v} | {t["fixed_vs_V0"]} | {t["regressed_vs_V0"]} |')
    lines += ['', '## 独立能力探针', '', '| 探针 | 实际输出 | 正确 |', '|---|---|---|']
    for r in probe_results: lines.append(f'| {r["pair_id"]} | {r["raw"].replace(chr(10)," ").replace("|","/")} | {r["correct"]} |')
    lines += ['', '## 条件组（每组20题，描述可包含其他条件）', '', '| 条件组 | V0 | V1 | V2 | V3 |', '|---|---:|---:|---:|---:|']
    for g in stats['V0']['groups']: lines.append('| '+g+' | '+' | '.join(str(stats[v]['groups'][g]) for v in stats)+' |')
    lines += ['', '## 解释范围与复核', '',
        '这是刻意平衡、优先挑选单条件不满足负例的训练侧诊断集，统计不代表自然数据分布。分组依据标注属性，仅用于选题和事后分析，不进入模型输入。',
        'V1 同时改变布局并显式展示单位，因此观察到的改善只能归因于这组输入表达变化。V3 只测试一组例题，不代表任何少样例提示都有效。',
        '数值、查字段、提取条件等探针只有少量样例；答对不能证明该能力普遍可靠，答错也不能单凭最终输出确定内部原因。',
        '完整开发验证尚未运行。任何版本选择都应在独立开发验证中检查。', '',
        '查看 tasks.jsonl 可见每次实际消息（无本题答案）；cases_answers.json 是独立标准答案；predictions.jsonl 是模型原始回答；review_cases.json 是事后合并复核材料，不能直接输入模型。',
        'environment.json 记录软件和推理配置，manifest.json 记录选题清单统计及指纹。']
    (OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','run','report'])
    stage = parser.parse_args().stage
    {'prepare':prepare,'run':run,'report':report}[stage]()
