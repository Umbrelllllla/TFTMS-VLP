"""Append explicit field rules to frozen V2 messages; reuse the same inference engine."""
import argparse
import json
import statistics
from pathlib import Path
import diagnose as base

SOURCE = base.OUT
OUT = base.ROOT / 'runs/qwen2b_prompt_diagnostic_v4'
RULES = '''
FIELD-BY-FIELD MATCHING GUIDE
Interpret the meaning of the description before applying these correspondences.
Do not merely search for words: a negated attribute is excluded, and an attribute
of a reference object must not be assigned to the candidate target.

1. CATEGORY: car, van, truck and bus refer to candidate.type. Match the requested
category exactly. A generic vehicle does not restrict the candidate to car.
If the target is described as not a bus, candidate.type must not be bus.
2. COLOR: white, black, red and other color terms refer to candidate.color.
Match the target's required color; reject a color explicitly excluded.
3. DISTANCE: near/far distance constraints in metres refer to distance_m,
not camera_position_m.z, bounding-box coordinates or dimensions_m.length.
4. SIZE: long/length refers to dimensions_m.length; wide/width refers to
dimensions_m.width; tall/height refers to dimensions_m.height. All are metres.
For the appropriate field v, at most / within / no more than B means v <= B;
at least / no less than A means v >= A; more than A means v > A;
less than B means v < B; A to B / between A and B means A <= v <= B.
Both inequalities of a range must hold. Do not round values before comparison
and do not invent extra tolerance for approximately in this benchmark.
5. IMAGE POSITION: top left, top, top right, left, center, right, bottom left,
bottom, bottom right refer to image_region. Use that already computed region.
These nine regions are distinct: left means the middle row's left cell;
top left is not the same as left or top. The origin is the image's top-left;
x increases rightward and y downward. Region uses box center and thirds of
image width/height, NOT the box's bottom-right corner and NOT half-image splits.
Do not recompute a different region from bbox_xyxy_px.
6. FACING: facing me, facing away from me and side facing me refer to
facing_convention, not image_region, color, or observation_angle_rad.
Match the full direction phrase; facing away from me is not facing me.
7. OCCLUSION: not occluded / no occluded means occlusion = no occluded (code 0);
partially occluded means occlusion = partially occluded (code 1).
Use occlusion or occlusion_code, NOT truncation_code. Truncation is a different
attribute. Do not invent a meaning for unknown occlusion codes.

COMBINE CONDITIONS: separate requirements joined by and must ALL hold.
For an explicit A or B alternative, at least one of A and B must hold;
other requirements still apply. Negation reverses only the condition it modifies.
Never add a restriction on an unmentioned field.
Check every stated requirement against its corresponding candidate value.
Return 1 only when the complete description is satisfied. Otherwise return 0.
Output exactly one digit, 0 or 1, with no explanation.
'''


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'tasks.jsonl').exists():
        raise RuntimeError('V4 tasks already frozen; use run/report or a new version.')
    originals = [json.loads(line) for line in (SOURCE / 'tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    originals = [t for t in originals if t['version'] == 'V2']
    assert len(originals) == 120
    tasks = []
    for old in originals:
        new = json.loads(json.dumps(old))
        new['version'] = 'V4'
        new['task_id'] = 'V4:' + old['pair_id']
        assert len(new['messages']) == 2
        new['messages'][0]['content'] += RULES
        assert new['messages'][1] == old['messages'][1]
        assert new['max_new_tokens'] == old['max_new_tokens'] == 8
        tasks.append(new)
    with (OUT / 'tasks.jsonl').open('w', encoding='utf-8') as f:
        for task in tasks: f.write(json.dumps(task, ensure_ascii=False) + '\n')
    (OUT / 'added_rules.txt').write_text(RULES, encoding='utf-8')
    (OUT / 'full_system_prompt.txt').write_text(tasks[0]['messages'][0]['content'], encoding='utf-8')
    base.write(OUT / 'manifest.json', {'version':'V4', 'cases':120,
        'task_sha256':base.sha(OUT / 'tasks.jsonl'),
        'source_hashes':{name:base.sha(SOURCE / name) for name in
                         ('tasks.jsonl','cases_answers.json','predictions.jsonl','environment.json')},
        'change':'Append field-specific semantic and comparison rules to V2 system message only.',
        'checks':'PASS: all 120 user messages, pair IDs, order and token budgets identical to V2.',
        'limitations':'Same training-side diagnostic set, now used for iterative prompt design. No held-out claim.'})
    print('PREPARED: 120 tasks; only system message changed.', flush=True)


def check_source():
    for name, expected in base.read(OUT / 'manifest.json')['source_hashes'].items():
        assert base.sha(SOURCE / name) == expected, name


def report():
    check_source()
    cases = {c['input']['pair_id']:c for c in base.read(SOURCE / 'cases_answers.json')}
    rows = [json.loads(x) for x in (OUT / 'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['pair_id'] for r in rows}) == 120
    assert {r['pair_id'] for r in rows} == set(cases)
    assert all(r['task_id'] == 'V4:' + r['pair_id'] for r in rows)
    for k in ('revision','gpu','versions','quantization','do_sample','enable_thinking','model_task_tokens'):
        assert base.read(OUT / 'environment.json')[k] == base.read(SOURCE / 'environment.json')[k], k
    tp = sum(r['prediction']==1 and cases[r['pair_id']]['target']==1 for r in rows)
    fp = sum(r['prediction']==1 and cases[r['pair_id']]['target']==0 for r in rows)
    fn = sum(r['prediction']!=1 and cases[r['pair_id']]['target']==1 for r in rows)
    correct = sum(r['prediction']==cases[r['pair_id']]['target'] for r in rows)
    invalid = sum(r['prediction'] is None for r in rows)
    stat = {'correct':correct,'total':120,'tp':tp,'fp':fp,'fn':fn,'invalid':invalid,
            'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn),
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            'median_seconds':statistics.median(r['seconds'] for r in rows),
            'peak_allocated_mib':max(r['peak_allocated_mib'] for r in rows)}
    old_rows = [json.loads(x) for x in (SOURCE / 'predictions.jsonl').read_text().splitlines()]
    old = {r['pair_id']:r for r in old_rows if r['version']=='V2'}
    review = [{'case':cases[r['pair_id']], 'V2':old[r['pair_id']], 'V4':r,
               'V2_correct':old[r['pair_id']]['prediction']==cases[r['pair_id']]['target'],
               'V4_correct':r['prediction']==cases[r['pair_id']]['target']} for r in rows]
    change = {'fixed':sum(not r['V2_correct'] and r['V4_correct'] for r in review),
              'regressed':sum(r['V2_correct'] and not r['V4_correct'] for r in review),
              'V2_false_positives_fixed':sum(r['case']['target']==0 and r['V2']['prediction']==1 and r['V4']['prediction']==0 for r in review)}
    groups = {}
    for g in base.read(SOURCE / 'manifest.json')['groups']:
        rr = [r for r in review if r['case']['group']==g]
        groups[g] = {'total':len(rr),'V2_correct':sum(r['V2_correct'] for r in rr),
                     'V4_correct':sum(r['V4_correct'] for r in rr)}
    assert correct == 120-fp-fn-sum(r['prediction'] is None and cases[r['pair_id']]['target']==0 for r in rows)
    base.write(OUT / 'metrics.json', {'V4':stat,'changes_vs_V2':change,'groups':groups})
    base.write(OUT / 'review_cases.json', review)
    lines = ['# V4 字段对应规则实验', '',
        '在 V2 系统提示词后增加明确的字段对应、数值上下界、位置/朝向/遮挡、否定和或者规则。',
        '保持同一批120题、原始描述、候选资料、逐行输入、模型、四位量化、非思考模式和输出预算不变。没有增加例题，没有训练。', '',
        '| 版本 | 正确/120 | 漏选 | 错选 | 格式错误 | F1 |', '|---|---:|---:|---:|---:|---:|']
    previous = base.read(SOURCE / 'metrics.json')['versions']
    for v,s in [('V0',previous['V0']),('V2',previous['V2']),('V4',stat)]:
        lines.append(f'| {v} | {s["correct"]} | {s["fn"]} | {s["fp"]} | {s["invalid"]} | {s["f1"]:.3f} |')
    lines += ['', f'相对 V2：改对 {change["fixed"]} 题，改错 {change["regressed"]} 题；原来20个错选中修正 {change["V2_false_positives_fixed"]} 个。',
              '', '| 条件组（每组20题） | V2正确 | V4正确 |', '|---|---:|---:|']
    for g,s in groups.items(): lines.append(f'| {g} | {s["V2_correct"]} | {s["V4_correct"]} |')
    lines += ['', f'V4 单题耗时中位数 {stat["median_seconds"]:.3f} 秒，PyTorch 分配显存峰值 {stat["peak_allocated_mib"]:.1f} MiB。速度为不同运行的描述性统计，非严格性能基准。',
        '', '## 适用范围', '',
        '这120题是训练侧诊断集，已用于多轮提示词设计，不能作为独立验证或比赛成绩。结果只支持这一整组规则的效果，不能确定某一句规则单独有效。否定和或者规则是本轮统一加入的，但未单独构造测试集评估其效果。',
        '几何定义不变：使用已计算的九宫格 image_region，而不是右下角像素或二等分。任何标注属性和本题答案都没有加入模型消息。',
        '', '## 复核入口', '',
        '`added_rules.txt`：本轮新增英文规则；`full_system_prompt.txt`：完整提示词；`tasks.jsonl`：所有实际消息；`predictions.jsonl`：原始回答；`review_cases.json`：V2/V4逐题对照（仅供事后复核，不能作为模型输入）。',
        '旧版文件未改，manifest.json记录旧版文件指纹，报告前已重新核对。']
    (OUT / 'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'V4':stat,'change':change,'groups':groups},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','run','report'])
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'report': report()
    else:
        check_source()
        base.OUT = OUT
        base.run()
