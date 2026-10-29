"""Prepare a small reviewed intermediate-supervision batch, without training."""
import collections
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/qwen2b_prompt_diagnostic_v1'
OUT = ROOT / 'runs/qwen2b_intermediate_supervision_v1'
# These descriptions were read against their properties in this preparation turn.
# Each selected image contributes its existing positive and negative pair.
REVIEWED = ['diagnostic-001','diagnostic-002','diagnostic-003','diagnostic-004',
            'diagnostic-005','diagnostic-007','diagnostic-014','diagnostic-016',
            'diagnostic-018','diagnostic-024','diagnostic-027','diagnostic-050']
FIELDS = {'category':('objectName','type'), 'color':('appearance','color'),
          'distance':('distance','distance_m'), 'length':('length','dimensions_m.length'),
          'width':('width','dimensions_m.width'), 'height':('height','dimensions_m.height'),
          'position':('place','image_region'), 'facing':('facing_direction','facing_convention'),
          'occlusion':('occlusion','occlusion')}
CN = dict(zip(FIELDS, ['类别','颜色','距离','长度','宽度','高度','画面位置','朝向','遮挡']))
SYSTEM = '''Match the English description against the candidate object.
Return JSON with attributes and match. attributes must contain category, color,
distance, length, width, height, position, facing, occlusion. Each attribute has
relevant (boolean), condition (object or null), satisfied (boolean or null).
Unmentioned attributes have relevant=false, condition=null, satisfied=null.
Category/color/position/facing/occlusion use condition {"op":"eq","value":...}.
Distance upper limits use {"op":"le","value":...,"unit":"m"}.
Size ranges use {"op":"between","min":...,"max":...,"inclusive":true,"unit":"m"}.
Read category from type, color from color, distance from distance_m, dimensions
from dimensions_m, position from image_region, facing from facing_convention,
and occlusion from occlusion. Generic vehicles do not restrict a specific type.
Use the provided geometric and occlusion meanings; do not recompute them.
For this reviewed batch, all stated conditions must hold. match is a boolean.
Output only JSON. Do not infer restrictions on unmentioned attributes.
'''


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def issues(case):
    text = case['input']['description'].lower()
    props = case['properties_for_audit_only']
    result = []
    # Screening, not a complete semantic parser. Decimal makes 55 and 55.0 equal.
    nums = lambda s: collections.Counter(Decimal(v) for v in re.findall(r'\d+(?:\.\d+)?',s))
    pnums = nums(' '.join(v for k,v in props.items() if k in ('height','length','width','distance')))
    if nums(text) != pnums:
        result.append('description/property numeric values differ; approximation may explain but not silently resolved')
    if 'objectName' in props:
        category = props['objectName']
        patterns = {'car':r'\bcars?\b','van':r'\bvans?\b','truck':r'\btrucks?\b','bus':r'\bbus(?:es)?\b'}
        if not re.search(patterns[category],text): result.append('property requires category absent from description')
    if 'distance' in props and 'within' not in text:
        result.append('distance is stated without explicit upper-bound wording; interpretation requires review')
    return result


def get_value(candidate, path):
    for part in path.split('.'): candidate = candidate[part]
    return candidate


def target(case):
    props = case['properties_for_audit_only']
    candidate = case['input']['candidate']
    attributes = {}
    for name,(key,path) in FIELDS.items():
        if key not in props:
            attributes[name] = {'relevant':False,'condition':None,'satisfied':None}
            continue
        actual = get_value(candidate,path)
        if key in ('length','width','height'):
            lo,hi = map(float,re.findall(r'\d+(?:\.\d+)?',props[key]))
            condition = {'op':'between','min':lo,'max':hi,'inclusive':True,'unit':'m'}
            satisfied = lo <= actual <= hi
        elif key == 'distance':
            limit = float(props[key].split()[0])
            condition = {'op':'le','value':limit,'unit':'m'}
            satisfied = actual <= limit
        else:
            condition = {'op':'eq','value':props[key]}
            satisfied = actual == props[key]
        attributes[name] = {'relevant':True,'condition':condition,'satisfied':satisfied}
    match = all(v['satisfied'] for v in attributes.values() if v['relevant'])
    assert match == bool(case['target']), 'Intermediate judgments disagree with original label'
    return {'attributes':attributes,'match':match}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases = read(SOURCE / 'cases_answers.json')
    by_id = {c['input']['pair_id']:c for c in cases}
    keys = {(by_id[i]['source_file'],by_id[i]['input']['description']) for i in REVIEWED}
    selected = [c for c in cases if (c['source_file'],c['input']['description']) in keys]
    train = set(read(ROOT / 'runs/20261015_qwen2b_input_v1/split_manifest.json')['train'])
    assert len(selected)==24 and len({c['source_file'] for c in selected})==12
    assert sum(c['target'] for c in selected)==12
    assert all(c['source_file'] in train and not issues(c) for c in selected)
    flagged = [{'pair_id':c['input']['pair_id'],'source_file':c['source_file'],
                'description':c['input']['description'],'properties':c['properties_for_audit_only'],
                'reasons':issues(c)} for c in cases if issues(c)]
    write(OUT / 'quarantine.json',flagged)
    write(OUT / 'review_decisions.json',[
        {'source_file':by_id[i]['source_file'],'description':by_id[i]['input']['description'],
         'properties':by_id[i]['properties_for_audit_only'],
         'review':'Assistant read description against properties; no omitted category or numeric discrepancy found. Not independent human verification.'}
        for i in REVIEWED])
    inputs, answers, review = [],[],[]
    for c in selected:
        pid = c['input']['pair_id']
        user = json.dumps({k:c['input'][k] for k in ('description','candidate')},ensure_ascii=False)
        messages = [{'role':'system','content':SYSTEM},{'role':'user','content':user}]
        answer = target(c)
        inputs.append({'pair_id':pid,'messages':messages})
        answers.append({'pair_id':pid,'target':answer,'binary_target':c['target']})
        review.append({'pair_id':pid,'source_file':c['source_file'],'input':c['input'],
                       'target':answer,'original_binary_target':c['target']})
    for name,rows in [('inputs',inputs),('answers',answers)]:
        with (OUT / (name+'.jsonl')).open('w',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    # Training fixture: clearly separated from inference inputs. No trainer yet.
    with (OUT / 'sft_preview.jsonl').open('w',encoding='utf-8') as f:
        for x,y in zip(inputs,answers):
            assert x['pair_id']==y['pair_id']
            f.write(json.dumps({'pair_id':x['pair_id'],'messages':x['messages']+
                [{'role':'assistant','content':json.dumps(y['target'],ensure_ascii=False,separators=(',',':'))}]},ensure_ascii=False)+'\n')
    write(OUT / 'review_cases.json',review)
    relevant = collections.Counter(k for a in answers for k,v in a['target']['attributes'].items() if v['relevant'])
    assert set(relevant)==set(FIELDS)
    for key in keys:
        aa=[target(c)['attributes'] for c in selected if (c['source_file'],c['input']['description'])==key]
        assert len(aa)==2
        assert {k:(v['relevant'],v['condition']) for k,v in aa[0].items()} == {k:(v['relevant'],v['condition']) for k,v in aa[1].items()}
    write(OUT / 'checks.json', {'status':'PASS','pairs':24,'images':12,'positive':12,'negative':12,
        'attribute_coverage':dict(relevant),'screened_pairs':len(cases),'flagged_pairs':len(flagged),
        'flagged_descriptions':len({(x['source_file'],x['description']) for x in flagged}),
        'not_flagged_is_not_certified':True,'remaining_unreviewed_pairs':len(cases)-len(flagged)-24,
        'checks':['training split only','all nine attributes covered','relevance/conditions invariant across same-description objects',
                  'unmentioned satisfied is null','intermediate conjunction agrees with original binary membership',
                  'input user payload contains only original description and candidate'],
        'scope':'Small preview only. No optimizer steps, no validation score, no automatic full-dataset approval.'})
    lines=['# 中间监督标注示例（24对、12张训练图片）','','尚未训练。这份材料先让人核对训练答案是否合理。',
        '相关性表示描述是否提及该属性；满足性表示当前候选是否满足要求。未提及时满足性为“不适用”，不是“不满足”。',
        '属性条件来自现有标注，经本批逐句核对；最终0/1保留原始成员标签，逐项条件的合取必须与其一致。',
        '这批只含能用全部条件同时满足表示的描述；不声称已处理任意否定、或者、参照物关系。','']
    for r in review:
        lines += ['## '+r['pair_id'],'',r['input']['description'],'',
                  '| 属性 | 相关 | 要求 | 候选值 | 满足 |','|---|---|---|---|---|']
        for name,v in r['target']['attributes'].items():
            actual=get_value(r['input']['candidate'],FIELDS[name][1])
            cond=v['condition']
            if cond is None: requirement='未提及'
            elif cond['op']=='between': requirement=f'{cond["min"]} ≤ 值 ≤ {cond["max"]} 米'
            elif cond['op']=='le': requirement=f'≤ {cond["value"]} 米'
            else: requirement=str(cond['value'])
            shown=f'{actual:.6f}' if isinstance(actual,float) else str(actual)
            lines.append(f'| {CN[name]} | {"是" if v["relevant"] else "否"} | {requirement} | {shown} | {"不适用" if v["satisfied"] is None else "是" if v["satisfied"] else "否"} |')
        lines += ['',f'原标注最终答案：{r["original_binary_target"]}；逐条件计算结果：{int(r["target"]["match"])}。','']
    lines += ['## 适用边界','','表内候选数值为阅读方便显示到6位小数，JSON保留原值，判断使用原值。',
              '目前只有car、van、bus示例，没有覆盖所有类别与颜色；24对不是充分训练集。',
              '输入中不包含public_properties或本题答案。sft_preview.jsonl明确是带标准回答的训练材料，不能当作推理输入。']
    (OUT / 'examples.md').write_text('\n'.join(lines),encoding='utf-8')
    write(OUT / 'provenance.json',{'source_sha256':digest(SOURCE/'cases_answers.json'),
         'script_sha256':digest(Path(__file__)),'selection':REVIEWED,
         'label_policy':'No raw annotations changed. Exclude flagged/ambiguous cases from this pilot. No automatic correction.'})
    print(json.dumps(read(OUT/'checks.json'),indent=2))


if __name__=='__main__': main()
