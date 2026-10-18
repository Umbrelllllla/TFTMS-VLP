"""Strict, reproducible scoring for the frozen small supervision experiment."""
import collections
import json
import math
import prepare_supervision as sup

DATA = sup.ROOT / 'runs/qwen2b_supervision_ab_v1'


def strict_json(raw):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result: raise ValueError('duplicate key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=unique)


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def parse(raw, arm):
    if arm == 'A':
        return (int(raw), None) if raw in ('0', '1') else (None, None)
    try:
        x = strict_json(raw)
        assert set(x) == {'attributes', 'match'} and type(x['match']) is bool
        assert set(x['attributes']) == set(sup.FIELDS)
        for field, a in x['attributes'].items():
            assert set(a) == {'relevant', 'condition', 'satisfied'} and type(a['relevant']) is bool
            c = a['condition']
            if not a['relevant']:
                assert c is None and a['satisfied'] is None
                continue
            assert type(a['satisfied']) is bool and type(c) is dict
            if field in ('distance',):
                assert set(c) == {'op', 'value', 'unit'} and c['op'] == 'le' and c['unit'] == 'm' and number(c['value'])
            elif field in ('length', 'width', 'height'):
                assert set(c) == {'op', 'min', 'max', 'inclusive', 'unit'}
                assert c['op'] == 'between' and c['unit'] == 'm' and c['inclusive'] is True
                assert number(c['min']) and number(c['max']) and c['min'] <= c['max']
            else:
                assert set(c) == {'op', 'value'} and c['op'] == 'eq' and type(c['value']) is str
        return int(x['match']), x
    except (AssertionError, ValueError, TypeError, KeyError):
        return None, None


def score(cases, predictions, arm):
    assert [p['pair_id'] for p in predictions] == [c['input']['pair_id'] for c in cases]
    count = collections.Counter(); aux = collections.Counter(); details = []
    for c, p in zip(cases, predictions):
        pred, structured = parse(p['raw'], arm)
        truth = c['target']; correct = pred is not None and pred == truth
        count['correct'] += correct; count['invalid'] += pred is None
        # Missing/invalid predictions select nothing for retrieval, but are always incorrect for accuracy.
        selected = pred == 1
        count['tp'] += selected and truth == 1
        count['fp'] += selected and truth == 0
        count['fn'] += not selected and truth == 1
        count['tn_unselected'] += not selected and truth == 0
        count['invalid_negative'] += pred is None and truth == 0
        count['hit_token_cap'] += p['hit_token_cap']
        errors = []
        if arm == 'B':
            gold = c['structured_target']
            # Diagnostic only: separate the final task answer from strict auxiliary schema compliance.
            # Primary metrics above retain the frozen strict policy, without repairing answers.
            try:
                loose = strict_json(p['raw'])
                valid_match = type(loose) is dict and type(loose.get('match')) is bool
            except (ValueError, TypeError):
                valid_match = False
            aux['readable_match_fields'] += valid_match
            aux['match_field_only_correct'] += valid_match and int(loose['match']) == truth
            aux['exact_output'] += structured == gold
            for field, expected in gold['attributes'].items():
                actual = structured['attributes'][field] if structured else None
                aux['all_attribute_slots'] += 1
                for key in ('relevant', 'condition', 'satisfied'):
                    ok = actual is not None and actual[key] == expected[key]
                    aux[key + '_correct'] += ok
                    if not ok: errors.append(field + '.' + key)
                    if expected['relevant']:
                        aux['mentioned_' + key + '_correct'] += ok
                if expected['relevant']: aux['mentioned_slots'] += 1
            if structured:
                aux['valid_outputs'] += 1
                conjunction = all(a['satisfied'] for a in structured['attributes'].values() if a['relevant'])
                aux['internally_inconsistent'] += structured['match'] != conjunction
        details.append({'pair_id':p['pair_id'], 'description':c['input']['description'],
                        'candidate':c['input']['candidate'], 'target':truth, 'prediction':pred,
                        'correct':correct, 'raw':p['raw'], 'attribute_errors':errors})
    n = len(cases); tp, fp, fn = (count[k] for k in ('tp','fp','fn'))
    result = dict(count, total=n, accuracy=count['correct']/n,
                  precision=tp/(tp+fp) if tp+fp else 0,
                  recall=tp/(tp+fn) if tp+fn else 0,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                  inference_seconds=sum(p['seconds'] for p in predictions), auxiliary=dict(aux))
    return result, details


def main():
    manifest = sup.read(DATA/'manifest.json')
    for name, digest in manifest['data_sha256'].items(): assert sup.digest(DATA/(name+'.json')) == digest
    cases = sup.read(DATA/'validation.json'); results = {}; details = {}; configs = {}
    for arm in ('A','B'):
        assert sup.read(DATA/arm/'complete.json')['status'] == 'PASS'
        configs[arm] = sup.read(DATA/arm/'config.json')
        for stage in ('before','after'):
            predictions = [json.loads(line) for line in (DATA/arm/(stage+'.predictions.jsonl')).read_text(encoding='utf-8').splitlines()]
            key = arm+'_'+stage
            results[key], details[key] = score(cases,predictions,arm)
    assert sup.read(DATA/'A/initialization.json') == sup.read(DATA/'B/initialization.json')
    for key in configs['A']:
        if key not in ('arm','max_new_tokens','training_supervised_tokens'):
            assert configs['A'][key] == configs['B'][key], key
    a = sup.read(DATA/'A/training_inputs.json'); b = sup.read(DATA/'B/training_inputs.json')
    assert [r['pair_id'] for r in a] == [r['pair_id'] for r in b]
    assert all(x['messages'][1] == y['messages'][1] for x,y in zip(a,b))
    changes = {}
    for arm in ('A','B'):
        changes[arm] = {'fixed':[y['pair_id'] for x,y in zip(details[arm+'_before'],details[arm+'_after']) if not x['correct'] and y['correct']],
                        'regressed':[y['pair_id'] for x,y in zip(details[arm+'_before'],details[arm+'_after']) if x['correct'] and not y['correct']]}
    report = {'metrics':results,'paired_changes':changes,'configs':configs,
              'checks':'same initialization, base revision, train order, user inputs, data hashes and training hyperparameters',
              'limits':'24 pairs from 12 development images; automatic conservative screening; not full benchmark; equal updates but unequal supervised tokens and compute'}
    sup.write(DATA/'results.json',report); sup.write(DATA/'scored_cases.json',details)
    lines = ['# 小规模中间监督 A/B 对照', '', '训练：64张图片128对；验证：12张图片24对，正负各半。原有相机隔离划分保持不变。',
             '', '|方式|正确|误选 FP|漏选 FN|格式错误|F1|', '|---|---:|---:|---:|---:|---:|']
    names = {'A_before':'只答0/1：训练前','A_after':'只答0/1：训练后','B_before':'逐属性JSON：训练前','B_after':'逐属性JSON：训练后'}
    for key,r in results.items(): lines.append(f"|{names[key]}|{r['correct']}/24|{r['fp']}|{r['fn']}|{r['invalid']}|{r['f1']:.3f}|")
    lines += ['', '格式错误计准确率错误；检索指标中按未选择处理。未选择的负例仍可能格式错误，因此不能直接用 TP+TN_unselected 计算准确率。',
              '', '两组每组128次更新，训练样本及顺序相同，初始化校验一致。B的答案更长，训练答案token和耗时更多，不是等计算量比较。',
              '', '这是经过保守自动筛选的小型开发子集，不是完整验证集或比赛成绩；同一图片的正负两例也不是独立场景。辅助答案由现有properties生成并与文字解析核对，不等于全部经过人工标注。',
              '', '## 中间字段', '', '下面分母包含非法回答，不能通过只统计成功解析的回答抬高结果。']
    for stage in ('before','after'):
        r=results['B_'+stage]; z=r['auxiliary']
        lines += ['', f"B {stage}：相关性正确 {z['relevant_correct']}/{z['all_attribute_slots']}；真实提及条件提取正确 {z['mentioned_condition_correct']}/{z['mentioned_slots']}；真实提及条件满足性正确 {z['mentioned_satisfied_correct']}/{z['mentioned_slots']}；整份JSON完全正确 {z['exact_output']}/24；有效回答内部矛盾 {z.get('internally_inconsistent',0)}。", '', f"补充诊断：只读取完整JSON中的布尔match字段，可读取{z['readable_match_fields']}/24，最终答案正确{z['match_field_only_correct']}/24。这允许其他中间字段不符合约定，与上表严格管线评分分开报告，避免把结构错误一概说成最终匹配错误。"]
    lines += ['', '## 更新与成本']
    for arm in ('A','B'):
        run=sup.read(DATA/arm/'complete.json')
        lines += ['',f"{arm}：修正{len(changes[arm]['fixed'])}题，新增错误{len(changes[arm]['regressed'])}题；训练答案 {configs[arm]['training_supervised_tokens']} token；训练{run['training_seconds']:.1f}秒；训练显存分配峰值{run['peak_allocated_mib']/1024:.2f}GiB。"]
    lines += ['', '## B训练后的错误案例']
    for r in details['B_after']:
        if r['correct'] and not r['attribute_errors']: continue
        lines += ['', '### '+r['pair_id'], '', r['description'], '', f"最终答案：标准{r['target']}，预测{r['prediction']}；字段差异：{', '.join(r['attribute_errors']) or '无'}", '', '```json', json.dumps(r['candidate'],ensure_ascii=False,indent=2), '```', '', '模型原始输出：', '```text', r['raw'], '```']
    (DATA/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status_path=sup.ROOT/'docs/status.md'
    status=status_path.read_text(encoding='utf-8')
    marker='当前：正在执行首次小规模监督A/B对照。'
    if marker in status:
        status=status[:status.index(marker)]
        ra,rb=results['A_after'],results['B_after']
        status+=('当前：首次小规模监督A/B对照已完成，两组各128次更新，数据指纹、初始化、训练顺序和基础模型冻结检查通过。'
                 f"A只答0/1：{results['A_before']['correct']}/24→{ra['correct']}/24；"
                 f"B逐属性JSON（严格管线评分）：{results['B_before']['correct']}/24→{rb['correct']}/24。"
                 f"B训练后只读取最终match字段时正确{rb['auxiliary']['match_field_only_correct']}/24。"
                 '两种口径分别记录，不能将中间结构错误等同于最终匹配错误。'
                 '这是经过自动筛选的12张开发图片的小型对照，不是完整验证或比赛成绩。'
                 '详见[实验说明](supervision_ab.md)及[结果与错误案例](../runs/qwen2b_supervision_ab_v1/README.md)。\n')
        status_path.write_text(status,encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
