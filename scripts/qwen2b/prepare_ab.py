"""Conservative text/property screening and frozen small A/B data selection."""
import collections
import importlib.util
import json
from pathlib import Path
import random
import re
import prepare_supervision as sup

ROOT=sup.ROOT
OUT=ROOT/'runs/qwen2b_supervision_ab_v1'
NUM=r'(\d+(?:\.\d+)?)'
COLORS=['black','blue','dark-brown','green','red','silver-grey','white','yellow-orange']
REGIONS=['top left','top right','bottom left','bottom right','top','bottom','left','right','center']
GROUPS={'size':{'height','width','length'},'attribute':{'objectName','appearance'},
        'distance':{'distance'},'place':{'place'},'facing':{'facing_direction'},'occlusion':{'occlusion'}}
ALLOWED=set(('vehicles vehicle which are is have has a an the of from to and with in on at scene area side '
 'located positioned present identified as appearance appear facing face towards me away not occluded occlusion '
 'partially without any being approximately within ranging range ranges length lengths width widths height heights '
 'meters metres car cars van vans truck trucks bus buses top left right bottom center black blue dark brown '
 'green red silver grey white yellow orange').split())


def extract(text):
    t=text.lower().replace('centre','center')
    words=set(re.findall(r'[a-z]+',t))
    if words-ALLOWED: raise ValueError('unsupported_words:'+','.join(sorted(words-ALLOWED)))
    parsed={}
    spans=[]
    for match in re.finditer(r'\b(widths?|lengths?|heights?)\b[^0-9]{0,55}'+NUM+r'\s+to\s+'+NUM+r'\s+(?:meters|metres)\b',t):
        key=match[1].rstrip('s')
        if key in parsed: raise ValueError('duplicate_size')
        parsed[key]=[float(match[2]),float(match[3])]; spans.append(match.span())
    for match in re.finditer(NUM+r'\s+to\s+'+NUM+r'\s+(?:meters|metres)\s+in\s+(width|length|height)\b',t):
        if match[3] in parsed: raise ValueError('duplicate_size')
        parsed[match[3]]=[float(match[1]),float(match[2])]; spans.append(match.span())
    distances=list(re.finditer(r'\bwithin\s+(?:approximately\s+)?'+NUM+r'\s+(?:meters|metres)\b',t))
    if len(distances)>1: raise ValueError('multiple_distances')
    if distances: parsed['distance']=float(distances[0][1]); spans.append(distances[0].span())
    remaining=list(t)
    for a,b in spans: remaining[a:b]=' '*(b-a)
    if re.search(r'\d',''.join(remaining)): raise ValueError('unbound_numeric_condition')
    for key,pattern in [('car',r'\bcars?\b'),('van',r'\bvans?\b'),('truck',r'\btrucks?\b'),('bus',r'\bbus(?:es)?\b')]:
        if re.search(pattern,t):
            if 'objectName' in parsed: raise ValueError('multiple_categories')
            parsed['objectName']=key
    for color in COLORS:
        if re.search(r'\b'+re.escape(color)+r'\b',t):
            if 'appearance' in parsed: raise ValueError('multiple_colors')
            parsed['appearance']=color
    positions=t
    for region in REGIONS:
        pattern=r'\b'+region+r'\b'
        if re.search(pattern,positions):
            if 'place' in parsed: raise ValueError('multiple_positions')
            parsed['place']=region; positions=re.sub(pattern,' ',positions)
    if 'facing away from me' in t: parsed['facing_direction']='facing away from me'
    elif 'side facing me' in t: parsed['facing_direction']='side facing me'
    elif re.search(r'(?:facing|face) (?:towards )?me',t): parsed['facing_direction']='facing me'
    if 'partially occluded' in t: parsed['occlusion']='partially occluded'
    elif re.search(r'(?:not occluded|without (?:any )?(?:being )?(?:occluded|occlusion)|no occlusion)',t): parsed['occlusion']='no occluded'
    if ('occlu' in t) != ('occlusion' in parsed): raise ValueError('unparsed_occlusion')
    leftover=re.sub(r'not occluded|without (?:any )?(?:being )?(?:occluded|occlusion)','',t)
    if re.search(r'\b(?:not|without|or)\b',leftover): raise ValueError('unsupported_logic')
    for key in ('height','width','length'):
        if bool(re.search(r'\b'+key+r's?\b',t)) != (key in parsed): raise ValueError('unparsed_size')
    if ('facing' in t or 'face ' in t) != ('facing_direction' in parsed): raise ValueError('unparsed_facing')
    return parsed


def normalized(props):
    return {k: [float(v) for v in re.findall(r'\d+(?:\.\d+)?',value)] if k in ('height','width','length')
            else float(value.split()[0]) if k=='distance' else value for k,value in props.items()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists(): raise RuntimeError('Frozen data already exists.')
    audit=ROOT/'runs/20261015_qwen2b_data_audit'
    data=ROOT/'runs/20261015_qwen2b_input_v1'
    spec=importlib.util.spec_from_file_location('feature_prep',data/'prepare.py'); feature=importlib.util.module_from_spec(spec); spec.loader.exec_module(feature)
    rs=sup.read(audit/'training_records.json'); split=sup.read(data/'split_manifest.json')
    train=set(split['train']); val=set(split['validation'])
    assert not train&val
    old={c['source_file'] for c in sup.read(ROOT/'runs/qwen2b_prompt_diagnostic_v1/cases_answers.json')}
    old.update(c['source_file'] for c in sup.read(ROOT/'runs/qwen2b_prompt_diagnostic_v1/demonstrations.json'))
    pools={'train':[],'validation':[]}; rejected=[]; counts=collections.Counter()
    for r in rs:
        sn='train' if r['file'] in train else 'validation'
        for ai,a in enumerate(r['annotations']):
            try:
                parsed=extract(a['text'])
                if parsed!=normalized(a['properties']): raise ValueError('text_property_disagreement')
                counts[sn+'_text_pass']+=1
                if sn=='train' and r['file'] in old:
                    counts['prior_diagnostic_image_excluded']+=1; continue
                pos=set(a['positive_indices']); neg=set(range(len(r['objects'])))-pos
                if not pos or not neg: counts[sn+'_no_both_classes']+=1; continue
                cs=[]
                for oi,o in enumerate(r['objects']):
                    c={'input':{'description':a['text'],'candidate':feature.features(o,r['width'],r['height'])},
                       'target':int(oi in pos),'properties_for_audit_only':a['properties'],
                       'source_file':r['file'],'camera':r['camera'],'annotation_index':ai,'object_index':oi}
                    c['structured_target']=sup.target(c)
                    cs.append(c)
                pools[sn].append(cs)
            except (ValueError,AssertionError) as e:
                rejected.append({'split':sn,'file':r['file'],'annotation_index':ai,'reason':str(e) or 'membership_disagreement'})
    rng=random.Random(20261017)
    result={}
    for sn,quotas in [('train',[12,12,10,10,10,10]),('validation',[2,2,2,2,2,2])]:
        pool=pools[sn][:]; rng.shuffle(pool); used=set(); selected=[]
        for (group,keys),quota in zip(GROUPS.items(),quotas):
            got=0
            for cs in pool:
                if cs[0]['source_file'] in used or not keys&cs[0]['properties_for_audit_only'].keys(): continue
                positives=[c for c in cs if c['target']]; negatives=[c for c in cs if not c['target']]
                rng.shuffle(negatives)
                negative=min(negatives,key=lambda c:sum(v['satisfied'] is False for v in c['structured_target']['attributes'].values()))
                for c in [rng.choice(positives),negative]: c['group']=group; selected.append(c)
                used.add(cs[0]['source_file']); got+=1
                if got==quota: break
            assert got==quota,(sn,group,got)
        rng.shuffle(selected)
        for i,c in enumerate(selected): c['input']['pair_id']=f'{sn}-{i:03d}'
        result[sn]=selected
        sup.write(OUT/(sn+'.json'),selected)
    assert len(result['train'])==128 and len(result['validation'])==24
    assert not {c['camera'] for c in result['train']}&{c['camera'] for c in result['validation']}
    sup.write(OUT/'screening_rejections.json',rejected)
    sup.write(OUT/'screening_summary.json',dict(counts)|{'rejected_annotations':len(rejected),'eligible_train_annotations':len(pools['train']),'eligible_validation_annotations':len(pools['validation'])})
    sup.write(OUT/'manifest.json',{'seed':20261017,'train_pairs':128,'train_images':64,'validation_pairs':24,'validation_images':12,
        'train_positive':64,'validation_positive':12,'steps_per_arm':128,'epochs':1,
        'data_sha256':{sn:sup.digest(OUT/(sn+'.json')) for sn in result},
        'source_records_sha256':sup.digest(audit/'training_records.json'),'original_split_sha256':sup.digest(data/'split_manifest.json'),
        'checks':'same original split; camera disjoint; balanced; prior diagnostic train images excluded; parsed text conditions equal annotation properties; conjunction equals membership',
        'limits':'Conservative automatic screening, not exhaustive human validation. Filtered tiny development subset; not full benchmark. Hard negatives from positive-list union.'})
    print(json.dumps(sup.read(OUT/'screening_summary.json'),indent=2))
    print('FROZEN: train128/64images; validation24/12images')


if __name__=='__main__': main()
