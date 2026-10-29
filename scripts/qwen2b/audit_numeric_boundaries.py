"""Read raw training annotations; audit numeric conditions on listed positives."""
import ast
import collections
import json
import math
import re
import prepare_ab as prep
import prepare_supervision as sup

OUT=sup.ROOT/'runs/qwen2b_numeric_boundary_audit'


def main():
    counts=collections.Counter(); violations=[]; near=[]
    for path in sorted((sup.ROOT/'data/raw/MonoMulti3D/train').glob('*.json')):
        counts['files']+=1
        for a in sup.read(path)[0]:
            counts['annotations']+=1
            props=dict(x.split(' : ',1) for x in a['public_properties'])
            try:
                parsed=prep.extract(a['public_description']); counts['text_parser_accepted_annotations']+=1
            except (ValueError,AssertionError): parsed={}
            for oi,label in enumerate(a['label_3']):
                o=ast.literal_eval(label) if isinstance(label,str) else label
                counts['positive_occurrences']+=1
                for source,conditions in [('properties',props),('text_parser',parsed)]:
                    for field,idx in [('height',8),('width',9),('length',10),('distance',None)]:
                        if field not in conditions: continue
                        cond=conditions[field]
                        if source=='properties':
                            ns=[float(n) for n in re.findall(r'\d+(?:\.\d+)?',cond)]
                            if len(ns)!=(1 if field=='distance' else 2): raise ValueError(cond)
                        else: ns=[cond] if field=='distance' else cond
                        value=math.sqrt(sum(float(v)**2 for v in o[11:14])) if field=='distance' else float(o[idx])
                        lo,hi=(0,ns[0]) if field=='distance' else ns
                        counts[source+'_'+field+'_comparisons']+=1
                        row={'file':path.name,'ann_id':a['ann_id'],'positive_index':oi,'source':source,
                             'field':field,'value':value,'min':lo,'max':hi,
                             'description':a['public_description'],'public_properties':a['public_properties']}
                        if not lo<=value<=hi:
                            row['overrun_m']=max(lo-value,value-hi)
                            violations.append(row); counts[source+'_'+field+'_outside']+=1
                        elif min(value-lo,hi-value)<=0.05:
                            near.append(row)
    OUT.mkdir(exist_ok=True)
    sup.write(OUT/'summary.json',dict(counts)); sup.write(OUT/'positive_outside_intervals.json',violations)
    sup.write(OUT/'positive_near_inside_intervals.json',near)
    print(json.dumps(counts,indent=2))
    print('VIOLATION_EXAMPLES',json.dumps(violations[:12],ensure_ascii=False,indent=2))


if __name__=='__main__': main()
