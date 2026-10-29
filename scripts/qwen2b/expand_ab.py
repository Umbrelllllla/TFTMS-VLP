"""Frozen expanded validation, sequential saved-adapter inference, strict scoring."""
import argparse
import collections
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
import traceback
import prepare_supervision as sup
import prepare_ab as prep
import report_ab as scoring
import run_ab as original

ROOT=sup.ROOT
OLD=original.DATA
OUT=ROOT/'runs/qwen2b_supervision_ab_expanded_v1'


def boundary(case):
    """Diagnostic category only; never changes membership labels or predictions."""
    failures=[]
    for field,a in case['structured_target']['attributes'].items():
        if a['satisfied'] is not False: continue
        row={'field':field}
        if field in ('height','width','length'):
            value=case['input']['candidate']['dimensions_m'][field]; c=a['condition']
            row.update(value_m=value,min_m=c['min'],max_m=c['max'],
                       gap_m=max(c['min']-value,value-c['max'],0))
        elif field=='distance':
            row['gap_m']=case['input']['candidate']['distance_m']-a['condition']['value']
        failures.append(row)
    small=bool(failures) and all(x['field'] in ('height','width','length') and x['gap_m']<=0.05 for x in failures)
    return {'only_size_overrun_up_to_5cm':small,'failed_conditions':failures}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'manifest.json').exists(), 'Frozen experiment already exists'
    split=sup.read(ROOT/'runs/20261015_qwen2b_input_v1/split_manifest.json')
    val=set(split['validation'])
    excluded={c['source_file'] for c in sup.read(OLD/'validation.json')}
    training=sup.read(OLD/'train.json')
    spec=importlib.util.spec_from_file_location('feature_prep',ROOT/'runs/20261015_qwen2b_input_v1/prepare.py')
    feature=importlib.util.module_from_spec(spec); spec.loader.exec_module(feature)
    pools={}; reasons=collections.Counter()
    for r in sup.read(ROOT/'runs/20261015_qwen2b_data_audit/training_records.json'):
        if r['file'] not in val or r['file'] in excluded: continue
        for ai,a in enumerate(r['annotations']):
            try:
                assert prep.extract(a['text'])==prep.normalized(a['properties']), 'text_property_disagreement'
                pos=set(a['positive_indices']); neg=set(range(len(r['objects'])))-pos
                if not pos or not neg: reasons['no_positive_or_negative']+=1; continue
                cases=[]
                for oi,o in enumerate(r['objects']):
                    c={'input':{'description':a['text'],'candidate':feature.features(o,r['width'],r['height'])},
                       'target':int(oi in pos),'properties_for_audit_only':a['properties'],
                       'source_file':r['file'],'camera':r['camera'],'annotation_index':ai,'object_index':oi}
                    c['structured_target']=sup.target(c); cases.append(c)
                pools.setdefault(r['file'],[]).append(cases)
            except (AssertionError,ValueError) as e: reasons[str(e)]+=1
    rng=random.Random(20261018)
    files=sorted(pools); rng.shuffle(files); assert len(files)>=120
    selected=[]
    for f in files[:120]:
        cases=rng.choice(pools[f]); pos=[c for c in cases if c['target']]; neg=[c for c in cases if not c['target']]
        rng.shuffle(neg)
        hard=min(neg,key=lambda c:sum(a['satisfied'] is False for a in c['structured_target']['attributes'].values()))
        selected.extend([rng.choice(pos),hard])
    rng.shuffle(selected)
    for i,c in enumerate(selected):
        c['input']['pair_id']=f'expanded-{i:03d}'; c['boundary_audit']=boundary(c)
    assert len(selected)==240 and sum(c['target'] for c in selected)==120
    assert not {c['camera'] for c in selected}&{c['camera'] for c in training}
    assert not {c['source_file'] for c in selected}&excluded
    sup.write(OUT/'validation.json',selected)
    refs={}
    for arm in ('A','B'):
        assert sup.read(OLD/arm/'complete.json')['status']=='PASS'
        refs[arm]={p.name:sup.digest(p) for p in (OLD/arm/'adapter').iterdir() if p.is_file()}
    manifest={'seed':20261018,'pairs':240,'images':120,'positives':120,'eligible_images':len(files),
              'validation_sha256':sup.digest(OUT/'validation.json'),'adapter_files':refs,
              'prompt_source_sha256':sup.digest(ROOT/'scripts/qwen2b/run_ab.py'),
              'source_split_sha256':sup.digest(ROOT/'runs/20261015_qwen2b_input_v1/split_manifest.json'),
              'training_sha256':sup.digest(OLD/'train.json'),
              'old_validation_images_excluded':len(excluded),
              'boundary_definition':'Only failed conditions are height/width/length, each exceeds bound by at most 0.05 m; analysis only, no relabeling.',
              'small_size_overrun_pairs':sum(c['boundary_audit']['only_size_overrun_up_to_5cm'] for c in selected),
              'train_exact_description_overlap_pairs':sum(c['input']['description'] in {x['input']['description'] for x in training} for c in selected),
              'attribute_mentions':dict(collections.Counter(k for c in selected for k,a in c['structured_target']['attributes'].items() if a['relevant'])),
              'screening_rejections':dict(reasons),
              'limits':'Filtered development subset, not full validation or test; one positive and one hard negative per image; same camera-token grouping, not independently verified physical camera identity.'}
    sup.write(OUT/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


def run(arm):
    out=OUT/arm; out.mkdir(exist_ok=False)
    os.environ['HF_HOME']=str(ROOT/'.cache/huggingface'); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer,BitsAndBytesConfig,Qwen3_5ForCausalLM
    from peft import PeftModel,prepare_model_for_kbit_training
    from download import DEST,REVISION,verify
    manifest=sup.read(OUT/'manifest.json')
    assert sup.digest(OUT/'validation.json')==manifest['validation_sha256']
    assert sup.digest(ROOT/'scripts/qwen2b/run_ab.py')==manifest['prompt_source_sha256']
    for name,digest in manifest['adapter_files'][arm].items(): assert sup.digest(OLD/arm/'adapter'/name)==digest
    for entry in sup.read(DEST/'source_manifest.json')['siblings']:
        if entry['rfilename']!='.gitattributes': assert verify(DEST/entry['rfilename'],entry)
    torch.manual_seed(20261017)
    tokenizer=AutoTokenizer.from_pretrained(DEST,local_files_only=True)
    model,info=Qwen3_5ForCausalLM.from_pretrained(DEST,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
    # Match the original post-training evaluation's preparation/dtypes, without updating anything.
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(model,OLD/arm/'adapter',is_trainable=False)
    model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache=True
    assert not any(p.requires_grad for p in model.parameters())
    cap=8 if arm=='A' else 384
    def predict(c):
        messages=original.messages(c,arm)
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(text,add_special_tokens=False,return_tensors='pt').to('cuda')
        assert inputs.input_ids.shape[1]+cap<=1536
        torch.cuda.synchronize(); started=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model.generate(**inputs,do_sample=False,max_new_tokens=cap,pad_token_id=tokenizer.eos_token_id,use_cache=True)
        torch.cuda.synchronize(); gen=output[0,inputs.input_ids.shape[1]:]
        return {'pair_id':c['input']['pair_id'],'raw':tokenizer.decode(gen,skip_special_tokens=True).strip(),
                'seconds':time.perf_counter()-started,'generated_tokens':len(gen),
                'hit_token_cap':len(gen)>=cap and gen[-1].item()!=tokenizer.eos_token_id}
    # A previous case is a reload check, excluded from the new evaluation score.
    anchor=sup.read(OLD/'validation.json')[0]
    reference=json.loads((OLD/arm/'after.predictions.jsonl').read_text(encoding='utf-8').splitlines()[0])
    check=predict(anchor); check['same_raw_as_original']=check['raw']==reference['raw']
    sup.write(out/'reload_check.json',check)
    assert check['same_raw_as_original'], 'Saved-adapter reload differs from original anchor; stop before new evaluation.'
    cases=sup.read(OUT/'validation.json')
    sup.write(out/'inputs.json',[{'pair_id':c['input']['pair_id'],'messages':original.messages(c,arm)} for c in cases])
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for i,c in enumerate(cases,1):
            row=predict(c); f.write(json.dumps(row)+'\n'); f.flush()
            print(arm,i,'/240',round(row['seconds'],1),'s',flush=True)
    for name,digest in manifest['adapter_files'][arm].items(): assert sup.digest(OLD/arm/'adapter'/name)==digest
    sup.write(out/'complete.json',{'status':'PASS','pairs':len(cases),'revision':REVISION,'adapter_unchanged':True,'reload_anchor_exact':True})


def report():
    cases=sup.read(OUT/'validation.json'); manifest=sup.read(OUT/'manifest.json')
    assert sup.digest(OUT/'validation.json')==manifest['validation_sha256']
    train_texts={c['input']['description'] for c in sup.read(OLD/'train.json')}
    results={}; details={}
    for arm in ('A','B'):
        assert sup.read(OUT/arm/'complete.json')['status']=='PASS'
        ps=[json.loads(x) for x in (OUT/arm/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        results[arm],details[arm]=scoring.score(cases,ps,arm)
        errors=[(c,d) for c,d in zip(cases,details[arm]) if not d['correct']]
        results[arm]['small_size_overrun_errors']=sum(c['boundary_audit']['only_size_overrun_up_to_5cm'] for c,d in errors)
        for label, predicate in [('seen_description',lambda c:c['input']['description'] in train_texts),
                                 ('unseen_description',lambda c:c['input']['description'] not in train_texts)]:
            subset=[d for c,d in zip(cases,details[arm]) if predicate(c)]
            results[arm][label]={'correct':sum(d['correct'] for d in subset),'total':len(subset)}
        sup.write(OUT/arm/'errors.json',[dict(d,boundary_audit=c['boundary_audit']) for c,d in errors])
    sup.write(OUT/'results.json',results); sup.write(OUT/'scored_cases.json',details)
    lines=['# 扩大验证结果','','120张新开发验证图片，240对样本，正负各120。既有权重和提示词不变；不含上一轮12张图片。',
           '', '|方式|正确|误选|漏选|格式错误|F1|仅尺寸越界≤5cm的错误|', '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in results.items():
        lines.append(f"|{arm}|{r['correct']}/240|{r['fp']}|{r['fn']}|{r['invalid']}|{r['f1']:.4f}|{r['small_size_overrun_errors']}|")
    for arm,r in results.items():
        seen,unseen=r['seen_description'],r['unseen_description']
        lines+=['',f"{arm}：与训练描述逐字重复的部分{seen['correct']}/{seen['total']}；未逐字重复的部分{unseen['correct']}/{unseen['total']}。两部分均为新验证图片。"]
    z=results['B']['auxiliary']
    lines+=['',f"B真实提及条件提取：{z['mentioned_condition_correct']}/{z['mentioned_slots']}；满足性判断：{z['mentioned_satisfied_correct']}/{z['mentioned_slots']}。只读取完整JSON最终match字段时正确{z['match_field_only_correct']}/240。",
            '', '尺寸越界分析仅描述错误性质，主成绩仍按原标签计分。5厘米不是官方容差；距离、类别、位置等不计入这一类。',
            '', '筛选仍要求文字与属性一致，有正负候选；不是完整800张开发集，更不是官方test。相同图片的两条样本有关联。逐题原始输出和错误边界差值见A/B子目录。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status=ROOT/'docs/status.md'
    message=f"\n扩大验证已完成：120张新图片、240对。A正确{results['A']['correct']}/240，B正确{results['B']['correct']}/240。详见[扩大验证报告](../runs/qwen2b_supervision_ab_expanded_v1/README.md)。\n"
    if message not in status.read_text(encoding='utf-8'):
        with status.open('a',encoding='utf-8') as f: f.write(message)
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


def pipeline():
    sup.write(OUT/'pipeline_status.json',{'status':'RUNNING','stage':'A'})
    for arm in ('A','B'):
        sup.write(OUT/'pipeline_status.json',{'status':'RUNNING','stage':arm})
        with (OUT/(arm+'_console.log')).open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u',str(PathFile),'run',arm],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    report(); sup.write(OUT/'pipeline_status.json',{'status':'COMPLETE'})


PathFile=__file__
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','run','pipeline','report']); p.add_argument('arm',nargs='?',choices=['A','B']); args=p.parse_args()
    try:
        if args.action=='prepare': prepare()
        elif args.action=='run':
            assert args.arm; run(args.arm)
        elif args.action=='pipeline': pipeline()
        else: report()
    except Exception as e:
        if OUT.exists(): sup.write(OUT/('failure_'+args.action+'.json'),{'error':str(e),'traceback':traceback.format_exc()})
        if args.action=='pipeline': sup.write(OUT/'pipeline_status.json',{'status':'FAILED','error':str(e)})
        raise
