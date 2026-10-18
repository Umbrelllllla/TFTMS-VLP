"""Resumable A inference on official test inputs; validate result/*.txt and ZIP."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import time
import traceback
import zipfile
import prepare_supervision as sup
import run_ab as original

ROOT=sup.ROOT
OUT=ROOT/'runs/qwen2b_test_A'
RAW=ROOT/'data/raw/MonoMulti3D/test'
ADAPTER=original.DATA/'A/adapter'
FEATURE_SCRIPT=ROOT/'runs/20261015_qwen2b_input_v1/prepare.py'


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)


def image_size(path):
    with path.open('rb') as f:
        assert f.read(2)==b'\xff\xd8','Expected JPEG'
        while True:
            b=f.read(1)
            if not b: raise ValueError('JPEG dimensions missing')
            if b!=b'\xff': continue
            while b==b'\xff': b=f.read(1)
            marker=b[0]
            if marker in (0xd8,0xd9): continue
            n=struct.unpack('>H',f.read(2))[0]
            if marker in (0xc0,0xc1,0xc2,0xc3):
                h,w=struct.unpack('>HH',f.read(5)[1:]); return w,h
            assert n>=2
            f.seek(n-2,1)


def feature_module():
    spec=importlib.util.spec_from_file_location('fixed_features',FEATURE_SCRIPT)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def objects(data,width,height):
    module=feature_module(); result=[]
    for line in data['test_data']:
        parts=line.split(); assert len(parts)==16
        o=[parts[0]]+[float(x) for x in parts[1:15]]+[parts[15]]
        assert o[1]==int(o[1]) and o[2]==int(o[2])
        o[1],o[2]=int(o[1]),int(o[2])
        result.append(module.features(o,width,height))
    return result


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'manifest.json').exists(),'Already prepared; use run to resume'
    entries=[]
    for path in sorted(RAW.glob('*.json')):
        d=sup.read(path)
        assert len(d['public_description'])==3 and all(isinstance(s,str) and s for s in d['public_description'])
        image=ROOT/'data/raw/img/test'/(path.stem+'.jpg')
        w,h=image_size(image); candidates=objects(d,w,h)
        assert candidates
        entries.append({'file':path.name,'sha256':sup.digest(path),'width':w,'height':h,'rows':len(candidates)})
    assert len(entries)==300
    assert sup.read(original.DATA/'A/complete.json')['status']=='PASS'
    assert sup.read(ROOT/'runs/qwen2b_supervision_ab_expanded_v1/A/complete.json')['status']=='PASS'
    page=ROOT/'runs/qwen2b_test_B/task_page.html'
    assert page.exists()
    (OUT/'task_page.html').write_bytes(page.read_bytes())
    manifest={'model':'Qwen3.5-2B + trained A adapter','files':entries,'file_count':len(entries),
              'candidates':sum(e['rows'] for e in entries),'pairs':sum(e['rows']*3 for e in entries),
              'adapter_sha256':{p.name:sup.digest(p) for p in ADAPTER.iterdir() if p.is_file()},
              'feature_script_sha256':sup.digest(FEATURE_SCRIPT),
              'prompt_script_sha256':sup.digest(ROOT/'scripts/qwen2b/run_ab.py'),
              'max_new_tokens':8,'do_sample':False,'enable_thinking':False,
              'format':'ZIP contains result/<input-stem>.txt; N rows in test_data order; 3 space-separated bits in public_description order',
              'source_url':'http://vlp.chd.edu.cn/tasks/2','task_page_sha256':sup.digest(page),
              'input_policy':'Only description and candidate features; no training answers or label/property lookup',
              'invalid_output_policy':'Preserve raw response, do not fabricate 0; withhold ZIP if any unresolved output'}
    sup.write(OUT/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('file_count','candidates','pairs','format')},ensure_ascii=False),flush=True)


def verify_sources(manifest):
    assert {p.name for p in RAW.glob('*.json')}=={e['file'] for e in manifest['files']}
    assert sup.digest(FEATURE_SCRIPT)==manifest['feature_script_sha256']
    assert sup.digest(ROOT/'scripts/qwen2b/run_ab.py')==manifest['prompt_script_sha256']
    for name,digest in manifest['adapter_sha256'].items(): assert sup.digest(ADAPTER/name)==digest
    for e in manifest['files']: assert sup.digest(RAW/e['file'])==e['sha256']


def matrix_text(records,n):
    assert len(records)==n*3
    for k,r in enumerate(records):
        assert r['row']==k//3 and r['column']==k%3
        assert r['raw'] in ('0','1') and type(r['prediction']) is int and r['prediction']==int(r['raw'])
    return ''.join(' '.join(str(records[i*3+j]['prediction']) for j in range(3))+'\n' for i in range(n))


def pack():
    manifest=sup.read(OUT/'manifest.json'); verify_sources(manifest)
    expected={Path(e['file']).stem+'.txt' for e in manifest['files']}
    assert {p.name for p in (OUT/'result').glob('*')}==expected
    contents={}
    for e in manifest['files']:
        stem=Path(e['file']).stem; records=sup.read(OUT/'responses'/(stem+'.json'))
        text=matrix_text(records,e['rows']); file=OUT/'result'/(stem+'.txt')
        assert file.read_text(encoding='ascii')==text
        contents['result/'+file.name]=text.encode('ascii')
    tmp=OUT/'submission_A.zip.tmp'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in contents.items(): z.writestr(name,content)
    with zipfile.ZipFile(tmp) as z:
        assert set(z.namelist())==set(contents) and len(z.namelist())==300 and z.testzip() is None
        assert all(z.read(n)==v for n,v in contents.items())
    tmp.replace(OUT/'submission_A.zip')
    atomic(OUT/'complete.json',{'status':'PASS','files':300,'pairs':manifest['pairs'],
           'zip_sha256':sup.digest(OUT/'submission_A.zip'),'source_and_adapter_unchanged':True,'all_answers_from_model':True})
    atomic(OUT/'status.json',{'status':'COMPLETE','files':300,'pairs':manifest['pairs'],'zip':'submission_A.zip'})
    status=ROOT/'docs/status.md'
    message='\nA模型test推理与打包已完成，全部300个TXT及ZIP格式校验通过。结果：runs/qwen2b_test_A/submission_A.zip；尚未上传网站，不包含官方得分。\n'
    if message not in status.read_text(encoding='utf-8'):
        with status.open('a',encoding='utf-8') as f: f.write(message)
    print('TEST_A_COMPLETE',flush=True)


def run():
    import msvcrt
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'run.lock').open('a+b'); lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    # The OS releases this lock when the process exits, including a crash.
    manifest=sup.read(OUT/'manifest.json'); verify_sources(manifest)
    if (OUT/'complete.json').exists(): pack(); return
    for folder in ('responses','result'): (OUT/folder).mkdir(exist_ok=True)
    atomic(OUT/'status.json',{'status':'LOADING','total_pairs':manifest['pairs']})
    os.environ['HF_HOME']=str(ROOT/'.cache/huggingface'); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    from transformers import AutoTokenizer,BitsAndBytesConfig,Qwen3_5ForCausalLM
    from peft import PeftModel,prepare_model_for_kbit_training
    from download import DEST,verify
    for entry in sup.read(DEST/'source_manifest.json')['siblings']:
        if entry['rfilename']!='.gitattributes': assert verify(DEST/entry['rfilename'],entry)
    torch.manual_seed(20261017)
    tokenizer=AutoTokenizer.from_pretrained(DEST,local_files_only=True)
    model,info=Qwen3_5ForCausalLM.from_pretrained(DEST,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False)
    model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache=True
    assert not any(p.requires_grad for p in model.parameters())
    def predict(c):
        messages=original.messages(c,'A')
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(text,add_special_tokens=False,return_tensors='pt').to('cuda')
        assert inputs.input_ids.shape[1]+8<=1536
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model.generate(**inputs,do_sample=False,max_new_tokens=8,pad_token_id=tokenizer.eos_token_id,use_cache=True)
        torch.cuda.synchronize(); gen=output[0,inputs.input_ids.shape[1]:]
        raw=tokenizer.decode(gen,skip_special_tokens=True).strip()
        return {'raw':raw,'prediction':int(raw) if raw in ('0','1') else None,
                'seconds':time.perf_counter()-start,'generated_tokens':len(gen)}
    anchor=sup.read(original.DATA/'validation.json')[0]
    ref=json.loads((original.DATA/'A/after.predictions.jsonl').read_text(encoding='utf-8').splitlines()[0])
    check=predict(anchor); check['same_raw_as_original']=check['raw']==ref['raw']
    atomic(OUT/'reload_check.json',check); assert check['same_raw_as_original']
    completed=0; invalid=0; seconds=0.; files_done=0
    for e in manifest['files']:
        stem=Path(e['file']).stem; path=OUT/'responses'/(stem+'.json')
        rows=sup.read(path) if path.exists() else []
        assert len(rows)<=e['rows']*3
        for k,r in enumerate(rows):
            assert r['row']==k//3 and r['column']==k%3
            assert r['prediction']==(int(r['raw']) if r['raw'] in ('0','1') else None)
        d=sup.read(RAW/e['file']); candidates=objects(d,e['width'],e['height'])
        completed+=len(rows); invalid+=sum(r['prediction'] is None for r in rows); seconds+=sum(r['seconds'] for r in rows)
        for k in range(len(rows),e['rows']*3):
            i,j=divmod(k,3)
            row=predict({'input':{'description':d['public_description'][j],'candidate':candidates[i]}})
            row.update(row=i,column=j); rows.append(row); atomic(path,rows)
            completed+=1; invalid+=row['prediction'] is None; seconds+=row['seconds']
            atomic(OUT/'status.json',{'status':'RUNNING','completed_pairs':completed,'total_pairs':manifest['pairs'],
                    'completed_files':files_done,'invalid_outputs':invalid,'current_file':e['file'],
                    'mean_seconds_per_pair':seconds/completed,'remaining_hours_estimate':(manifest['pairs']-completed)*seconds/completed/3600})
            if completed%25==0: print(completed,'/',manifest['pairs'],'invalid',invalid,flush=True)
        if all(r['prediction'] is not None for r in rows):
            text=matrix_text(rows,e['rows']); dest=OUT/'result'/(stem+'.txt'); tmp=dest.with_suffix('.txt.tmp')
            tmp.write_text(text,encoding='ascii'); tmp.replace(dest); files_done+=1
    if invalid:
        atomic(OUT/'status.json',{'status':'NEEDS_REVIEW','invalid_outputs':invalid,'completed_pairs':completed,'completed_files':files_done})
        print('Unresolved model outputs; no submission ZIP created',flush=True)
    else: pack()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run','pack']); args=parser.parse_args()
    try: {'prepare':prepare,'run':run,'pack':pack}[args.action]()
    except Exception as exc:
        if OUT.exists():
            atomic(OUT/'failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
            if args.action=='run':
                atomic(OUT/'status.json',{'status':'FAILED','error':str(exc),'resume':'Re-run the same run command; completed responses are preserved'})
        raise
