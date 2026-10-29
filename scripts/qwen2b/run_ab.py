"""Frozen small A/B: baseline evaluation, one epoch, post-training evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
import prepare_supervision as sup

ROOT=sup.ROOT
DATA=ROOT/'runs/qwen2b_supervision_ab_v1'
COMMON='''Match the English description against the candidate object.
Read category from type, color from color, distance from distance_m,
length/width/height from dimensions_m, position from image_region,
facing from facing_convention, and occlusion from occlusion.
Generic vehicles do not restrict a specific type. Use the provided geometric
and occlusion meanings, without recomputing them. Distance within D means <= D;
size ranges include both endpoints. All stated conditions must hold.
Do not infer restrictions on unmentioned attributes.
'''
FORMAT_A='Return exactly 1 for a match or 0 for a non-match, with no explanation.'
FORMAT_B='''Return only JSON with attributes and match. attributes contains category,
color, distance, length, width, height, position, facing, occlusion.
Each attribute has relevant (boolean), condition (object or null), and
satisfied (boolean or null). Unmentioned attributes have relevant=false,
condition=null, satisfied=null. Category/color/position/facing/occlusion use
condition {"op":"eq","value":...}; distance limits use
{"op":"le","value":...,"unit":"m"}; size ranges use
{"op":"between","min":...,"max":...,"inclusive":true,"unit":"m"}.
match is a boolean indicating whether all relevant conditions are satisfied.'''


def messages(c,arm):
    return [{'role':'system','content':COMMON+(FORMAT_A if arm=='A' else FORMAT_B)},
            {'role':'user','content':json.dumps({k:c['input'][k] for k in ('description','candidate')},ensure_ascii=False)}]


def main(arm,out):
    os.environ['HF_HOME']=str(ROOT/'.cache/huggingface'); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    from windows_runtime import prepare_runtime
    prepare_runtime()
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,BitsAndBytesConfig,Qwen3_5ForCausalLM
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,get_peft_model_state_dict
    from safetensors.torch import load_file
    from download import DEST,REVISION,verify
    assert torch.cuda.is_available()
    torch.manual_seed(20261017)
    manifest=sup.read(DATA/'manifest.json')
    for sn,digest in manifest['data_sha256'].items(): assert sup.digest(DATA/(sn+'.json'))==digest
    for entry in sup.read(DEST/'source_manifest.json')['siblings']:
        if entry['rfilename']!='.gitattributes': assert verify(DEST/entry['rfilename'],entry)
    train=sup.read(DATA/'train.json'); validation=sup.read(DATA/'validation.json')
    tokenizer=AutoTokenizer.from_pretrained(DEST,local_files_only=True)
    prepared=[]; message_log=[]
    for c in train:
        msg=messages(c,arm)
        prefix=tokenizer.encode(tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        answer=str(c['target']) if arm=='A' else json.dumps(c['structured_target'],ensure_ascii=False,separators=(',',':'))
        suffix=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
        assert len(prefix+suffix)<=1024 and tokenizer.decode(suffix[:-1])==answer
        prepared.append((c['input']['pair_id'],prefix,suffix))
        message_log.append({'pair_id':c['input']['pair_id'],'messages':msg,'training_answer':answer,'prompt_tokens':len(prefix),'supervised_tokens':len(suffix)})
    sup.write(out/'training_inputs.json',message_log)
    config={'arm':arm,'revision':REVISION,'seed':20261017,'batch_size':1,'rank':4,'alpha':8,'dropout':0.,
            'target_modules':'all-linear','lr':0.0001,'steps':128,'epochs':1,'data_manifest_sha256':sup.digest(DATA/'manifest.json'),
            'quantization':'nf4 double quantization, bfloat16 compute','checkpointing':'non-reentrant',
            'max_new_tokens':8 if arm=='A' else 384,'do_sample':False,'enable_thinking':False,
            'training_supervised_tokens':sum(len(x[2]) for x in prepared),
            'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','accelerate')}}
    sup.write(out/'config.json',config)
    print('LOADING',arm,flush=True)
    model,info=Qwen3_5ForCausalLM.from_pretrained(DEST,local_files_only=True,dtype=torch.bfloat16,
       device_map={'':0},attn_implementation='sdpa',output_loading_info=True,
       quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,target_modules='all-linear',bias='none',task_type='CAUSAL_LM'))
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert all('lora_' in n for n in trainable)
    frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
    h=hashlib.sha256()
    for n,p in initial.items(): h.update(n.encode()); h.update(p.numpy().tobytes())
    sup.write(out/'initialization.json',{'adapter_sha256':h.hexdigest(),'trainable_parameters':sum(p.numel() for p in trainable.values())})

    def evaluate(stage):
        model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache=True
        with (out/(stage+'.predictions.jsonl')).open('w',encoding='utf-8') as f:
            for i,c in enumerate(validation):
                msg=messages(c,arm)
                text=tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=tokenizer(text,add_special_tokens=False,return_tensors='pt').to('cuda')
                assert inputs.input_ids.shape[1]+config['max_new_tokens']<=1536
                torch.cuda.synchronize(); start=time.perf_counter()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model.generate(**inputs,do_sample=False,max_new_tokens=config['max_new_tokens'],pad_token_id=tokenizer.eos_token_id,use_cache=True)
                torch.cuda.synchronize()
                generated=output[0,inputs.input_ids.shape[1]:]
                raw=tokenizer.decode(generated,skip_special_tokens=True).strip()
                row={'pair_id':c['input']['pair_id'],'raw':raw,'seconds':time.perf_counter()-start,
                     'generated_tokens':len(generated),'hit_token_cap':len(generated)>=config['max_new_tokens'] and generated[-1].item()!=tokenizer.eos_token_id}
                f.write(json.dumps(row)+'\n'); f.flush()
                print(arm,stage,i+1,'/24',f'{row["seconds"]:.1f}s',flush=True)
        sup.write(out/(stage+'.inputs.json'),[{'pair_id':c['input']['pair_id'],'messages':messages(c,arm)} for c in validation])

    evaluate('before')
    model.train(); model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW(list(trainable.values()),lr=0.0001)
    steps=[]
    with (out/'steps.jsonl').open('w',encoding='utf-8') as f:
        for i,(pid,prefix,suffix) in enumerate(prepared,1):
            ids=torch.tensor([prefix+suffix],device='cuda'); P=len(prefix); N=ids.shape[1]
            positions=torch.arange(P-1,N-1,device='cuda'); labels=ids[:,P:]
            optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=positions).logits
                loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.,error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            assert all(p.grad is None and p._version==frozen[n] for n,p in model.named_parameters() if not p.requires_grad)
            record={'step':i,'pair_id':pid,'loss':loss.item(),'gradient_norm':norm.item(),'seconds':time.perf_counter()-start,
                    'supervised_tokens':labels.numel(),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
            steps.append(record); f.write(json.dumps(record)+'\n'); f.flush()
            if i%8==0: print(arm,'train',i,'/128',record['loss'],flush=True)
            del ids,positions,labels,logits,loss
    changed=sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in trainable.items())
    assert changed>0
    model.save_pretrained(out/'adapter',safe_serialization=True)
    saved=load_file(out/'adapter/adapter_model.safetensors'); live=get_peft_model_state_dict(model)
    assert set(saved)==set(live) and all(torch.equal(saved[k],live[k].detach().cpu()) for k in saved)
    del optimizer,saved,live
    evaluate('after')
    sup.write(out/'complete.json',{'status':'PASS','steps':len(steps),'changed_tensors':changed,
        'training_seconds':sum(s['seconds'] for s in steps),'peak_allocated_mib':max(s['peak_allocated_mib'] for s in steps),
        'source_frozen':True,'adapter_save_verified':True})
    print('ARM_COMPLETE',arm,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('arm',choices=['A','B']); args=parser.parse_args()
    out=DATA/args.arm
    out.mkdir(parents=True,exist_ok=False)
    try: main(args.arm,out)
    except Exception as e:
        sup.write(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc()}); raise
