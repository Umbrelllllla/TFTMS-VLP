"""Three real QLoRA updates: memory/gradient/save verification, not efficacy training."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'runs/qwen2b_intermediate_supervision_v1'
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def write(p, x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def main(out):
    from windows_runtime import prepare_runtime
    prepare_runtime()
    print('RUNTIME_READY',flush=True)
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict
    from safetensors.torch import load_file
    from download import DEST, REVISION, verify
    assert torch.cuda.is_available()
    torch.manual_seed(20261017)
    for entry in json.loads((DEST/'source_manifest.json').read_text(encoding='utf-8'))['siblings']:
        if entry['rfilename'] != '.gitattributes': assert verify(DEST/entry['rfilename'], entry)
    tokenizer=AutoTokenizer.from_pretrained(DEST,local_files_only=True)
    rows=[json.loads(x) for x in (DATA/'sft_preview.jsonl').read_text(encoding='utf-8').splitlines()]
    prepared=[]
    for row in rows:
        prompt=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=tokenizer.encode(prompt,add_special_tokens=False)
        answer=row['messages'][-1]['content']
        suffix=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
        assert tokenizer.decode(suffix[:-1])==answer
        assert len(prefix+suffix)<=1024
        prepared.append({'pair_id':row['pair_id'],'prefix':prefix,'suffix':suffix,
                         'match':json.loads(answer)['match'],'prompt':prompt,'answer':answer})
    # Include longest sample plus opposite class; no diagnostic-set scoring.
    prepared.sort(key=lambda x:len(x['prefix'])+len(x['suffix']),reverse=True)
    chosen=[prepared[0],next(x for x in prepared if x['match']!=prepared[0]['match'])]
    chosen.append(next(x for x in prepared if x['pair_id'] not in {v['pair_id'] for v in chosen}))
    write(out/'training_samples.json',[{'pair_id':x['pair_id'],'prompt':x['prompt'],'answer':x['answer'],
          'masked_prompt_tokens':len(x['prefix']),'supervised_answer_and_eos_tokens':len(x['suffix']),
          'loss_prediction_positions':[len(x['prefix'])-1,len(x['prefix'])+len(x['suffix'])-2]} for x in chosen])
    config={'revision':REVISION,'gpu':torch.cuda.get_device_name(),'seed':20261017,
            'steps':3,'batch_size':1,'learning_rate':0.0001,'rank':4,'alpha':8,'dropout':0.0,
            'target_modules':'all-linear','quantization':'NF4 double quantization; bfloat16 compute',
            'gradient_checkpointing':'non-reentrant','loss':'mean next-token cross entropy on assistant JSON plus EOS only',
            'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','bitsandbytes','peft']},
            'source_sha256':hashlib.sha256((DATA/'sft_preview.jsonl').read_bytes()).hexdigest(),
            'purpose':'Training feasibility only; no validation score or learning claim.'}
    write(out/'config.json',config)
    print('LOADING_MODEL',flush=True)
    model,info=Qwen3_5ForCausalLM.from_pretrained(DEST,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
    write(out/'loading_info.json',{k:sorted(v) if isinstance(v,set) else v for k,v in info.items()})
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,
                     target_modules='all-linear',bias='none',task_type='CAUSAL_LM'))
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in n for n in trainable)
    frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
    write(out/'trainable_parameters.json',{'count':sum(p.numel() for p in trainable.values()),
         'tensors':{n:list(p.shape) for n,p in trainable.items()}})
    optimizer=torch.optim.AdamW(list(trainable.values()),lr=0.0001)
    model.train()
    records=[]
    print('TRAINING_START',flush=True)
    with (out/'steps.jsonl').open('w',encoding='utf-8') as f:
        for step,item in enumerate(chosen,1):
            ids=torch.tensor([item['prefix']+item['suffix']],device='cuda')
            P=len(item['prefix']); N=ids.shape[1]
            # Position P-1 predicts the first answer token; N-2 predicts EOS.
            # Select only answer-predicting logits to avoid allocating prompt-vocab logits.
            positions=torch.arange(P-1,N-1,device='cuda')
            labels=ids[:,P:]
            assert positions.numel()==labels.numel() and positions[0].item()==P-1
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                             logits_to_keep=positions).logits
                loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1))
            assert torch.isfinite(loss)
            print(f'STEP {step} forward loss={loss.item():.6f}',flush=True)
            loss.backward()
            grads=[p.grad for p in trainable.values() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all().item() for g in grads)
            norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0,error_if_nonfinite=True)
            assert norm.item()>0
            optimizer.step(); torch.cuda.synchronize()
            assert all(p.grad is None and p._version==frozen[n] for n,p in model.named_parameters() if not p.requires_grad)
            record={'step':step,'pair_id':item['pair_id'],'loss':loss.item(),'gradient_norm':norm.item(),
                    'seconds':time.perf_counter()-start,'total_tokens':N,'supervised_tokens':labels.numel(),
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
            records.append(record); f.write(json.dumps(record)+'\n'); f.flush()
            print(json.dumps(record),flush=True)
            del logits,loss,ids,labels,positions
    changed=[n for n,p in trainable.items() if not torch.equal(initial[n],p.detach().cpu())]
    assert changed
    model.save_pretrained(out/'adapter',safe_serialization=True)
    saved=load_file(out/'adapter/adapter_model.safetensors')
    live=get_peft_model_state_dict(model)
    assert set(saved)==set(live)
    assert all(torch.equal(saved[k],live[k].detach().cpu()) for k in saved)
    write(out/'summary.json',{'status':'PASS','steps':len(records),'changed_adapter_tensors':len(changed),
        'trainable_parameters':sum(p.numel() for p in trainable.values()),
        'frozen_parameter_versions_unchanged':True,'saved_adapter_tensor_roundtrip':True,
        'peak_allocated_mib':max(r['peak_allocated_mib'] for r in records),
        'peak_reserved_mib':max(r['peak_reserved_mib'] for r in records),
        'seconds_per_step':[r['seconds'] for r in records],
        'note':'Only 3 updates on 3 training samples. Not a trained final model or quality comparison.'})
    print('TRAIN_SMOKE_PASS',out,flush=True)


if __name__=='__main__':
    output=ROOT/'runs'/('qwen2b_train_smoke_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',output,flush=True)
    try: main(output)
    except Exception as error:
        write(output/'failure.json',{'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
        raise
