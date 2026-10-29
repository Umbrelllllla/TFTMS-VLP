"""Real CUDA/NF4 operation; no model download or training."""
import json
import faulthandler
from pathlib import Path
from windows_runtime import prepare_runtime

prepare_runtime()
faulthandler.dump_traceback_later(60, repeat=True)
print('SYSTEM_CPP_RUNTIME_LOADED', flush=True)
import torch
print('TORCH_IMPORTED', flush=True)
import bitsandbytes as bnb
print('BITSANDBYTES_IMPORTED', flush=True)
from transformers import Qwen3_5ForCausalLM
print('QWEN_CLASS_IMPORTED', flush=True)

assert torch.cuda.is_available(), 'CUDA not available'
x = torch.randn(1, 64, device='cuda', dtype=torch.bfloat16)
m = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16, quant_type='nf4').to('cuda')
y = m(x)
torch.cuda.synchronize()
assert y.shape == (1, 32) and torch.isfinite(y).all()
result = {'status': 'CUDA_NF4_CHECK_PASS', 'torch': torch.__version__,
          'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
          'output_shape': list(y.shape), 'system_runtime_preloaded': True}
out = Path(__file__).resolve().parents[2] / 'runs' / 'qwen2b_setup'
out.mkdir(parents=True, exist_ok=True)
(out / 'runtime_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
faulthandler.cancel_dump_traceback_later()
