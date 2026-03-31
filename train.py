import functools
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

try:
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception:
    pass
try:
    import torch._inductor.config as _ic
    _ic.coordinate_descent_tuning = True
    _ic.triton.unique_kernel_names = True
    _ic.fx_graph_cache = True
    _ic.triton.cudagraph_trees = True
    _ic.epilogue_fusion = True
    _ic.shape_padding = True
except Exception:
    pass
try:
    import torch._dynamo.config as _dc
    _dc.cache_size_limit = 128
    _dc.suppress_errors = True
    _dc.assume_static_by_default = True
    _dc.automatic_dynamic_shapes = False
    _dc.optimize_ddp = True
except Exception:
    pass

from flash_attn.losses.cross_entropy import CrossEntropyLoss as _FlashCELoss
_fce = _FlashCELoss(ignore_index=-100)

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None

_P = set()
def get_strategy():
    return {"dp_size": 4, "tp_size": 1}

def _prep(m):
    if id(m) in _P: return
    _P.add(id(m))
    if hasattr(m, "config"):
        m.config.use_cache = False
    UN = 10
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        n = len(m.model.layers)
        for i, l in enumerate(m.model.layers):
            if hasattr(l, "self_attn") and hasattr(l.self_attn, "layer_idx"):
                l.self_attn.layer_idx = 0
            if hasattr(l, "gradient_checkpointing") and i >= n - UN:
                l.gradient_checkpointing = False

def _wp(m):
    lc = set()
    if hasattr(m, "model") and hasattr(m.model, "layers") and len(m.model.layers) > 0:
        lc.add(type(m.model.layers[0]))
    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=lc)

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    _prep(model)
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, auto_wrap_policy=_wp(model),
                 mixed_precision=mp, device_id=device, use_orig_params=True,
                 forward_prefetch=True, backward_prefetch=BackwardPrefetch.BACKWARD_PRE)
    cf = torch.compile(lambda x: model(x).logits, mode="reduce-overhead", dynamic=False)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
    ai, al = [], []
    tpb = 0
    for _ in range(num_steps):
        b = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        ai.append(b[:, :-1].contiguous()); al.append(b[:, 1:].contiguous()); tpb = b.numel()
    torch.cuda.synchronize(device)
    tt = num_steps * tpb
    os_fn, oz_fn, ce = optimizer.step, optimizer.zero_grad, _fce
    try:
        lo = cf(ai[0])
    except Exception:
        try:
            cf = torch.compile(lambda x: model(x).logits, mode="default", dynamic=False)
            lo = cf(ai[0])
        except Exception:
            cf = lambda x: model(x).logits
            lo = cf(ai[0])
    V = lo.size(-1)
    loss = ce(lo.reshape(-1, V), al[0].reshape(-1)); loss.backward(); os_fn(); oz_fn(set_to_none=True)
    for s in range(1, num_steps):
        lo = cf(ai[s]); loss = ce(lo.reshape(-1, V), al[s].reshape(-1)); loss.backward(); os_fn(); oz_fn(set_to_none=True)
    fl, fv = lo.detach(), loss.item()
    r = dist.get_rank() if dist.is_initialized() else 0
    fs = None
    with FSDP.summon_full_params(model, writeback=False):
        raw = model.module if hasattr(model, "module") else model
        if r == 0:
            sd = raw.state_dict()
            p = {k: torch.empty_like(v, device="cpu").pin_memory() for k, v in sd.items()}
            for k, v in sd.items(): p[k].copy_(v, non_blocking=True)
            torch.cuda.synchronize(device); fs = p
    return InnerStepsResult(final_logits=fl, total_tokens=tt, final_loss=fv, final_state=fs)
