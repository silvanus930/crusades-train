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

_prepared_ids = set()
def get_strategy():
    return {"dp_size": 4, "tp_size": 1}

def _prep(mod):
    if id(mod) in _prepared_ids:
        return
    _prepared_ids.add(id(mod))
    if hasattr(mod, "config"):
        mod.config.use_cache = False
    tail_keep = 10
    if hasattr(mod, "model") and hasattr(mod.model, "layers"):
        n_layers = len(mod.model.layers)
        for idx, layer in enumerate(mod.model.layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = 0
            if hasattr(layer, "gradient_checkpointing") and idx >= n_layers - tail_keep:
                layer.gradient_checkpointing = False

def _wp(mod):
    wrap_classes = set()
    if hasattr(mod, "model") and hasattr(mod.model, "layers") and len(mod.model.layers) > 0:
        wrap_classes.add(type(mod.model.layers[0]))
    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=wrap_classes)

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    _prep(model)
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=_wp(model),
        mixed_precision=mixed_precision,
        device_id=device,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )
    logits_fn = torch.compile(lambda x: model(x).logits, mode="reduce-overhead", dynamic=False)
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=True
        )
    inputs_seq, targets_seq = [], []
    n_tokens_batch = 0
    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        inputs_seq.append(batch[:, :-1].contiguous())
        targets_seq.append(batch[:, 1:].contiguous())
        n_tokens_batch = batch.numel()
    torch.cuda.synchronize(device)
    token_total = num_steps * n_tokens_batch
    step, zero_grad, ce = optimizer.step, optimizer.zero_grad, _fce
    try:
        logits = logits_fn(inputs_seq[0])
    except Exception:
        try:
            logits_fn = torch.compile(lambda x: model(x).logits, mode="default", dynamic=False)
            logits = logits_fn(inputs_seq[0])
        except Exception:
            logits_fn = lambda x: model(x).logits
            logits = logits_fn(inputs_seq[0])
    vocab = logits.size(-1)
    loss = ce(logits.reshape(-1, vocab), targets_seq[0].reshape(-1))
    loss.backward()
    step()
    zero_grad(set_to_none=True)
    for step_idx in range(1, num_steps):
        logits = logits_fn(inputs_seq[step_idx])
        loss = ce(logits.reshape(-1, vocab), targets_seq[step_idx].reshape(-1))
        loss.backward()
        step()
        zero_grad(set_to_none=True)
    logits_final, loss_val = logits.detach(), loss.item()
    rank = dist.get_rank() if dist.is_initialized() else 0
    cpu_state = None
    with FSDP.summon_full_params(model, writeback=False):
        unwrapped = model.module if hasattr(model, "module") else model
        if rank == 0:
            state_dict = unwrapped.state_dict()
            pinned = {k: torch.empty_like(v, device="cpu").pin_memory() for k, v in state_dict.items()}
            for k, v in state_dict.items():
                pinned[k].copy_(v, non_blocking=True)
            torch.cuda.synchronize(device)
            cpu_state = pinned
    return InnerStepsResult(
        final_logits=logits_final, total_tokens=token_total, final_loss=loss_val, final_state=cpu_state
    )
