import functools
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
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
    import torch._inductor.config as _ind_cfg
    _ind_cfg.coordinate_descent_tuning = True
    _ind_cfg.triton.unique_kernel_names = True
    _ind_cfg.fx_graph_cache = True
    _ind_cfg.triton.cudagraph_trees = True
    _ind_cfg.epilogue_fusion = True
    _ind_cfg.shape_padding = True
except Exception:
    pass

try:
    import torch._dynamo.config as _dyn_cfg
    _dyn_cfg.cache_size_limit = 128
    _dyn_cfg.suppress_errors = True
    _dyn_cfg.assume_static_by_default = True
    _dyn_cfg.automatic_dynamic_shapes = False
    _dyn_cfg.optimize_ddp = True
except Exception:
    pass

from flash_attn.losses.cross_entropy import CrossEntropyLoss as _FlashCELoss
_flash_ce_inst = _FlashCELoss(ignore_index=-100)


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


_PREPARED = set()


def get_strategy():
    return {"dp_size": 4, "tp_size": 1}

def _prepare_model(model):
    mid = id(model)
    if mid in _PREPARED:
        return
    _PREPARED.add(mid)

    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False
        if hasattr(model.config, "output_attentions"):
            model.config.output_attentions = False

    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                # Same trick as the top files to avoid some graph instability.
                layer.self_attn.layer_idx = 0
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = False


def _get_wrap_policy(model):
    layer_cls = set()
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        layer_cls.add(type(model.model.layers[0]))
    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls)


def _maybe_compile_fwd(model):
    def fwd_fn(input_ids):
        return model(input_ids).logits

    try:
        return torch.compile(fwd_fn, mode="default", dynamic=False)
    except Exception:
        return fwd_fn


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    _prepare_model(model)

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        auto_wrap_policy=_get_wrap_policy(model),
        mixed_precision=bf16_policy,
        device_id=device,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )

    compiled_fwd = _maybe_compile_fwd(model)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        )

    all_inputs = []
    all_labels = []
    total_tokens = 0

    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        inp = batch[:, :-1].contiguous()
        lbl = batch[:, 1:].contiguous()
        all_inputs.append(inp)
        all_labels.append(lbl)
        total_tokens += batch.numel()

    opt_step = optimizer.step
    opt_zero = optimizer.zero_grad
    ce = _flash_ce_inst

    logits = None
    loss = None

    for step in range(num_steps):
        logits = compiled_fwd(all_inputs[step])
        loss = ce(logits.reshape(-1, logits.size(-1)), all_labels[step].reshape(-1))
        loss.backward()
        opt_step()
        opt_zero(set_to_none=True)

    final_logits = logits.detach()
    final_loss = float(loss.item())

    rank = dist.get_rank() if dist.is_initialized() else 0
    full_state = None
    with FSDP.summon_full_params(model, writeback=False):
        raw = model.module if hasattr(model, "module") else model
        if rank == 0:
            sd = raw.state_dict()
            full_state = {
                k: v.detach().to(device="cpu", copy=True)
                for k, v in sd.items()
            }

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )
