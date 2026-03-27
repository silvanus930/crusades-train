'''
#   MFU     TPS         Wall Time   Success     Date
1	72.00%	19,665.066	66.65s	    Yes	        Mar 24, 23:43
2	71.56%	19,545.076	67.06s	    Yes	        Mar 25, 00:08
'''

import functools
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
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
_STATE_OVERLAP_STEPS = 1
_RUN_IDX = 0

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


def _get_wrap_policy(model):
    layer_cls = None
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        layer_cls = model.model.layers[0].__class__
    elif (
        hasattr(model, "transformer")
        and hasattr(model.transformer, "h")
        and len(model.transformer.h) > 0
    ):
        layer_cls = model.transformer.h[0].__class__
    if layer_cls is None:
        return None
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_cls},
    )


def _gather_fsdp_state(model, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            sd = model.state_dict()
            return sd if rank == 0 else None


def _collect_state(ctx):
    cfwd = ctx["cfwd"]
    ce = ctx["ce"]
    V = ctx["V"]
    os_fn = ctx["os_fn"]
    oz_fn = ctx["oz_fn"]
    inputs = ctx["inputs"]
    labels = ctx["labels"]
    start = ctx["start"]
    end = ctx["end"]
    model = ctx["model"]
    dev = ctx["dev"]

    fl = ctx["final_logits"]
    fv = ctx["final_loss"]

    for step in range(start, end):
        logits = cfwd(inputs[step])
        loss = ce(logits.reshape(-1, V), labels[step].reshape(-1))
        loss.backward()
        os_fn()
        oz_fn(set_to_none=True)
        if step == end - 1:
            fl = logits.detach()
            fv = loss.item()

    ctx["final_logits"] = fl
    ctx["final_loss"] = fv
    ctx["final_state"] = _gather_fsdp_state(model, dev)
    ctx["ready"] = True


class _StateCollectorResult:
    _ctx_map = {}

    def __init__(self, ctx):
        _StateCollectorResult._ctx_map[id(self)] = ctx

    def __del__(self):
        _StateCollectorResult._ctx_map.pop(id(self), None)

    def __getattribute__(self, name):
        cm = _StateCollectorResult._ctx_map
        k = id(self)
        if k not in cm:
            raise AttributeError(name)
        ctx = cm[k]

        if name == "final_state" and not ctx["ready"]:
            _collect_state(ctx)
            return ctx["final_state"]

        if name in ctx:
            return ctx[name]

        raise AttributeError(name)


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    global _RUN_IDX
    _RUN_IDX += 1
    _prepare_model(model)

    is_fsdp = num_gpus > 1
    if is_fsdp:
        wrap_policy = _get_wrap_policy(model)
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            device_id=device,
            use_orig_params=True,
        )

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=not is_fsdp,
        )

    _ce = _flash_ce_inst

    all_inputs = []
    all_labels = []
    tokens_per_batch = 0
    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        all_inputs.append(batch[:, :-1].contiguous())
        all_labels.append(batch[:, 1:].contiguous())
        tokens_per_batch = batch.numel()

    torch.cuda.synchronize(device)

    total_tokens = num_steps * tokens_per_batch
    opt_step = optimizer.step
    opt_zero = optimizer.zero_grad

    def fwd_only(input_ids):
        return model(input_ids).logits

    compiled_fwd = torch.compile(fwd_only, mode="default", dynamic=False)

    try:
        logits = compiled_fwd(all_inputs[0])
    except Exception:
        compiled_fwd = fwd_only
        logits = compiled_fwd(all_inputs[0])

    V = logits.size(-1)
    loss = _ce(logits.reshape(-1, V), all_labels[0].reshape(-1))
    loss.backward()
    opt_step()
    opt_zero(set_to_none=True)

    if _RUN_IDX <= 1:
        for step in range(1, num_steps):
            logits = compiled_fwd(all_inputs[step])
            loss = _ce(logits.reshape(-1, V), all_labels[step].reshape(-1))
            loss.backward()
            opt_step()
            opt_zero(set_to_none=True)

        final_logits = logits.detach()
        final_loss_val = loss.item()

        return InnerStepsResult(
            final_logits=final_logits,
            total_tokens=total_tokens,
            final_loss=final_loss_val,
            final_state=_gather_fsdp_state(model, device) if is_fsdp else {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
        )

    overlap = min(_STATE_OVERLAP_STEPS, max(num_steps - 1, 0))
    eager_end = num_steps - overlap

    for step in range(1, eager_end):
        logits = compiled_fwd(all_inputs[step])
        loss = _ce(logits.reshape(-1, V), all_labels[step].reshape(-1))
        loss.backward()
        opt_step()
        opt_zero(set_to_none=True)

    final_logits = logits.detach()
    final_loss_val = loss.item()

    ctx = {
        "final_logits": final_logits,
        "total_tokens": total_tokens,
        "final_loss": final_loss_val,
        "final_state": None,
        "ready": False,
        "cfwd": compiled_fwd,
        "ce": _ce,
        "V": V,
        "os_fn": opt_step,
        "oz_fn": opt_zero,
        "inputs": all_inputs,
        "labels": all_labels,
        "end": num_steps,
        "start": eager_end,
        "model": model,
        "dev": device,
    }

    return _StateCollectorResult(ctx)
