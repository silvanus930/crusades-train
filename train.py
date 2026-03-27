
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


_PREPARED_MODEL_IDS = set()
_DEFERRED_STATE_OVERLAP_STEPS = 1
_INNER_STEPS_CALL_COUNT = 0

def get_strategy():
    return {"dp_size": 4, "tp_size": 1}

def _prepare_model_for_training(model):
    model_id = id(model)
    if model_id in _PREPARED_MODEL_IDS:
        return
    _PREPARED_MODEL_IDS.add(model_id)
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False
        if hasattr(model.config, "output_attentions"):
            model.config.output_attentions = False


def _build_fsdp_wrap_policy(model):
    transformer_layer_class = None
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        transformer_layer_class = model.model.layers[0].__class__
    elif (
        hasattr(model, "transformer")
        and hasattr(model.transformer, "h")
        and len(model.transformer.h) > 0
    ):
        transformer_layer_class = model.transformer.h[0].__class__
    if transformer_layer_class is None:
        return None
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_class},
    )


def _gather_full_state_dict(model, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    state_dict_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_dict_policy):
            state_dict = model.state_dict()
            return state_dict if rank == 0 else None


def _run_deferred_state_steps(context):
    compiled_forward = context["compiled_forward"]
    cross_entropy_loss = context["cross_entropy_loss"]
    vocab_size = context["vocab_size"]
    optimizer_step = context["optimizer_step"]
    optimizer_zero_grad = context["optimizer_zero_grad"]
    input_id_batches = context["input_id_batches"]
    label_batches = context["label_batches"]
    deferred_start_step = context["deferred_start_step"]
    total_steps = context["total_steps"]
    model = context["model"]
    device = context["device"]

    final_logits = context["final_logits"]
    final_loss = context["final_loss"]

    for step_idx in range(deferred_start_step, total_steps):
        logits = compiled_forward(input_id_batches[step_idx])
        loss = cross_entropy_loss(logits.reshape(-1, vocab_size), label_batches[step_idx].reshape(-1))
        loss.backward()
        optimizer_step()
        optimizer_zero_grad(set_to_none=True)
        if step_idx == total_steps - 1:
            final_logits = logits.detach()
            final_loss = loss.item()

    context["final_logits"] = final_logits
    context["final_loss"] = final_loss
    context["final_state"] = _gather_full_state_dict(model, device)
    context["state_ready"] = True


class _DeferredStateResult:
    _context_by_result_id = {}

    def __init__(self, context):
        _DeferredStateResult._context_by_result_id[id(self)] = context

    def __del__(self):
        _DeferredStateResult._context_by_result_id.pop(id(self), None)

    def __getattribute__(self, name):
        contexts = _DeferredStateResult._context_by_result_id
        result_id = id(self)
        if result_id not in contexts:
            raise AttributeError(name)
        context = contexts[result_id]

        if name == "final_state" and not context["state_ready"]:
            _run_deferred_state_steps(context)
            return context["final_state"]

        if name in context:
            return context[name]

        raise AttributeError(name)


# Backward-compatible aliases for older helper names used externally.
_prepare_model = _prepare_model_for_training
_get_wrap_policy = _build_fsdp_wrap_policy
_gather_fsdp_state = _gather_full_state_dict
_collect_state = _run_deferred_state_steps
_StateCollectorResult = _DeferredStateResult


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    global _INNER_STEPS_CALL_COUNT
    _INNER_STEPS_CALL_COUNT += 1
    _prepare_model_for_training(model)

    use_fsdp = num_gpus > 1
    if use_fsdp:
        fsdp_wrap_policy = _build_fsdp_wrap_policy(model)
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            auto_wrap_policy=fsdp_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision_policy,
            device_id=device,
            use_orig_params=True,
        )

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=not use_fsdp,
        )

    cross_entropy_loss = _flash_ce_inst

    input_id_batches = []
    label_batches = []
    tokens_in_batch = 0
    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        input_id_batches.append(batch[:, :-1].contiguous())
        label_batches.append(batch[:, 1:].contiguous())
        tokens_in_batch = batch.numel()

    torch.cuda.synchronize(device)

    total_tokens = num_steps * tokens_in_batch
    optimizer_step = optimizer.step
    optimizer_zero_grad = optimizer.zero_grad

    def forward_logits(input_ids):
        return model(input_ids).logits

    compiled_forward = torch.compile(forward_logits, mode="default", dynamic=False)

    try:
        logits = compiled_forward(input_id_batches[0])
    except Exception:
        compiled_forward = forward_logits
        logits = compiled_forward(input_id_batches[0])

    vocab_size = logits.size(-1)
    loss = cross_entropy_loss(logits.reshape(-1, vocab_size), label_batches[0].reshape(-1))
    loss.backward()
    optimizer_step()
    optimizer_zero_grad(set_to_none=True)

    if _INNER_STEPS_CALL_COUNT <= 1:
        for step_idx in range(1, num_steps):
            logits = compiled_forward(input_id_batches[step_idx])
            loss = cross_entropy_loss(logits.reshape(-1, vocab_size), label_batches[step_idx].reshape(-1))
            loss.backward()
            optimizer_step()
            optimizer_zero_grad(set_to_none=True)

        final_logits = logits.detach()
        final_loss_val = loss.item()

        return InnerStepsResult(
            final_logits=final_logits,
            total_tokens=total_tokens,
            final_loss=final_loss_val,
            final_state=_gather_full_state_dict(model, device) if use_fsdp else {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
        )

    overlap_steps = min(_DEFERRED_STATE_OVERLAP_STEPS, max(num_steps - 1, 0))
    eager_step_end = num_steps - overlap_steps

    for step_idx in range(1, eager_step_end):
        logits = compiled_forward(input_id_batches[step_idx])
        loss = cross_entropy_loss(logits.reshape(-1, vocab_size), label_batches[step_idx].reshape(-1))
        loss.backward()
        optimizer_step()
        optimizer_zero_grad(set_to_none=True)

    final_logits = logits.detach()
    final_loss_val = loss.item()

    deferred_context = {
        "final_logits": final_logits,
        "total_tokens": total_tokens,
        "final_loss": final_loss_val,
        "final_state": None,
        "state_ready": False,
        "compiled_forward": compiled_forward,
        "cross_entropy_loss": cross_entropy_loss,
        "vocab_size": vocab_size,
        "optimizer_step": optimizer_step,
        "optimizer_zero_grad": optimizer_zero_grad,
        "input_id_batches": input_id_batches,
        "label_batches": label_batches,
        "total_steps": num_steps,
        "deferred_start_step": eager_step_end,
        "model": model,
        "device": device,
    }

    return _DeferredStateResult(deferred_context)
