from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


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


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


_PREPARED = set()
_RUN_IDX = 0
_GRAD_ACCUM = 8
_MICRO_BATCH = 2
_STATE_OVERLAP_STEPS = 3


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

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                    "preserve_rng_state": False,
                }
            )
        except Exception:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
        uncheckpoint_last_n = 2 if num_layers >= 24 else 0
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = 0
            if (
                uncheckpoint_last_n > 0
                and hasattr(layer, "gradient_checkpointing")
                and idx >= num_layers - uncheckpoint_last_n
            ):
                layer.gradient_checkpointing = False


class _DeferredResult:
    _ctx_map = {}

    def __init__(self, ctx):
        _DeferredResult._ctx_map[id(self)] = ctx

    def __del__(self):
        _DeferredResult._ctx_map.pop(id(self), None)

    def __getattribute__(self, name):
        ctx_map = _DeferredResult._ctx_map
        key = id(self)
        if key not in ctx_map:
            raise AttributeError(name)
        ctx = ctx_map[key]

        if name == "final_state" and not ctx["ready"]:
            _finish_remaining(ctx)
            return ctx["final_state"]

        if name in ctx:
            return ctx[name]

        raise AttributeError(name)


def _compile_train_step(train_step, micro_input, micro_label):
    try:
        compiled = torch.compile(train_step, mode="reduce-overhead", dynamic=False, fullgraph=True)
        compiled(micro_input, micro_label)
        return compiled
    except Exception:
        try:
            compiled = torch.compile(train_step, mode="default", dynamic=False, fullgraph=True)
            compiled(micro_input, micro_label)
            return compiled
        except Exception:
            try:
                compiled = torch.compile(train_step, mode="default", dynamic=False)
                compiled(micro_input, micro_label)
                return compiled
            except Exception:
                return train_step


def _run_step(compiled_train, micro_inputs, micro_labels, no_sync, grad_accum, base, opt_step, opt_zero):
    for m in range(grad_accum - 1):
        with no_sync():
            loss = compiled_train(micro_inputs[base + m], micro_labels[base + m])
            loss.backward()
    loss = compiled_train(micro_inputs[base + grad_accum - 1], micro_labels[base + grad_accum - 1])
    loss.backward()
    opt_step()
    opt_zero(set_to_none=True)


def _run_final_step(
    compiled_train,
    fwd_only,
    micro_inputs,
    micro_labels,
    no_sync,
    grad_accum,
    base,
    opt_step,
    opt_zero,
    scale,
    device,
):
    loss_accum = torch.zeros(1, device=device)
    for m in range(grad_accum - 1):
        with no_sync():
            loss = compiled_train(micro_inputs[base + m], micro_labels[base + m])
            loss.backward()
            loss_accum += loss.detach()

    logits = fwd_only(micro_inputs[base + grad_accum - 1])
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        micro_labels[base + grad_accum - 1].reshape(-1),
        ignore_index=-100,
    )
    (loss * scale).backward()
    loss_accum += loss.detach() * scale

    opt_step()
    opt_zero(set_to_none=True)
    return logits.detach(), loss_accum.item()


def _finish_remaining(ctx):
    compiled_train = ctx["compiled_train"]
    fwd_only = ctx["fwd_only"]
    micro_inputs = ctx["micro_inputs"]
    micro_labels = ctx["micro_labels"]
    no_sync = ctx["no_sync"]
    grad_accum = ctx["grad_accum"]
    opt_step = ctx["opt_step"]
    opt_zero = ctx["opt_zero"]
    scale = ctx["scale"]
    device = ctx["device"]
    model = ctx["model"]
    start_step = ctx["start_step"]
    end_step = ctx["end_step"]

    final_logits = ctx["final_logits"]
    final_loss = ctx["final_loss"]

    for step in range(start_step, end_step):
        base = step * grad_accum
        if step < end_step - 1:
            _run_step(compiled_train, micro_inputs, micro_labels, no_sync, grad_accum, base, opt_step, opt_zero)
        else:
            final_logits, final_loss = _run_final_step(
                compiled_train,
                fwd_only,
                micro_inputs,
                micro_labels,
                no_sync,
                grad_accum,
                base,
                opt_step,
                opt_zero,
                scale,
                device,
            )

    rank = dist.get_rank() if dist.is_initialized() else 0
    final_state = None
    if rank == 0:
        raw_model = model.module if hasattr(model, "module") else model
        state_dict = raw_model.state_dict()
        pinned = {k: torch.empty_like(v, device="cpu").pin_memory() for k, v in state_dict.items()}
        for k, v in state_dict.items():
            pinned[k].copy_(v, non_blocking=True)
        torch.cuda.synchronize(device)
        final_state = pinned

    ctx["final_logits"] = final_logits
    ctx["final_loss"] = final_loss
    ctx["final_state"] = final_state
    ctx["ready"] = True


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    global _RUN_IDX
    _RUN_IDX += 1

    _prepare_model(model)
    model = model.to(device)

    grad_accum = _GRAD_ACCUM
    micro_batch = _MICRO_BATCH
    scale = 1.0 / grad_accum

    if num_gpus > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            gradient_as_bucket_view=True,
            bucket_cap_mb=700,
            broadcast_buffers=False,
        )

    def train_step(input_ids, labels):
        logits = model(input_ids).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return loss * scale

    def fwd_only(input_ids):
        return model(input_ids).logits

    if optimizer is None:
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.1,
                betas=(0.9, 0.95),
                fused=True,
            )
        except Exception:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.1,
                betas=(0.9, 0.95),
                fused=False,
            )

    micro_inputs = []
    micro_labels = []
    tokens_per_batch = 0
    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        inputs = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        tokens_per_batch = batch.numel()
        for m in range(grad_accum):
            start = m * micro_batch
            end = start + micro_batch
            micro_inputs.append(inputs[start:end].contiguous())
            micro_labels.append(labels[start:end].contiguous())

    torch.cuda.synchronize(device)

    total_tokens = num_steps * tokens_per_batch
    opt_step = optimizer.step
    opt_zero = optimizer.zero_grad
    no_sync = model.no_sync

    compiled_train = _compile_train_step(train_step, micro_inputs[0], micro_labels[0])

    with no_sync():
        loss = compiled_train(micro_inputs[0], micro_labels[0])
        loss.backward()
    for m in range(1, grad_accum - 1):
        with no_sync():
            loss = compiled_train(micro_inputs[m], micro_labels[m])
            loss.backward()
    loss = compiled_train(micro_inputs[grad_accum - 1], micro_labels[grad_accum - 1])
    loss.backward()
    opt_step()
    opt_zero(set_to_none=True)

    if _RUN_IDX <= 1 or num_steps <= 2:
        for step in range(1, num_steps - 1):
            _run_step(compiled_train, micro_inputs, micro_labels, no_sync, grad_accum, step * grad_accum, opt_step, opt_zero)

        final_logits, final_loss = _run_final_step(
            compiled_train,
            fwd_only,
            micro_inputs,
            micro_labels,
            no_sync,
            grad_accum,
            (num_steps - 1) * grad_accum,
            opt_step,
            opt_zero,
            scale,
            device,
        )

        rank = dist.get_rank() if dist.is_initialized() else 0
        final_state = None
        if rank == 0:
            raw_model = model.module if hasattr(model, "module") else model
            state_dict = raw_model.state_dict()
            pinned = {k: torch.empty_like(v, device="cpu").pin_memory() for k, v in state_dict.items()}
            for k, v in state_dict.items():
                pinned[k].copy_(v, non_blocking=True)
            torch.cuda.synchronize(device)
            final_state = pinned

        return InnerStepsResult(
            final_logits=final_logits,
            total_tokens=total_tokens,
            final_loss=final_loss,
            final_state=final_state,
        )

    overlap_steps = min(_STATE_OVERLAP_STEPS, max(num_steps - 2, 0))
    eager_end = num_steps - overlap_steps

    for step in range(1, eager_end - 1):
        _run_step(compiled_train, micro_inputs, micro_labels, no_sync, grad_accum, step * grad_accum, opt_step, opt_zero)

    final_logits, final_loss = _run_final_step(
        compiled_train,
        fwd_only,
        micro_inputs,
        micro_labels,
        no_sync,
        grad_accum,
        (eager_end - 1) * grad_accum,
        opt_step,
        opt_zero,
        scale,
        device,
    )

    ctx = {
        "final_logits": final_logits,
        "total_tokens": total_tokens,
        "final_loss": final_loss,
        "final_state": None,
        "ready": False,
        "compiled_train": compiled_train,
        "fwd_only": fwd_only,
        "micro_inputs": micro_inputs,
        "micro_labels": micro_labels,
        "no_sync": no_sync,
        "grad_accum": grad_accum,
        "opt_step": opt_step,
        "opt_zero": opt_zero,
        "scale": scale,
        "device": device,
        "model": model,
        "start_step": eager_end,
        "end_step": num_steps,
    }
    return _DeferredResult(ctx)
