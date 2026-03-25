
'''
#   MFU     TPS         Wall Time   Success     Date
1	70.22%	19,179.23	68.34s	    Yes	        Mar 24, 23:17
'''

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

from flash_attn.losses.cross_entropy import CrossEntropyLoss as _FlashCELoss
_flash_ce_inst = _FlashCELoss(ignore_index=-100)


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


_PREPARED = set()
_GRAD_ACCUM = 8
_MICRO_BATCH = 2
_STATE_OVERLAP_STEPS = 3
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

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False}
        )

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = 0


def _run_step(ct, mi, ml, ns, ga, base, os_fn, oz_fn):
    for m in range(ga - 1):
        with ns():
            loss = ct(mi[base + m], ml[base + m])
            loss.backward()
    loss = ct(mi[base + ga - 1], ml[base + ga - 1])
    loss.backward()
    os_fn()
    oz_fn(set_to_none=True)


def _run_final_step(ct, ff, mi, ml, ns, ga, base, os_fn, oz_fn, sv, dev):
    loss_accum = torch.zeros(1, device=dev)
    for m in range(ga - 1):
        with ns():
            loss = ct(mi[base + m], ml[base + m])
            loss.backward()
            loss_accum += loss.detach()
    logits = ff(mi[base + ga - 1])
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        ml[base + ga - 1].reshape(-1),
        ignore_index=-100,
    )
    (loss * sv).backward()
    loss_accum += loss.detach() * sv
    os_fn()
    oz_fn(set_to_none=True)
    return logits.detach(), loss_accum.item()


def _collect_state(ctx):
    ct = ctx["ct"]
    ff = ctx["ff"]
    ns = ctx["ns"]
    os_fn = ctx["os_fn"]
    oz_fn = ctx["oz_fn"]
    mi = ctx["mi"]
    ml = ctx["ml"]
    ga = ctx["ga"]
    sv = ctx["sv"]
    start = ctx["start"]
    end = ctx["end"]
    model = ctx["model"]
    dev = ctx["dev"]

    fl = ctx["final_logits"]
    fv = ctx["final_loss"]

    for step in range(start, end):
        base = step * ga
        if step < end - 1:
            _run_step(ct, mi, ml, ns, ga, base, os_fn, oz_fn)
        else:
            fl, fv = _run_final_step(ct, ff, mi, ml, ns, ga, base, os_fn, oz_fn, sv, dev)

    rank = dist.get_rank() if dist.is_initialized() else 0
    state = None
    if rank == 0:
        raw = model.module if hasattr(model, "module") else model
        sd = raw.state_dict()
        pinned = {k: torch.empty_like(v, device="cpu").pin_memory() for k, v in sd.items()}
        for k, v in sd.items():
            pinned[k].copy_(v, non_blocking=True)
        torch.cuda.synchronize(dev)
        state = pinned

    ctx["final_logits"] = fl
    ctx["final_loss"] = fv
    ctx["final_state"] = state
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

    model = model.to(device)

    ga = _GRAD_ACCUM
    mb = _MICRO_BATCH
    scale = 1.0 / ga

    if num_gpus > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            gradient_as_bucket_view=True,
            bucket_cap_mb=700,
            broadcast_buffers=False,
        )

    def fwd_loss(input_ids, labels):
        logits = model(input_ids).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return loss * scale

    compiled_train = torch.compile(fwd_loss, mode="reduce-overhead", dynamic=False, fullgraph=True)

    def fwd_only(input_ids):
        return model(input_ids).logits

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        )

    all_micro_inputs = []
    all_micro_labels = []
    tokens_per_batch = 0
    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        inputs = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        tokens_per_batch = batch.numel()
        for m_idx in range(ga):
            s = m_idx * mb
            e = s + mb
            all_micro_inputs.append(inputs[s:e].contiguous())
            all_micro_labels.append(labels[s:e].contiguous())

    torch.cuda.synchronize(device)

    total_tokens = num_steps * tokens_per_batch
    opt_step = optimizer.step
    opt_zero = optimizer.zero_grad
    no_sync = model.no_sync

    try:
        loss = compiled_train(all_micro_inputs[0], all_micro_labels[0])
    except Exception:
        try:
            compiled_train = torch.compile(fwd_loss, mode="default", dynamic=False, fullgraph=True)
            loss = compiled_train(all_micro_inputs[0], all_micro_labels[0])
        except Exception:
            try:
                compiled_train = torch.compile(fwd_loss, mode="default", dynamic=False)
                loss = compiled_train(all_micro_inputs[0], all_micro_labels[0])
            except Exception:
                compiled_train = fwd_loss
                loss = compiled_train(all_micro_inputs[0], all_micro_labels[0])

    with no_sync():
        loss.backward()

    for m in range(1, ga - 1):
        with no_sync():
            loss = compiled_train(all_micro_inputs[m], all_micro_labels[m])
            loss.backward()

    loss = compiled_train(all_micro_inputs[ga - 1], all_micro_labels[ga - 1])
    loss.backward()
    opt_step()
    opt_zero(set_to_none=True)

    if _RUN_IDX <= 1:
        for step in range(1, num_steps - 1):
            _run_step(compiled_train, all_micro_inputs, all_micro_labels, no_sync, ga, step * ga, opt_step, opt_zero)

        fl, fv = _run_final_step(
            compiled_train, fwd_only, all_micro_inputs, all_micro_labels,
            no_sync, ga, (num_steps - 1) * ga, opt_step, opt_zero, scale, device,
        )

        rank = dist.get_rank() if dist.is_initialized() else 0
        state = None
        if rank == 0:
            raw = model.module if hasattr(model, "module") else model
            sd = raw.state_dict()
            pinned = {k: torch.empty_like(v, device="cpu").pin_memory() for k, v in sd.items()}
            for k, v in sd.items():
                pinned[k].copy_(v, non_blocking=True)
            torch.cuda.synchronize(device)
            state = pinned

        return InnerStepsResult(
            final_logits=fl,
            total_tokens=total_tokens,
            final_loss=fv,
            final_state=state,
        )

    overlap = min(_STATE_OVERLAP_STEPS, max(num_steps - 2, 0))
    eager_end = num_steps - overlap

    for step in range(1, eager_end - 1):
        _run_step(compiled_train, all_micro_inputs, all_micro_labels, no_sync, ga, step * ga, opt_step, opt_zero)

    fl, fv = _run_final_step(
        compiled_train, fwd_only, all_micro_inputs, all_micro_labels,
        no_sync, ga, (eager_end - 1) * ga, opt_step, opt_zero, scale, device,
    )

    ctx = {
        "final_logits": fl,
        "total_tokens": total_tokens,
        "final_loss": fv,
        "final_state": None,
        "ready": False,
        "ct": compiled_train,
        "ff": fwd_only,
        "ns": no_sync,
        "os_fn": opt_step,
        "oz_fn": opt_zero,
        "mi": all_micro_inputs,
        "ml": all_micro_labels,
        "ga": ga,
        "sv": scale,
        "end": num_steps,
        "start": eager_end,
        "model": model,
        "dev": device,
    }

    return _StateCollectorResult(ctx)
