
'''
#   MFU     TPS         Wall Time   Success     Date
1	62.02%	16,940.369	77.37s  	Yes	        Mar 23, 20:34
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
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False})

    _UNCHECKPOINT_LAST_N = 6  # Qwen2.5-7B has 28 layers; uncheckpoint last N to save memory
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = 0
            if hasattr(layer, "gradient_checkpointing") and idx >= num_layers - _UNCHECKPOINT_LAST_N:
                layer.gradient_checkpointing = False

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
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

    # Fused forward+loss: avoids materializing full logits tensor
    # Scale is baked in so the multiply gets fused into the graph
    def fwd_loss(input_ids, labels):
        logits = model(input_ids).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return loss * scale

    compiled_train = torch.compile(fwd_loss, mode="reduce-overhead", dynamic=False, fullgraph=True)

    # Uncompiled forward for the single final micro-batch where we need logits
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

    # Pre-load all batches and split into micro-batches
    all_micro_inputs = []
    all_micro_labels = []
    tokens_per_batch = 0
    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        inputs = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        tokens_per_batch = batch.numel()
        for m in range(ga):
            s = m * mb
            e = s + mb
            all_micro_inputs.append(inputs[s:e].contiguous())
            all_micro_labels.append(labels[s:e].contiguous())

    torch.cuda.synchronize(device)

    total_tokens = num_steps * tokens_per_batch
    opt_step = optimizer.step
    opt_zero = optimizer.zero_grad
    no_sync = model.no_sync

    # Warmup: compile the fused train path
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

    # Step 0: first micro-batch (already computed above)
    with no_sync():
        loss.backward()

    # Step 0: remaining micro-batches
    for m in range(1, ga - 1):
        with no_sync():
            loss = compiled_train(all_micro_inputs[m], all_micro_labels[m])
            loss.backward()

    # Step 0: last micro-batch (WITH gradient sync)
    loss = compiled_train(all_micro_inputs[ga - 1], all_micro_labels[ga - 1])
    loss.backward()
    opt_step()
    opt_zero(set_to_none=True)

    # Main training loop (steps 1 to num_steps-2: fused forward+loss, no logits)
    for step in range(1, num_steps - 1):
        base = step * ga
        for m in range(ga - 1):
            with no_sync():
                loss = compiled_train(all_micro_inputs[base + m], all_micro_labels[base + m])
                loss.backward()
        loss = compiled_train(all_micro_inputs[base + ga - 1], all_micro_labels[base + ga - 1])
        loss.backward()
        opt_step()
        opt_zero(set_to_none=True)

    # Final step: accumulate loss as tensors to avoid .item() CUDA syncs
    base = (num_steps - 1) * ga
    loss_accum = torch.zeros(1, device=device)
    for m in range(ga - 1):
        with no_sync():
            loss = compiled_train(all_micro_inputs[base + m], all_micro_labels[base + m])
            loss.backward()
            loss_accum += loss.detach()
    # Last micro-batch: need logits for final_logits return
    logits = fwd_only(all_micro_inputs[base + ga - 1])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), all_micro_labels[base + ga - 1].reshape(-1), ignore_index=-100)
    (loss * scale).backward()
    loss_accum += loss.detach() * scale

    opt_step()
    opt_zero(set_to_none=True)

    final_logits = logits.detach()
    final_loss = loss_accum.item()

    rank = dist.get_rank() if dist.is_initialized() else 0
    final_state = None
    if rank == 0:
        raw_model = model.module if hasattr(model, "module") else model
        sd = raw_model.state_dict()
        pinned = {k: torch.empty_like(v, device="cpu").pin_memory() for k, v in sd.items()}
        for k, v in sd.items():
            pinned[k].copy_(v, non_blocking=True)
        torch.cuda.synchronize(device)
        final_state = pinned

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=final_state,
    )
