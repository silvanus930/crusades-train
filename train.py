from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None


def get_strategy():
    return {"dp_size": 2, "tp_size": 2}


_PREPARED_MODEL_IDS = set()


def _prepare_model(model):
    model_id = id(model)
    if model_id in _PREPARED_MODEL_IDS:
        return
    _PREPARED_MODEL_IDS.add(model_id)

    model.train()

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if hasattr(model, "config"):
        model.config.use_cache = False
        try:
            model.config.output_attentions = False
        except Exception:
            pass
        try:
            model.config.output_hidden_states = False
        except Exception:
            pass


def _apply_tp(model, tp_mesh):
    for _, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "o_proj"):
            parallelize_module(
                module,
                tp_mesh,
                {
                    "q_proj": ColwiseParallel(),
                    "k_proj": ColwiseParallel(),
                    "v_proj": ColwiseParallel(),
                    "o_proj": RowwiseParallel(),
                },
            )
        if hasattr(module, "gate_proj") and hasattr(module, "down_proj"):
            parallelize_module(
                module,
                tp_mesh,
                {
                    "gate_proj": ColwiseParallel(),
                    "up_proj": ColwiseParallel(),
                    "down_proj": RowwiseParallel(),
                },
            )
    return model


def _allreduce_grads(model, dp_pg):
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        local_grad = grad._local_tensor if hasattr(grad, "_local_tensor") else grad
        dist.all_reduce(local_grad, op=dist.ReduceOp.AVG, group=dp_pg)


def _gather_full_state(model):
    state = {}

    for name, param in model.named_parameters():
        value = param.data
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        state[name] = value.detach().cpu().clone()

    for name, buf in model.named_buffers():
        value = buf.data
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        state[name] = value.detach().cpu().clone()

    state_dict = model.state_dict()
    for key in state_dict:
        if key in state:
            continue
        value = state_dict[key]
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        state[key] = value.detach().cpu().clone()

    return state


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    _prepare_model(model)

    strategy = get_strategy()
    expected_gpus = strategy["dp_size"] * strategy["tp_size"]
    if num_gpus != expected_gpus:
        raise ValueError(
            f"get_strategy() requires {expected_gpus} GPUs "
            f"(dp_size={strategy['dp_size']} * tp_size={strategy['tp_size']}), "
            f"but num_gpus={num_gpus}"
        )

    dp_pg = None
    is_multi = num_gpus > 1

    if is_multi:
        mesh_2d = init_device_mesh(
            "cuda",
            (strategy["dp_size"], strategy["tp_size"]),
            mesh_dim_names=("dp", "tp"),
        )
        tp_mesh = mesh_2d["tp"]
        dp_pg = mesh_2d.get_group("dp")
        model = _apply_tp(model, tp_mesh)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=False,
        )

    batches = [None] * num_steps
    inputs = [None] * num_steps
    labels = [None] * num_steps
    tokens_per_batch = 0

    for step in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        batches[step] = batch
        inputs[step] = batch[:, :-1].contiguous()
        labels[step] = batch[:, 1:].contiguous()
        tokens_per_batch = batch.numel()

    total_tokens = num_steps * tokens_per_batch
    final_logits = None
    final_loss = 0.0
    last_step = num_steps - 1

    opt_step = optimizer.step
    opt_zero = optimizer.zero_grad
    ce_loss = F.cross_entropy

    opt_zero(set_to_none=True)

    for step in range(num_steps):
        input_ids = inputs[step]
        step_labels = labels[step]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = ce_loss(
                logits.reshape(-1, logits.size(-1)),
                step_labels.reshape(-1),
                ignore_index=-100,
            )

        loss.backward()

        if dp_pg is not None:
            _allreduce_grads(model, dp_pg)

        opt_step()
        opt_zero(set_to_none=True)

        if step == last_step:
            final_logits = logits.detach()
            final_loss = loss.item()

    if is_multi:
        rank = dist.get_rank() if dist.is_initialized() else 0
        gathered_state = _gather_full_state(model)
        final_state = gathered_state if rank == 0 else None
    else:
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=final_state,
    )
