from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()


def initialize_distributed(*, allow_cpu: bool = False) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} GPU(s) are visible"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif allow_cpu:
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise RuntimeError(
            "CUDA is not available. Check Docker --gpus/NVIDIA Container Toolkit, "
            "or pass --allow-cpu only for a smoke test."
        )

    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(hours=6),
            device_id=device if device.type == "cuda" else None,
        )
    return DistributedContext(rank, local_rank, world_size, device)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_sums(values: dict[str, float], context: DistributedContext) -> dict[str, float]:
    if not context.distributed:
        return dict(values)
    keys = sorted(values)
    tensor = torch.tensor(
        [values[key] for key in keys], dtype=torch.float64, device=context.device
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, tensor.cpu().tolist())}
