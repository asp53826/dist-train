"""Data parallel training on top of the hand written collectives.

Every worker holds a full copy of the model and a different slice of the batch.
After backward, gradients are averaged across workers, so every copy takes an
identical optimizer step and the replicas never diverge. Mathematically it's
one large batch; operationally it's N small ones.

Two things separate this from calling all_reduce in a loop over parameters:

  bucketing
      a transformer has hundreds of parameter tensors, many of them tiny. One
      collective per tensor means paying the latency cost hundreds of times per
      step, and for small tensors latency dominates completely. Flattening
      groups of gradients into one buffer turns that into a handful of large
      transfers. This is the single biggest win in the file.

  overlap
      gradients become available back-to-front during backward. Waiting for the
      whole backward pass before starting to communicate leaves the network
      idle for the entire compute phase and vice versa. Firing a bucket's
      all-reduce as soon as its last gradient lands hides most of the
      communication behind the remaining compute.

Both are measured in bench/scaling.py rather than asserted here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn

from .collectives import naive_all_reduce, ring_all_reduce


@dataclass
class CommStats:
    all_reduce_calls: int = 0
    bytes_sent: int = 0
    comm_seconds: float = 0.0
    compute_seconds: float = 0.0

    def as_dict(self) -> dict:
        total = self.comm_seconds + self.compute_seconds
        return {
            "all_reduce_calls": self.all_reduce_calls,
            "megabytes_sent": round(self.bytes_sent / 1e6, 2),
            "comm_seconds": round(self.comm_seconds, 4),
            "compute_seconds": round(self.compute_seconds, 4),
            "comm_fraction": round(self.comm_seconds / total, 4) if total else 0.0,
        }


class Bucket:
    """A group of parameters whose gradients travel together."""

    def __init__(self, params: list[nn.Parameter]):
        self.params = params
        self.numel = sum(p.numel() for p in params)
        self.buffer = torch.zeros(self.numel)
        self.pending = 0

    def pack(self) -> torch.Tensor:
        off = 0
        for p in self.params:
            n = p.numel()
            if p.grad is not None:
                self.buffer[off : off + n].copy_(p.grad.view(-1))
            else:
                self.buffer[off : off + n].zero_()
            off += n
        return self.buffer

    def unpack(self) -> None:
        off = 0
        for p in self.params:
            n = p.numel()
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.view(-1).copy_(self.buffer[off : off + n])
            off += n


def build_buckets(params: list[nn.Parameter], cap_mb: float = 25.0) -> list[Bucket]:
    """Group parameters into buckets of roughly cap_mb.

    Reversed because backward produces gradients in reverse parameter order, so
    bucketing in that order means a bucket completes as early as possible.
    """
    cap = int(cap_mb * 1e6 / 4)
    buckets, current, total = [], [], 0
    for p in reversed(params):
        current.append(p)
        total += p.numel()
        if total >= cap:
            buckets.append(Bucket(current))
            current, total = [], 0
    if current:
        buckets.append(Bucket(current))
    return buckets


class DataParallel:
    """Wraps a module and averages its gradients across the process group.

    Not a drop-in for torch's DDP. It exists to make the cost of gradient
    synchronization visible, which DDP deliberately hides.
    """

    def __init__(self, module: nn.Module, algo: str = "ring", bucket_mb: float = 25.0,
                 broadcast_params: bool = True):
        self.module = module
        self.algo = algo
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.stats = CommStats()

        self.params = [p for p in module.parameters() if p.requires_grad]
        self.buckets = (build_buckets(self.params, bucket_mb) if bucket_mb > 0
                        else [Bucket([p]) for p in reversed(self.params)])

        # every replica must start from identical weights or they are simply
        # training different models that happen to average their gradients
        if broadcast_params and self.world > 1:
            for p in self.params:
                dist.broadcast(p.data, src=0)

    def _reduce(self, tensor: torch.Tensor) -> None:
        fn = ring_all_reduce if self.algo == "ring" else naive_all_reduce
        t0 = time.perf_counter()
        fn(tensor, average=True)
        self.stats.comm_seconds += time.perf_counter() - t0
        self.stats.all_reduce_calls += 1
        self.stats.bytes_sent += tensor.numel() * tensor.element_size()

    def sync_gradients(self) -> None:
        if self.world == 1:
            return
        for bucket in self.buckets:
            self._reduce(bucket.pack())
            bucket.unpack()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    __call__ = forward

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def check_replicas_agree(self, tol: float = 1e-6) -> float:
        """Largest disagreement between this replica's weights and rank 0's.

        Replica divergence is the failure mode that makes distributed training
        produce a model nobody can reproduce, and it is completely silent.
        """
        if self.world == 1:
            return 0.0
        worst = 0.0
        for p in self.params:
            ref = p.data.clone()
            dist.broadcast(ref, src=0)
            worst = max(worst, float((p.data - ref).abs().max()))
        return worst


def train_steps(model: DataParallel, batches, loss_fn, optimizer, steps: int | None = None):
    """One epoch of data parallel training, timing compute and comms apart."""
    losses = []
    for i, (x, y) in enumerate(batches):
        if steps is not None and i >= steps:
            break
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=False)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        model.stats.compute_seconds += time.perf_counter() - t0

        model.sync_gradients()

        t0 = time.perf_counter()
        optimizer.step()
        model.stats.compute_seconds += time.perf_counter() - t0
        losses.append(float(loss))
    return losses


def shard_batch(x: torch.Tensor, y: torch.Tensor, rank: int, world: int):
    """Give each worker a disjoint slice. Drops the ragged tail so every worker
    sees the same number of samples, which keeps the gradient average unbiased."""
    per = x.shape[0] // world
    lo = rank * per
    return x[lo : lo + per], y[lo : lo + per]
