"""Splitting a model across workers, two different ways.

Data parallel replicates the model and splits the batch, which stops working
the moment one copy of the model doesn't fit on one worker. Then you split the
model instead, and there are two axes to split along.

TENSOR PARALLEL splits individual layers. A linear layer y = xA is cut into
column blocks A = [A1 A2], each worker computes x @ Ai, and the outputs
concatenate. The next layer is then cut by rows so its input is already
sharded, which means an MLP block needs exactly one all-reduce instead of two:

    x --> ColumnParallel --> activation --> RowParallel --> all_reduce --> y
          (no comm, output      (elementwise,   (needs the      (one collective
           is sharded)           stays sharded)  sharded input)  per block)

That pairing is the whole trick. Column-then-row is what Megatron does and it
is why tensor parallel is viable at all.

PIPELINE PARALLEL splits by depth: worker 0 holds the first layers, worker 1
the next, and activations flow forward while gradients flow back. The obvious
problem is the bubble, since worker 1 has nothing to do until worker 0 finishes.
Splitting the batch into micro-batches keeps everyone busy and shrinks the
bubble to (stages - 1) / (micro_batches + stages - 1), which is measured in
bench/pipeline.py.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .collectives import ring_all_gather, ring_all_reduce


def _split_size(total: int, world: int, rank: int) -> tuple[int, int]:
    base, extra = divmod(total, world)
    size = base + (1 if rank < extra else 0)
    start = rank * base + min(rank, extra)
    return start, size


class ColumnParallelLinear(nn.Module):
    """Splits the weight by output features. No communication in forward.

    Each worker produces a slice of the output. `gather_output=False` leaves it
    sharded, which is what you want when a RowParallelLinear comes next.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gather_output: bool = False, seed: int = 0):
        super().__init__()
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        start, size = _split_size(out_features, self.world, self.rank)
        self.start, self.size = start, size
        # every rank's shard length, known analytically so the gather needs no
        # extra round trip to discover them
        self.all_sizes = [_split_size(out_features, self.world, r)[1]
                          for r in range(self.world)]

        # build the full weight from a shared seed, then keep only this
        # worker's slice. that way a sharded model is numerically identical to
        # the unsharded one, which is what makes the equivalence test possible.
        g = torch.Generator().manual_seed(seed)
        full_w = torch.randn(out_features, in_features, generator=g) / (in_features ** 0.5)
        self.weight = nn.Parameter(full_w[start : start + size].clone())
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output and self.world > 1:
            parts = ring_all_gather(out.contiguous(), sizes=self.all_sizes)
            out = torch.cat(parts, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    """Splits the weight by input features. Needs one all-reduce in forward.

    Expects an already-sharded input, ie the output of a ColumnParallelLinear.
    Each worker computes a partial sum over its slice of the input dimension,
    and the all-reduce completes the sum.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 seed: int = 0):
        super().__init__()
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        start, size = _split_size(in_features, self.world, self.rank)
        self.start, self.size = start, size

        g = torch.Generator().manual_seed(seed)
        full_w = torch.randn(out_features, in_features, generator=g) / (in_features ** 0.5)
        self.weight = nn.Parameter(full_w[:, start : start + size].clone())
        # bias is added once, after the reduce, or it would be counted N times
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        out = F.linear(x, self.weight)
        if self.world > 1:
            ring_all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out


class ParallelMLP(nn.Module):
    """The column-then-row pairing: one collective per block, not two."""

    def __init__(self, d_model: int, d_ff: int, seed: int = 0):
        super().__init__()
        self.up = ColumnParallelLinear(d_model, d_ff, gather_output=False, seed=seed)
        self.down = RowParallelLinear(d_ff, d_model, seed=seed + 1)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class ReferenceMLP(nn.Module):
    """The same maths on one worker, built from the same seeds.

    Only here so the tests can assert the sharded version computes the same
    thing. A tensor parallel layer that is merely plausible is worthless.
    """

    def __init__(self, d_model: int, d_ff: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w1 = torch.randn(d_ff, d_model, generator=g) / (d_model ** 0.5)
        g2 = torch.Generator().manual_seed(seed + 1)
        w2 = torch.randn(d_model, d_ff, generator=g2) / (d_ff ** 0.5)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        with torch.no_grad():
            self.up.weight.copy_(w1)
            self.up.bias.zero_()
            self.down.weight.copy_(w2)
            self.down.bias.zero_()

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


# ------------------------------------------------------------- pipeline

class PipelineStage(nn.Module):
    """One contiguous slice of the layer stack, owned by one worker."""

    def __init__(self, layers: nn.Sequential, stage: int, num_stages: int):
        super().__init__()
        self.layers = layers
        self.stage = stage
        self.num_stages = num_stages

    @property
    def is_first(self) -> bool:
        return self.stage == 0

    @property
    def is_last(self) -> bool:
        return self.stage == self.num_stages - 1

    def forward(self, x):
        return self.layers(x)


def split_layers(layers: list[nn.Module], num_stages: int, stage: int) -> nn.Sequential:
    start, size = _split_size(len(layers), num_stages, stage)
    return nn.Sequential(*layers[start : start + size])


def pipeline_forward(stage: PipelineStage, x: torch.Tensor | None,
                     micro_batches: int, batch_shape: tuple, dtype=torch.float32):
    """GPipe style forward: push micro-batches through the stages in order.

    Forward only. Backward through a pipeline needs stashed activations and a
    schedule (1F1B) to keep memory bounded, which is a separate problem and not
    implemented here.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    outputs = []

    for m in range(micro_batches):
        if stage.is_first:
            micro = x[m]
        else:
            micro = torch.empty(batch_shape, dtype=dtype)
            dist.recv(micro, rank - 1)

        out = stage(micro)

        if stage.is_last:
            outputs.append(out)
        else:
            dist.send(out.contiguous(), rank + 1)

    return outputs if stage.is_last else None


def bubble_fraction(num_stages: int, micro_batches: int) -> float:
    """Share of the pipeline's time that stages spend idle.

    (S-1)/(M+S-1). At M=1 it's (S-1)/S, so a 4 stage pipeline wastes 75% of its
    capacity. At M=16 the same pipeline wastes 16%.
    """
    if num_stages <= 1:
        return 0.0
    return (num_stages - 1) / (micro_batches + num_stages - 1)
