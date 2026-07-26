"""Data parallel and model parallel correctness.

The bar for all of these is the same: distributed must compute what one worker
would have computed. Anything else is a slower way to get a different answer.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from dt.data_parallel import DataParallel, build_buckets, shard_batch
from dt.launcher import run_workers
from dt.model_parallel import (
    ColumnParallelLinear,
    ParallelMLP,
    ReferenceMLP,
    RowParallelLinear,
    bubble_fraction,
    split_layers,
)


def make_model(seed=0, d_in=16, d_hidden=32, d_out=4):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def fixed_batch(n=64, d_in=16, d_out=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_in, generator=g)
    y = torch.randn(n, d_out, generator=g)
    return x, y


# ------------------------------------------------- data parallel equivalence

def _single_worker_grads():
    x, y = fixed_batch()
    model = make_model()
    nn.functional.mse_loss(model(x), y).backward()
    return [p.grad.clone() for p in model.parameters()]


def _w_dp_matches_single(rank, world, algo):
    x, y = fixed_batch()
    model = DataParallel(make_model(), algo=algo, bucket_mb=0.001)
    xs, ys = shard_batch(x, y, rank, world)

    nn.functional.mse_loss(model(xs), ys).backward()
    model.sync_gradients()

    # compare in the worker and return a scalar. the reference is deterministic
    # and cheap, and scalars are far safer to send back than tensors.
    ref = _single_worker_grads()
    return max(float((p.grad - r).abs().max()) for p, r in zip(model.params, ref))


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("algo", ["ring", "naive"])
def test_data_parallel_gradients_match_one_big_batch(world, algo):
    """The defining property: N workers on batch/N each must produce the same
    gradient as one worker on the whole batch. If this drifts, distributed
    training is silently optimizing a different objective."""
    for err in run_workers(_w_dp_matches_single, world, args=(algo,)):
        assert err < 1e-5, err


def _w_replicas_agree(rank, world):
    x, y = fixed_batch()
    model = DataParallel(make_model(seed=rank), algo="ring")  # deliberately different init
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(5):
        xs, ys = shard_batch(x, y, rank, world)
        opt.zero_grad()
        nn.functional.mse_loss(model(xs), ys).backward()
        model.sync_gradients()
        opt.step()
    return model.check_replicas_agree()


@pytest.mark.parametrize("world", [2, 4])
def test_replicas_stay_identical(world):
    """Weights are broadcast at construction, so replicas that started from
    different seeds must still converge to bit-identical parameters."""
    assert all(d < 1e-6 for d in run_workers(_w_replicas_agree, world))


def _run_with_cap(rank, world, cap):
    x, y = fixed_batch()
    model = DataParallel(make_model(), bucket_mb=cap)
    xs, ys = shard_batch(x, y, rank, world)
    nn.functional.mse_loss(model(xs), ys).backward()
    model.sync_gradients()
    return model


def _w_bucketing_equivalent(rank, world):
    small = _run_with_cap(rank, world, 0.000001)
    large = _run_with_cap(rank, world, 100.0)
    diff = max(float((a.grad - b.grad).abs().max())
               for a, b in zip(small.params, large.params))
    return (diff, small.stats.all_reduce_calls, large.stats.all_reduce_calls)


def test_bucket_size_changes_call_count_not_results():
    """Bucketing is a performance knob. It must not touch the numbers."""
    for diff, calls_small, calls_large in run_workers(_w_bucketing_equivalent, 2):
        assert calls_small > calls_large, "smaller buckets mean more collectives"
        assert calls_large == 1, "everything should fit one bucket at 100MB"
        assert diff < 1e-6, diff


def test_build_buckets_covers_every_parameter():
    params = list(make_model().parameters())
    for cap in (0.000001, 0.01, 100.0):
        buckets = build_buckets(params, cap)
        seen = [p for b in buckets for p in b.params]
        assert len(seen) == len(params)
        assert {id(p) for p in seen} == {id(p) for p in params}
        assert sum(b.numel for b in buckets) == sum(p.numel() for p in params)


def _w_shard_disjoint(rank, world):
    x, y = fixed_batch(n=64)
    xs, _ = shard_batch(x, y, rank, world)
    return xs.sum().item(), xs.shape[0]


@pytest.mark.parametrize("world", [2, 4, 8])
def test_shards_are_disjoint_and_cover_the_batch(world):
    x, y = fixed_batch(n=64)
    results = run_workers(_w_shard_disjoint, world)
    assert sum(n for _, n in results) == 64
    assert sum(s for s, _ in results) == pytest.approx(x.sum().item(), abs=1e-3)


# ------------------------------------------------- tensor parallel equivalence

def _w_tensor_parallel(rank, world):
    torch.manual_seed(0)
    x = torch.randn(8, 32)
    sharded = ParallelMLP(32, 64, seed=5)
    reference = ReferenceMLP(32, 64, seed=5)
    a = sharded(x)
    b = reference(x)
    return float((a - b).abs().max())


@pytest.mark.parametrize("world", [1, 2, 4])
def test_tensor_parallel_matches_the_unsharded_model(world):
    """Column-then-row splitting has to be numerically transparent."""
    assert all(e < 1e-4 for e in run_workers(_w_tensor_parallel, world))


def _w_column_shapes(rank, world):
    layer = ColumnParallelLinear(16, 33, gather_output=False, seed=1)
    out = layer(torch.randn(4, 16))
    gathered = ColumnParallelLinear(16, 33, gather_output=True, seed=1)(torch.randn(4, 16))
    return (out.shape[-1], gathered.shape[-1], layer.size)


@pytest.mark.parametrize("world", [1, 2, 4, 5])
def test_column_parallel_shards_uneven_output(world):
    """33 outputs across 4 workers doesn't divide. Shards must still tile."""
    results = run_workers(_w_column_shapes, world)
    assert sum(local for local, _, _ in results) == 33
    assert all(full == 33 for _, full, _ in results)


def _w_row_parallel_bias_once(rank, world):
    layer = RowParallelLinear(16, 8, bias=True, seed=2)
    with torch.no_grad():
        layer.bias.fill_(1.0)
        layer.weight.zero_()
    out = layer(torch.randn(3, layer.size))
    return float((out - 1.0).abs().max())


@pytest.mark.parametrize("world", [1, 2, 4])
def test_row_parallel_adds_bias_exactly_once(world):
    """Bias after the reduce, not before, or it gets counted N times."""
    assert all(e < 1e-6 for e in run_workers(_w_row_parallel_bias_once, world))


# ------------------------------------------------------------- pipeline

def test_split_layers_partitions_the_stack():
    layers = [nn.Linear(4, 4) for _ in range(10)]
    for stages in (1, 2, 3, 4):
        pieces = [split_layers(layers, stages, s) for s in range(stages)]
        assert sum(len(p) for p in pieces) == 10
        flat = [m for p in pieces for m in p]
        assert [id(m) for m in flat] == [id(m) for m in layers], "order must be preserved"


def test_bubble_shrinks_with_micro_batches():
    """The reason pipeline parallel needs micro-batching at all."""
    assert bubble_fraction(1, 1) == 0.0
    assert bubble_fraction(4, 1) == pytest.approx(0.75)
    assert bubble_fraction(4, 16) == pytest.approx(3 / 19)
    assert bubble_fraction(4, 64) < 0.05
    # more stages is more bubble at a fixed micro-batch count
    assert bubble_fraction(8, 8) > bubble_fraction(2, 8)
