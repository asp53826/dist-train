"""Collectives checked against torch.distributed's own implementations.

Hand written collectives are exactly the kind of code that looks right, passes
a two-worker smoke test, and quietly corrupts one chunk at world size 5. So
every one of these runs at several world sizes, including sizes that don't
divide the tensor evenly, and compares against the reference.
"""

import pytest
import torch
import torch.distributed as dist

from dt.collectives import (
    _chunk_bounds,
    bytes_moved,
    naive_all_reduce,
    ring_all_gather,
    ring_all_reduce,
    ring_broadcast,
    ring_reduce_scatter,
)
from dt.launcher import run_workers

WORLDS = [1, 2, 3, 4, 5, 8]
# deliberately awkward: primes, sizes below world size, and a size that is not
# a multiple of any world size here
SIZES = [1, 3, 7, 17, 1000, 65537]


def _local(rank, numel, seed=0):
    g = torch.Generator().manual_seed(seed * 1000 + rank + 1)
    return torch.randn(numel, generator=g)


# ---------------------------------------------------------------- all-reduce

def _w_all_reduce(rank, world):
    errors = {}
    for numel in SIZES:
        base = _local(rank, numel)
        ref = base.clone()
        dist.all_reduce(ref)
        mine = base.clone()
        ring_all_reduce(mine)
        errors[numel] = float((mine - ref).abs().max())
    return errors


@pytest.mark.parametrize("world", WORLDS)
def test_ring_all_reduce_matches_torch(world):
    for errors in run_workers(_w_all_reduce, world):
        for numel, err in errors.items():
            assert err < 1e-5, f"numel={numel} err={err}"


def _w_naive(rank, world):
    errors = {}
    for numel in (7, 1000):
        base = _local(rank, numel, seed=2)
        ref = base.clone()
        dist.all_reduce(ref)
        mine = base.clone()
        naive_all_reduce(mine)
        errors[numel] = float((mine - ref).abs().max())
    return errors


@pytest.mark.parametrize("world", [1, 2, 4])
def test_naive_all_reduce_matches_torch(world):
    """The slow baseline has to be correct too, or the comparison is worthless."""
    for errors in run_workers(_w_naive, world):
        assert all(e < 1e-5 for e in errors.values()), errors


def _w_average(rank, world):
    t = torch.full((8,), float(rank + 1))
    ring_all_reduce(t, average=True)
    expected = sum(range(1, world + 1)) / world
    return float((t - expected).abs().max())


@pytest.mark.parametrize("world", [1, 2, 4, 5])
def test_all_reduce_average(world):
    assert all(e < 1e-5 for e in run_workers(_w_average, world))


def _w_inplace(rank, world):
    t = torch.ones(10) * (rank + 1)
    out = ring_all_reduce(t)
    return out.data_ptr() == t.data_ptr()


def test_all_reduce_is_in_place():
    assert all(run_workers(_w_inplace, 3))


def _w_2d(rank, world):
    """Shape must survive: the implementation flattens internally."""
    t = torch.ones(4, 6) * (rank + 1)
    ring_all_reduce(t)
    expected = float(sum(range(1, world + 1)))
    return (tuple(t.shape), float((t - expected).abs().max()))


@pytest.mark.parametrize("world", [2, 4])
def test_all_reduce_preserves_shape(world):
    for shape, err in run_workers(_w_2d, world):
        assert shape == (4, 6) and err < 1e-5


# ---------------------------------------------------------------- all-gather

def _w_all_gather(rank, world):
    t = _local(rank, 11, seed=3)
    mine = ring_all_gather(t)
    ref = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(ref, t)
    return max(float((a - b).abs().max()) for a, b in zip(mine, ref))


@pytest.mark.parametrize("world", [1, 2, 3, 4, 8])
def test_ring_all_gather_matches_torch(world):
    assert all(e < 1e-6 for e in run_workers(_w_all_gather, world))


# ------------------------------------------------------------ reduce-scatter

def _w_reduce_scatter(rank, world):
    base = _local(rank, 100, seed=4)
    full = base.clone()
    dist.all_reduce(full)
    mine = ring_reduce_scatter(base.clone())

    bounds = _chunk_bounds(100, world)
    lo, hi = bounds[(rank + 1) % world]
    return (mine.numel(), float((mine - full[lo:hi]).abs().max()))


@pytest.mark.parametrize("world", [1, 2, 4, 5])
def test_reduce_scatter_is_a_slice_of_all_reduce(world):
    """Each worker's slice must equal the corresponding slice of a full
    all-reduce, which is what makes sharded optimizer state safe."""
    results = run_workers(_w_reduce_scatter, world)
    assert sum(n for n, _ in results) == 100, "slices must tile the tensor exactly"
    assert all(err < 1e-5 for _, err in results)


# ---------------------------------------------------------------- broadcast

def _w_broadcast(rank, world):
    t = torch.full((16,), float(rank)) if rank != 2 % world else torch.arange(16.0)
    src = 2 % world
    t = torch.arange(16.0) if rank == src else torch.zeros(16)
    ring_broadcast(t, src=src)
    return float((t - torch.arange(16.0)).abs().max())


@pytest.mark.parametrize("world", [1, 2, 3, 5])
def test_ring_broadcast(world):
    assert all(e == 0.0 for e in run_workers(_w_broadcast, world))


# ---------------------------------------------------------------- cost model

def test_chunk_bounds_tile_exactly():
    for numel in (1, 7, 100, 65537):
        for world in (1, 2, 3, 5, 8):
            b = _chunk_bounds(numel, world)
            assert len(b) == world
            assert b[0][0] == 0 and b[-1][1] == numel
            assert all(b[i][1] == b[i + 1][0] for i in range(world - 1))
            assert all(hi >= lo for lo, hi in b)


def test_chunk_sizes_differ_by_at_most_one():
    b = _chunk_bounds(100, 7)
    sizes = [hi - lo for lo, hi in b]
    assert max(sizes) - min(sizes) <= 1


def test_ring_traffic_is_bounded_but_naive_grows():
    """The whole argument for a ring, as arithmetic."""
    size, item = 1_000_000, 4
    ring = [bytes_moved(size, w, item, "ring") for w in (2, 4, 8, 64)]
    naive = [bytes_moved(size, w, item, "naive") for w in (2, 4, 8, 64)]

    total = size * item
    assert all(r < 2 * total for r in ring)
    assert ring[-1] / ring[0] < 2.0        # ring barely moves with world size
    assert naive[-1] / naive[0] == pytest.approx(63.0)  # naive grows linearly
