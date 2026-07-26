"""Collective operations built from point-to-point send and recv.

`dist.all_reduce` is one line and it hides the only thing worth understanding
about distributed training: what it costs. So these are written out from
`isend`/`irecv` and nothing else.

The cost model, for N workers and a tensor of S bytes:

  naive (gather to root, reduce, broadcast)
      root sends and receives (N-1)*S. Every extra worker adds S bytes of
      traffic through one machine, so the root's link is the bottleneck and
      scaling gets worse the more workers you add.

  ring all-reduce
      each worker sends 2*(N-1)/N * S and receives the same. That approaches
      2*S and *stops*, no matter how many workers there are. Nothing is a
      bottleneck because every link carries identical traffic.

That second property is why every real data-parallel trainer uses a ring (or a
tree, which trades bandwidth for latency at small sizes). `bench/collectives.py`
measures both.

Ring all-reduce runs in two phases, each N-1 steps:

  reduce-scatter   chunk i ends up fully summed, on exactly one worker
  all-gather       that chunk is passed around until everyone has it
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _chunk_bounds(numel: int, world: int) -> list[tuple[int, int]]:
    """Split into `world` near-equal chunks. Handles numel < world."""
    base, extra = divmod(numel, world)
    bounds = []
    start = 0
    for i in range(world):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def ring_all_reduce(tensor: torch.Tensor, group=None, average: bool = False) -> torch.Tensor:
    """In-place all-reduce (sum) over a ring. Returns the same tensor.

    Each worker only ever talks to its two neighbours, which is what keeps the
    per-worker traffic independent of world size.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    if world == 1:
        return tensor.div_(1) if not average else tensor

    flat = tensor.view(-1)
    bounds = _chunk_bounds(flat.numel(), world)
    send_to = (rank + 1) % world
    recv_from = (rank - 1) % world

    # phase 1: reduce-scatter.
    # on step s, send the chunk we finished accumulating last step and receive
    # the one our predecessor just finished. after N-1 steps chunk
    # (rank+1) % world is fully summed here and nowhere else.
    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        s0, s1 = bounds[send_idx]
        r0, r1 = bounds[recv_idx]

        recv_buf = torch.empty(r1 - r0, dtype=flat.dtype, device=flat.device)
        reqs = []
        if s1 > s0:
            reqs.append(dist.isend(flat[s0:s1].contiguous(), send_to, group))
        if r1 > r0:
            reqs.append(dist.irecv(recv_buf, recv_from, group))
        for r in reqs:
            r.wait()
        if r1 > r0:
            flat[r0:r1] += recv_buf

    # phase 2: all-gather. same ring, but overwrite instead of accumulate,
    # because the chunk being passed is already final.
    for step in range(world - 1):
        send_idx = (rank - step + 1) % world
        recv_idx = (rank - step) % world
        s0, s1 = bounds[send_idx]
        r0, r1 = bounds[recv_idx]

        recv_buf = torch.empty(r1 - r0, dtype=flat.dtype, device=flat.device)
        reqs = []
        if s1 > s0:
            reqs.append(dist.isend(flat[s0:s1].contiguous(), send_to, group))
        if r1 > r0:
            reqs.append(dist.irecv(recv_buf, recv_from, group))
        for r in reqs:
            r.wait()
        if r1 > r0:
            flat[r0:r1] = recv_buf

    if average:
        flat.div_(world)
    return tensor


def naive_all_reduce(tensor: torch.Tensor, group=None, average: bool = False) -> torch.Tensor:
    """Gather everything to rank 0, sum, broadcast back.

    Correct, simple, and the traffic through rank 0 grows linearly with world
    size. Here to be measured against, not used.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    if world == 1:
        return tensor

    flat = tensor.view(-1)
    if rank == 0:
        buf = torch.empty_like(flat)
        for src in range(1, world):
            dist.recv(buf, src, group)
            flat += buf
        for dst in range(1, world):
            dist.send(flat, dst, group)
    else:
        dist.send(flat, 0, group)
        dist.recv(flat, 0, group)

    if average:
        flat.div_(world)
    return tensor


def ring_all_gather(tensor: torch.Tensor, group=None,
                    sizes: list[int] | None = None) -> list[torch.Tensor]:
    """Every worker ends up with every worker's tensor, in rank order.

    `sizes` gives each rank's length along the last dimension. It's needed
    whenever the shards are uneven — splitting 33 outputs across 5 workers
    gives 7,7,7,6,6, and sizing every receive buffer like the local tensor
    means a 7-element send meets a 6-element receive and the ring deadlocks.
    Nothing errors, the workers just wait for each other forever.

    Left as None when every shard really is the same size, which is the common
    case and needs no extra bookkeeping.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    out: list[torch.Tensor | None] = [None] * world
    out[rank] = tensor.clone()
    if world == 1:
        return [out[0]]

    def buffer_for(idx: int) -> torch.Tensor:
        if sizes is None:
            return torch.empty_like(tensor)
        shape = list(tensor.shape)
        shape[-1] = sizes[idx]
        return torch.empty(shape, dtype=tensor.dtype, device=tensor.device)

    send_to = (rank + 1) % world
    recv_from = (rank - 1) % world
    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        recv_buf = buffer_for(recv_idx)
        r1 = dist.isend(out[send_idx].contiguous(), send_to, group)
        r2 = dist.irecv(recv_buf, recv_from, group)
        r1.wait()
        r2.wait()
        out[recv_idx] = recv_buf
    return out


def ring_reduce_scatter(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Sum across workers, but each keeps only its own slice.

    Half of an all-reduce. This is what ZeRO / FSDP use to shard optimizer
    state: no worker ever holds the full reduced gradient.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    flat = tensor.view(-1)
    bounds = _chunk_bounds(flat.numel(), world)
    if world == 1:
        return flat.clone()

    send_to = (rank + 1) % world
    recv_from = (rank - 1) % world
    work = flat.clone()
    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        s0, s1 = bounds[send_idx]
        r0, r1 = bounds[recv_idx]
        recv_buf = torch.empty(r1 - r0, dtype=flat.dtype, device=flat.device)
        reqs = []
        if s1 > s0:
            reqs.append(dist.isend(work[s0:s1].contiguous(), send_to, group))
        if r1 > r0:
            reqs.append(dist.irecv(recv_buf, recv_from, group))
        for r in reqs:
            r.wait()
        if r1 > r0:
            work[r0:r1] += recv_buf

    lo, hi = bounds[(rank + 1) % world]
    return work[lo:hi].clone()


def ring_broadcast(tensor: torch.Tensor, src: int = 0, group=None) -> torch.Tensor:
    """Pass along the ring rather than src sending N-1 times.

    Same total bytes, but src's link carries S instead of (N-1)*S, so it
    doesn't become the bottleneck.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    if world == 1:
        return tensor
    hops = (rank - src) % world
    if hops > 0:
        dist.recv(tensor, (rank - 1) % world, group)
    if hops < world - 1:
        dist.send(tensor, (rank + 1) % world, group)
    return tensor


def bytes_moved(numel: int, world: int, itemsize: int, algo: str) -> int:
    """Bytes this worker sends, for the cost model in the README."""
    size = numel * itemsize
    if world <= 1:
        return 0
    if algo == "ring":
        return int(2 * (world - 1) / world * size)
    if algo == "naive":
        # rank 0 carries (world-1) sends plus (world-1) receives
        return (world - 1) * size
    raise ValueError(algo)
