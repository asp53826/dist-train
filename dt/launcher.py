"""Process group setup and a spawn helper.

Uses a file store rather than TCP so nothing binds a port, which makes running
several world sizes back to back in one benchmark reliable — a TCP store can
leave the port in TIME_WAIT and the next run picks up the previous rendezvous.
"""

from __future__ import annotations

import os
import tempfile
import traceback
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@contextmanager
def process_group(rank: int, world: int, init_file: str, backend: str = "gloo"):
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
    )
    try:
        # one thread per process, otherwise 4 workers each spawning 10 BLAS
        # threads oversubscribe the machine and the scaling numbers become a
        # measurement of the scheduler rather than of the algorithm
        torch.set_num_threads(1)
        yield
    finally:
        dist.barrier()
        dist.destroy_process_group()


def _detach(obj):
    """Convert torch tensors to numpy before they cross a process boundary.

    Tensors pickle by handing over a shared memory handle rather than copying.
    If the producing process then exits before the parent has consumed it, the
    parent blocks forever on a queue read. Converting to numpy makes the return
    value pickle by value, which is what a result channel wants anyway.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone().numpy()
    if isinstance(obj, dict):
        return {k: _detach(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach(v) for v in obj)
    return obj


def _entry(rank, world, init_file, backend, fn, args, result_queue):
    try:
        with process_group(rank, world, init_file, backend):
            out = fn(rank, world, *args)
        result_queue.put((rank, _detach(out), None))
    except Exception:
        result_queue.put((rank, None, traceback.format_exc()))


def run_workers(fn, world: int, args: tuple = (), backend: str = "gloo",
                timeout: float = 300.0):
    """Spawn `world` processes running fn(rank, world, *args), collect returns.

    Returns results in rank order. Raises if any worker did, with that
    worker's traceback rather than a bare exit code.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    fd, init_file = tempfile.mkstemp(prefix="dt-rdzv-")
    os.close(fd)
    os.unlink(init_file)  # torch wants to create it itself

    procs = []
    try:
        for rank in range(world):
            p = ctx.Process(target=_entry,
                            args=(rank, world, init_file, backend, fn, args, queue))
            p.start()
            procs.append(p)

        results = {}
        errors = {}
        for _ in range(world):
            rank, out, err = queue.get(timeout=timeout)
            if err:
                errors[rank] = err
            else:
                results[rank] = out

        for p in procs:
            p.join(timeout=timeout)

        if errors:
            first = min(errors)
            raise RuntimeError(f"rank {first} failed:\n{errors[first]}")
        return [results[r] for r in range(world)]
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        if os.path.exists(init_file):
            os.unlink(init_file)
