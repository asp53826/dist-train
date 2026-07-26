#!/usr/bin/env python3
"""Scaling efficiency, measured honestly.

Two numbers people quote and one they don't:

  speedup     time(1 worker) / time(N workers)
  efficiency  speedup / N. this is the honest one. 3.2x on 4 workers is 80%
              efficiency, and the missing 20% went somewhere specific.

The "somewhere specific" is what the comm/compute split is for. Gradient
synchronization is pure overhead — it does no useful arithmetic — so efficiency
is governed by how much compute each worker does per byte of gradient it has to
send. Finding where that crosses over is the point of this file.

Note the lever is batch size, not model size. Growing the model scales gradient
bytes and per-worker compute together (both O(d^2)) so the ratio barely moves;
growing the batch adds compute against an unchanged gradient volume. The first
version of this benchmark swept model size expecting efficiency to rise, and
the numbers said the opposite. `--sweep-batches` is the corrected experiment.

Everything runs on CPU with gloo over loopback, which is *slow* — around
0.5 GB/s, against ~300 GB/s for NVLink. So communication dominates here far
more than it would on real hardware, and the absolute efficiency numbers are
pessimistic. Ratios between configurations still hold; absolute numbers do not
transfer.

  python bench/scaling.py
  python bench/scaling.py --worlds 1,2,4,8 --json results/scaling.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt.collectives import bytes_moved, naive_all_reduce, ring_all_reduce  # noqa: E402
from dt.data_parallel import DataParallel, shard_batch  # noqa: E402
from dt.launcher import run_workers  # noqa: E402


def build_model(d_model: int, layers: int) -> nn.Sequential:
    torch.manual_seed(0)
    mods: list[nn.Module] = []
    for _ in range(layers):
        mods += [nn.Linear(d_model, d_model), nn.GELU()]
    mods.append(nn.Linear(d_model, 10))
    return nn.Sequential(*mods)


# ------------------------------------------------------- collective microbench

def _w_collective(rank, world, numel, iters, algo):
    t = torch.randn(numel)
    fn = ring_all_reduce if algo == "ring" else naive_all_reduce
    fn(t.clone())  # warm up the connections
    best = None
    for _ in range(iters):
        buf = t.clone()
        t0 = time.perf_counter()
        fn(buf)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def collective_bench(worlds, sizes, iters):
    rows = []
    for numel in sizes:
        for world in worlds:
            for algo in ("ring", "naive"):
                if world == 1:
                    continue
                times = run_workers(_w_collective, world, args=(numel, iters, algo))
                slowest = max(times)  # a collective is only done when everyone is
                mb = numel * 4 / 1e6
                rows.append({
                    "algo": algo,
                    "world": world,
                    "megabytes": round(mb, 2),
                    "seconds": round(slowest, 6),
                    "effective_gbps": round(
                        bytes_moved(numel, world, 4, algo) / slowest / 1e9, 3),
                    "bytes_per_worker_mb": round(
                        bytes_moved(numel, world, 4, algo) / 1e6, 2),
                })
    return rows


# ------------------------------------------------------- training scaling

def _w_train(rank, world, d_model, layers, global_batch, steps, algo, bucket_mb):
    torch.manual_seed(0)
    model = DataParallel(build_model(d_model, layers), algo=algo, bucket_mb=bucket_mb)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    g = torch.Generator().manual_seed(1)
    x = torch.randn(global_batch, d_model, generator=g)
    y = torch.randint(0, 10, (global_batch,), generator=g)
    xs, ys = shard_batch(x, y, rank, world)
    loss_fn = nn.CrossEntropyLoss()

    # a couple of untimed steps so lazy allocations and gloo handshakes don't
    # land inside the measurement
    for _ in range(2):
        opt.zero_grad()
        loss_fn(model(xs), ys).backward()
        model.sync_gradients()
        opt.step()

    model.stats.comm_seconds = 0.0
    model.stats.compute_seconds = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        c0 = time.perf_counter()
        opt.zero_grad()
        loss_fn(model(xs), ys).backward()
        model.stats.compute_seconds += time.perf_counter() - c0

        model.sync_gradients()   # times itself into comm_seconds

        c0 = time.perf_counter()
        opt.step()
        model.stats.compute_seconds += time.perf_counter() - c0
    wall = time.perf_counter() - t0

    return {
        "wall": wall,
        "params": sum(p.numel() for p in model.params),
        **model.stats.as_dict(),
    }


def training_bench(worlds, d_model, layers, global_batch, steps, algo, bucket_mb):
    rows = []
    baseline = None
    for world in worlds:
        out = run_workers(_w_train, world,
                          args=(d_model, layers, global_batch, steps, algo, bucket_mb))
        # the step isn't finished until the slowest worker finishes it
        wall = max(r["wall"] for r in out)
        comm = max(r["comm_seconds"] for r in out)
        compute = max(r["compute_seconds"] for r in out)
        if baseline is None:
            baseline = wall
        rows.append({
            "world": world,
            "algo": algo,
            "params": out[0]["params"],
            "seconds": round(wall, 3),
            "steps_per_sec": round(steps / wall, 2),
            "speedup": round(baseline / wall, 3),
            "efficiency": round(baseline / wall / world, 3),
            "comm_seconds": round(comm, 3),
            "compute_seconds": round(compute, 3),
            "comm_fraction": round(comm / (comm + compute), 3) if comm + compute else 0.0,
            "megabytes_sent": out[0]["megabytes_sent"],
            "all_reduce_calls": out[0]["all_reduce_calls"],
        })
    return rows


def table(rows, cols):
    head = "".join(n[:w].ljust(w + 1) if i == 0 else n[:w].rjust(w + 1)
                   for i, (n, w) in enumerate(cols))
    out = [head, "-" * len(head)]
    for r in rows:
        line = ""
        for i, (n, w) in enumerate(cols):
            v = r.get(n, "-")
            line += str(v).ljust(w + 1) if i == 0 else str(v).rjust(w + 1)
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="1,2,4,8")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--global-batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--bucket-mb", type=float, default=25.0)
    ap.add_argument("--collective-sizes", default="65536,1048576,8388608")
    ap.add_argument("--collective-iters", type=int, default=10)
    ap.add_argument("--skip-collectives", action="store_true")
    ap.add_argument("--sweep-batches", default=None,
                    help="comma separated global batch sizes, shows the comm/compute crossover")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    worlds = [int(w) for w in args.worlds.split(",")]
    print(f"{os.cpu_count()} cores, gloo over loopback, one BLAS thread per worker")
    print()

    result = {"args": vars(args), "cores": os.cpu_count()}

    if not args.skip_collectives:
        sizes = [int(s) for s in args.collective_sizes.split(",")]
        print("all-reduce microbenchmark (best of "
              f"{args.collective_iters}, slowest worker)")
        rows = collective_bench(worlds, sizes, args.collective_iters)
        print(table(rows, [("algo", 8), ("world", 7), ("megabytes", 11),
                           ("bytes_per_worker_mb", 21), ("seconds", 11),
                           ("effective_gbps", 16)]))
        result["collectives"] = rows

        print("\nring vs naive at the largest size:")
        big = max(r["megabytes"] for r in rows)
        for world in worlds:
            if world == 1:
                continue
            ring = next((r for r in rows if r["algo"] == "ring" and r["world"] == world
                         and r["megabytes"] == big), None)
            naive = next((r for r in rows if r["algo"] == "naive" and r["world"] == world
                          and r["megabytes"] == big), None)
            if ring and naive:
                print(f"  world={world}: ring {ring['seconds'] * 1000:7.2f} ms  "
                      f"naive {naive['seconds'] * 1000:7.2f} ms  "
                      f"({naive['seconds'] / ring['seconds']:.2f}x)")
        print()

    print(f"data parallel scaling: {args.layers} layers x {args.d_model}, "
          f"global batch {args.global_batch}, {args.steps} steps")
    rows = training_bench(worlds, args.d_model, args.layers, args.global_batch,
                          args.steps, "ring", args.bucket_mb)
    print(table(rows, [("world", 7), ("seconds", 10), ("steps_per_sec", 15),
                       ("speedup", 9), ("efficiency", 12), ("comm_seconds", 14),
                       ("comm_fraction", 15), ("all_reduce_calls", 18)]))
    result["training"] = rows

    print(f"\nmodel has {rows[0]['params']:,} parameters "
          f"({rows[0]['params'] * 4 / 1e6:.1f} MB of gradients per step)")
    for r in rows[1:]:
        lost = 1 - r["efficiency"]
        print(f"  world={r['world']}: {r['speedup']}x speedup, "
              f"{r['efficiency'] * 100:.0f}% efficiency, "
              f"{lost * 100:.0f}% lost, {r['comm_fraction'] * 100:.0f}% of step time is comms")

    if args.sweep_batches:
        # Batch size is the lever, not model size. Growing the model scales
        # gradient bytes and per worker compute together (both O(d^2)), so the
        # ratio barely moves. Growing the batch adds compute against an
        # unchanged gradient volume, which is what actually amortizes the
        # collective. Measured rather than assumed, because the first version
        # of this swept model size and the numbers said the opposite.
        print(f"\ncompute vs communication crossover (world={worlds[-1]}, "
              f"{args.layers} layers x {args.d_model}, varying global batch):")
        sweep = []
        for b in [int(x) for x in args.sweep_batches.split(",")]:
            r = training_bench([1, worlds[-1]], args.d_model, args.layers, b,
                               max(args.steps // 2, 5), "ring", args.bucket_mb)
            sweep.append(r[-1] | {"global_batch": b,
                                  "batch_per_worker": b // worlds[-1]})
        print(table(sweep, [("global_batch", 14), ("batch_per_worker", 18),
                            ("seconds", 10), ("speedup", 9), ("efficiency", 12),
                            ("comm_seconds", 14), ("compute_seconds", 17),
                            ("comm_fraction", 15)]))
        result["sweep"] = sweep

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
