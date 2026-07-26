# dist-train

Distributed training built from the bottom up: ring all-reduce written from
send/recv, data parallel on top of it, tensor and pipeline parallelism, and
scaling measurements that report where the efficiency went.

`dist.all_reduce` is one line, and it hides the only thing worth understanding
about distributed training — what it costs. So none of the collectives here are
imported.

```bash
pip install -r requirements.txt
pytest                          # 57 tests, all against torch.distributed as reference
python bench/scaling.py         # the tables below
```

## The cost model

For N workers and a tensor of S bytes:

| | bytes each worker sends | grows with N? |
|---|---|---|
| naive (gather to root, reduce, broadcast) | `(N-1) * S` | linearly |
| ring all-reduce | `2 * (N-1)/N * S` | approaches `2S` and stops |

That second row is the entire reason every real trainer uses a ring. Nothing is
a bottleneck because every link carries identical traffic. Measured, 33.55 MB
tensor:

| world | ring bytes/worker | naive bytes/worker | ring time | naive time | ring speedup |
|---|---|---|---|---|---|
| 2 | 33.6 MB | 33.6 MB | 69.5 ms | 72.4 ms | 1.04x |
| 4 | 50.3 MB | 100.7 MB | 167.2 ms | 213.8 ms | **1.28x** |
| 8 | 58.7 MB | 234.9 MB | 342.0 ms | 502.7 ms | **1.47x** |

Ring traffic goes 33.6 → 58.7 MB as workers go 2 → 8. Naive goes 33.6 → 234.9
MB. The gap widens with N exactly as the arithmetic says it should.

### Where the ring loses

At **small** tensors naive is faster, and that isn't a bug:

| world | 0.26 MB tensor: ring | naive |
|---|---|---|
| 2 | 1.70 ms | **1.00 ms** |
| 4 | 2.38 ms | **1.97 ms** |
| 8 | 9.72 ms | **4.29 ms** |

A ring is `2(N-1)` sequential hops — 14 round trips at world=8 — so it's
latency-bound when there aren't many bytes to amortize them over. Naive is 2
hops regardless. This is why NCCL switches algorithm by message size rather
than picking one; a ring is a bandwidth-optimal algorithm, not a fast one.

## Scaling

6 layers x 1024 (6.3M params, 25.2 MB of gradients per step), global batch
16384, 10-core machine, gloo over loopback:

| world | s/step | speedup | efficiency | comm fraction |
|---|---|---|---|---|
| 1 | 0.867 | 1.00x | 100% | 0% |
| 2 | 0.641 | 1.35x | 68% | 11% |
| 4 | 0.565 | **1.53x** | 38% | 28% |
| 8 | 0.621 | 1.40x | 18% | 44% |

Efficiency, not speedup, is the honest column. And note it goes *down* past 4
workers — on a 10-core box with 8 workers plus gloo threads, adding workers
starts costing more in communication than it returns in compute.

### The lever is batch size, not model size

Efficiency is governed by compute per byte of gradient. My first attempt swept
model size expecting efficiency to rise, and it fell instead — growing the model
scales gradient bytes *and* per-worker compute together, both `O(d²)`, so the
ratio barely moves. Growing the batch adds compute against an unchanged
gradient volume. That's the corrected experiment (world=8):

| global batch | per worker | speedup | efficiency | comm fraction |
|---|---|---|---|---|
| 256 | 32 | 0.09x | 1% | 95% |
| 1,024 | 128 | 0.22x | 3% | 88% |
| 4,096 | 512 | 0.66x | 8% | 71% |
| 16,384 | 2,048 | **1.40x** | 18% | **45%** |

**16x the speedup from the same code and the same collective.** Comm time is
essentially flat (1.22 → 1.48 s) while compute grows 30x (0.06 → 1.80 s).

At batch 256 the run is *slower than a single worker* — 0.09x. That number is
worth sitting with: naively parallelizing a small-batch job makes it eleven
times worse, and nothing in the code errors to tell you.

## What's implemented

### Ring all-reduce

Two phases, N-1 steps each. Reduce-scatter leaves chunk `i` fully summed on
exactly one worker; all-gather passes it around until everyone has it.

```python
for step in range(world - 1):
    send_idx = (rank - step) % world
    recv_idx = (rank - step - 1) % world
    reqs = [dist.isend(flat[s0:s1].contiguous(), send_to, group),
            dist.irecv(recv_buf, recv_from, group)]
    for r in reqs: r.wait()
    flat[r0:r1] += recv_buf         # accumulate in phase 1, overwrite in phase 2
```

Also `ring_all_gather`, `ring_reduce_scatter` (half an all-reduce — what
ZeRO/FSDP use so no worker ever holds the full gradient), and `ring_broadcast`.

### Data parallel

Every worker holds the model, each gets a different slice of the batch,
gradients are averaged so the replicas never diverge. Beyond looping
`all_reduce` over parameters:

- **bucketing** — a model has hundreds of parameter tensors, many tiny. One
  collective each means paying latency hundreds of times per step. Flattening
  into ~25 MB buckets turns that into a handful of large transfers. The tests
  check bucket size changes the *call count* and not a single gradient value.
- **replica divergence check** — `check_replicas_agree()` compares against
  rank 0, because replicas silently drifting apart is how you get a model
  nobody can reproduce.

### Tensor parallel

A linear layer cut into column blocks, then the next one cut by rows so its
input is already sharded. That pairing means an MLP block needs exactly **one**
all-reduce instead of two:

```
x --> ColumnParallel --> GELU --> RowParallel --> all_reduce --> y
      (no comm, output    (stays    (consumes the    (one collective
       is sharded)         sharded)  sharded input)   per block)
```

Bias on the row-parallel layer is added *after* the reduce, or it gets counted
N times. There's a test for exactly that.

### Pipeline parallel

Split by depth, with micro-batches to shrink the bubble:

```
bubble = (stages - 1) / (micro_batches + stages - 1)
```

4 stages at 1 micro-batch wastes 75% of the pipeline. At 16 micro-batches it
wastes 16%. Forward only — backward through a pipeline needs stashed
activations and a 1F1B schedule to bound memory, which isn't implemented.

## A real bug the tests caught

`ring_all_gather` sized every receive buffer like the local tensor. Splitting 33
outputs across 5 workers gives shards of 7,7,7,6,6 — so a 7-element send met a
6-element receive and the ring **deadlocked**. No error, no crash; the workers
just waited for each other forever, and the test suite hung until I killed it.

The fix passes each rank's shard length (known analytically, so no extra round
trip) and sizes buffers correctly. The test that found it:

```python
@pytest.mark.parametrize("world", [1, 2, 4, 5])
def test_column_parallel_shards_uneven_output(world):
    """33 outputs across 4 workers doesn't divide. Shards must still tile."""
    results = run_workers(_w_column_shapes, world)
    assert sum(local for local, _, _ in results) == 33
```

Worth noting the class of bug: every world size that divided evenly passed. A
suite that only tested 2, 4 and 8 would have shipped it.

A second one, in the harness rather than the algorithm: workers returned torch
tensors through a multiprocessing queue, which pickles a shared-memory handle
rather than copying. When the producer exits before the parent reads it, the
parent blocks forever. The launcher now converts tensors to numpy on the way
out.

## Tests

57 tests. Every collective is checked against `torch.distributed`'s own
implementation, at world sizes 1, 2, 3, 4, 5 and 8, on tensor sizes 1, 3, 7,
17, 1000 and 65537 — primes, sizes smaller than the world, and sizes that
divide evenly into nothing.

The ones that carry the most weight:

- N workers on batch/N produce the **same gradient** as one worker on the whole
  batch (for both ring and naive)
- replicas that start from *different* seeds end up bit-identical after 5 steps
- bucket size changes the collective count, never a value
- tensor parallel output matches an unsharded reference model built from the
  same seeds, to 1e-4
- row-parallel bias is added exactly once
- reduce-scatter slices tile the tensor exactly and each equals the
  corresponding slice of a full all-reduce
- ring traffic stays bounded as world size grows while naive grows linearly —
  the cost model, asserted as arithmetic

## Being straight about what this is

- **CPU and gloo over loopback**, ~0.5 GB/s against NVLink's ~300 GB/s. So
  communication dominates far more here than on real hardware and the
  efficiency numbers are pessimistic. The *shape* of every curve holds; the
  absolute numbers don't transfer.
- **8 workers on 10 cores** means the largest configuration is competing with
  itself for CPU, which is part of why efficiency drops past world=4.
- **No overlap of communication with backward.** Gradients become available
  back-to-front during backward, so firing each bucket's all-reduce as soon as
  its last gradient lands would hide most of the comms behind compute. Buckets
  are already built in reverse order for it; the hooks aren't wired up. This is
  the single biggest thing missing and it would move the efficiency column.
- **Pipeline parallel is forward only.**
- **No ZeRO/FSDP.** `ring_reduce_scatter` is the primitive they're built on,
  but optimizer state sharding isn't implemented.
- **`naive_all_reduce` is not a strawman** — it's the obvious correct
  implementation, it's tested against torch for correctness too, and at small
  sizes it wins.

## Layout

```
dt/collectives.py     ring all-reduce / all-gather / reduce-scatter / broadcast,
                      the naive baseline, and the byte-count cost model
dt/data_parallel.py   gradient bucketing, averaging, replica divergence check
dt/model_parallel.py  column/row parallel linear, pipeline stages, bubble maths
dt/launcher.py        process group setup, spawn helper, result marshalling
bench/scaling.py      collective microbenchmark + scaling + the batch sweep
tests/                57 tests
```
