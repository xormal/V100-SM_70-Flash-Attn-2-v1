# Track C: 5x KV compression at int8-comparable accuracy — measured, and refuted

**Goal.** Cut the per-token KV store 5x at accuracy comparable to the shipped e4m3 quantisation
(relative error 3.6e-4), so that decode stops being bound by store traffic. The plan came from a
prior mathematical design review: (Q1) partition keys into blocks, keep a
tiny per-block summary, and skip blocks a certified bound proves negligible; (Q2) store block-local
rank-`r` coordinates instead of full `d`; (Q3) compose the two.

**Result: dead on this model, by four independent measurements.** Not "hard" — the numbers below are
off by two orders of magnitude, and three of the four are ceilings that no better implementation can
move.

## How this was measured

Every number is from **this model's own stores**, not from a proxy: compressibility is a property of
the trained weights, so it cannot be argued from attention in general. The shim dumps real K/V
(fp16, pre-quantisation) from `reshape_and_cache_fp8` and real queries from
`flash_attn_prefill_paged`, both of which see every layer:

```bash
FA2SM70_DUMP_KV=/path/to/dir FA2SM70_DUMP_MAX=16384 ./start_model_fa2sm70_gpu23.sh
# then send a long prompt of REAL prose (not random tokens) and run the scripts here
python spectra.py     /path/to/dir     # Q2: spectral tails
python skip.py        /path/to/dir     # Q1: certified skip at s=512
python skip2.py       /path/to/dir     # Q1: certified skip vs block size, ball and ellipsoid bounds
python heur.py        /path/to/dir     # Q1 without the certificate: cheap score ranking
python oracle_rank.py /path/to/dir     # Q1 ceiling: rank by TRUE block mass (unimplementable)
```
Qwen3.6-27B-abliterated-AWQ, 16 full-attention layers, `d=256`, 16384 tokens/layer of real prose.

## Q2 — block-local low rank: dead

Residual energy after rank `r`, block `s=512`, centred (the most favourable setting in the plan):

| | r=16 | r=32 | r=64 | r=128 |
|---|---|---|---|---|
| K | 0.424 | **0.273** | 0.139 | 0.040 |
| V | 0.533 | **0.362** | 0.188 | 0.055 |

The plan needed `r=32` to reach 5.3x. At `r=32`, **27–36 % of the energy is outside the basis**. To
reach ~5 % residual you need `r≈128`, and at `r=128` the stored size is `r` coefficients plus the
basis amortised over the block (`d*r/s = 64`) = 192 vs 256 — **1.33x, at an accuracy worse than the
e4m3 quantisation it would replace**.

## Q1 — certified block skip: dead

Fraction of blocks a *certified* bound can prove negligible at `eps=1e-2`, vs the oracle (what is
actually there to skip):

| block `s` | oracle | ball bound | ellipsoid bound (top-32) |
|---|---|---|---|
| 64 | **0.951** | 0.026 | 0.018 |
| 128 | 0.902 | 0.012 | 0.014 |
| 256 | 0.809 | 0.000 | 0.003 |
| 512 | 0.631 | 0.000 | 0.000 |

ball: `<q,k_j> <= <q,mu_i> + ||q|| r_i`; ellipsoid adds the block's top-32 basis,
`<q,mu_i> + ||P^T q|| rho_i + ||q|| e_i`. The ellipsoid is **not** better — the residual term it must
carry costs more than the projected term saves, which is the same fact as Q2's 27 % residual.

## Q1 without the certificate — also dead, and this is the important one

Drop the guarantee and just rank blocks. Mean relative error of the head output, `s=64`:

| keep | read less | cheap score `<q,mu>+||q||r` | **oracle ranking (true mass)** |
|---|---|---|---|
| 50 % | 2.0x | 1.27e-1 | 1.73e-2 |
| 20 % | **5.0x** | 2.61e-1 | **7.18e-2** |
| 10 % | 10x | 3.53e-1 | 1.31e-1 |
| 5 % | 20x | 4.49e-1 | 2.09e-1 |

**At the 5x target the error floor is 7.2 % even with a perfect oracle** — 200x worse than the
3.6e-4 this had to preserve. So it is not a weak bound and not a weak heuristic: no block-selection
scheme of any kind reaches 5x here.

## Why the oracle skip fraction and the oracle error do not contradict

95 % of blocks each carry under 1 % of the mass, and skipping them still destroys the output. Those
are different statements: *individually* negligible is not *collectively* negligible. The plan's
error theorem bounds `||o - o~||` by the **total** skipped mass, and 800 blocks at 0.5 % each sum to
most of it. The mass here is genuinely spread, not concentrated — which is the same structural fact
the rank-32 residual reports.

## What this means for the throughput goal

"1000 tok/s at 256 K context" is not reachable on 2xV100 regardless of this work. One 256 K sequence
costs 8.4 GB of KV (32 KB/token over 16 layers x 4 kv-heads x 256 x 2 x 1 byte), and the pool is
~195 K tokens per 12 GiB/GPU, so at most ~2.9 such sequences fit at all. Their KV read alone is
8.4 GB per generated token, i.e. **~190 tok/s** against ~1.6 TB/s of aggregate bandwidth. Store
traffic does not amortise over a batch (each sequence's store is private), which is exactly why the
consult targeted compression — the arithmetic was right, only the compressibility was not there.

Measured aggregate decode on this server, short context, no compression (`batch_bench.py`):

| batch | 1 | 8 | 32 | 48 |
|---|---|---|---|---|
| aggregate decode tok/s | 42.9 | 227.1 | 430.1 | 497.1 |

So ~500 tok/s aggregate is reachable **today** at short context, because weights (unlike the KV
store) do amortise over a batch. The remaining lever for throughput is batch and the model's linear
path, not the KV format.
