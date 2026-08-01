# Roadmap

This document tracks the work required to evolve `fa2_sm70` from a V100-focused attention package
into a practical, API-compatible subset of FlashAttention-2. The project remains deliberately
specialized for Volta (`sm_70`): feature parity must not come at the cost of silently routing to a
slow fallback or overstating numerical guarantees.

## Scope and principles

- Preserve **exact FP16 attention** for the default prefill and decode paths.
- Treat an optimization as complete only with correctness coverage, a reproducible benchmark, and
  documented constraints.
- Keep the public API PyTorch-native and compatible with standard Q/K/V layouts where possible.
- Do not claim support for a feature that falls back to a materialized `Sq × Sk` score matrix.
- Retain third-party CUTLASS licensing and notices in `fa2_src/`.

## Status legend

| Status | Meaning |
|---|---|
| `done` | Implemented, covered by tests, and documented. |
| `next` | Highest-value planned work. |
| `planned` | Valid roadmap item, not yet scheduled. |
| `research` | Requires a feasibility/performance study before implementation. |
| `out of scope` | Deliberately not a target for this project. |

## Current baseline

`done` — The package currently provides:

- Exact prefill on `sm_70`: causal, non-causal, causal sliding-window, and two-sided local attention.
- Causal **work scheduling** on top of the cutlass mainloop: heavy-tiles-first block order and split-K
  over the key range, both exact and both with off-switches (`FA2SM70_FWD_REV`, `FA2SM70_FWD_SPLITS`).
- FP16 input plus BF16-to-FP16 conversion; zero-padding for non-aligned head dimensions.
- MHA/GQA/MQA inputs; prefill maps query heads to shared K/V heads natively without expansion.
- Exact split-KV decode for `Sq = 1`, including a grouped-head path for FP16 deferred decode.
- LSE output, INT8/INT4 quantized KV-cache decode, and MLA-oriented helpers.
- Head dimensions through 1024 for prefill and through 512 for decode/backward/varlen.

## P0 — Training parity

### Backward pass

`done` (exact `dQ`/`dK`/`dV`; head_dim <= 512, incl. Gemma-4 d=256/512)

`fa2_sm70.flash_attn_func(Q, K, V, causal, scale)` is a differentiable exact attention (cutlass FMHA
forward + cutlass FMHA **backward**, `AttentionBackwardKernel<Sm70, half_t, ...>`). Implemented via a
`torch.autograd.Function` (`_AttnFn`): forward saves `(Q, K, V, O, LSE)`; backward computes
`delta = (dO*O).sum(-1)` and launches the backward kernel (`attn_bwd_cutlass.cu`). No score-matrix
materialization (flash all the way).

**Done — evidence**

- FP32-reference gradient tests (`tests/test_backward.py`): `relL2 ~3-4e-4` (= fp16 round-off, not
  algorithmic error) for `dQ`, `dK`, `dV` across MHA / GQA / MQA, causal + non-causal, `Sq != Sk`
  (`CausalFromBottomRight`, matching the forward), and padded non-multiple-of-8 head dims. Forward
  path unchanged (`test_attention` / `test_decode` ALL OK). `gradcheck` is not used: the kernel is
  fp16-only, so finite differences are dominated by round-off; the fp32 analytic-gradient oracle is
  the stronger correctness gate.
- LSE reuse: the forward's natural-log LSE feeds backward directly (padded to `strideH % 8 == 0`).
- BSHD layout contract (see `attn_bwd_cutlass.cu`): the cutlass backward hard-codes `O`/`dQ`/`dK`/`dV`
  token-stride = `num_heads*head_dim`, so those are packed BSHD; `Q`/`K`/`V`/`dO` use configurable
  strides. GQA/MQA is expand-K/V-then-reduce-`dK`/`dV` (the backward kernel is one-KV-per-query-head).
- Perf/roofline: `128x64` tiles (`kBlockSizeI=128 >= kMaxK` => `kOutputInRF=true`, so `dK`/`dV` stay
  register-resident, no gmem gradKV workspace; the kernel also self-computes delta) + split-K
  occupancy fill give ~1.5-2x over the naive `ns=1`/`64x64` launch. The d128 multi-batch path now also
  carries a hand-written interleaved **QK‖GradK** mainloop (`fused_qk_gradk.h`, gated `kFuseQKGradK`,
  runtime `head_dim==kBlockSizeI`): **+4%** vs baseline (median-50, exact, 16/16).
- **Register-resident backward (`d≤128`) is competitive with or faster than SDPA-efficient**: measured
  `d=64` isolated speedup is 1.04–1.19×, `d=128` is 1.16–1.31×, and fwd+bwd is 1.05–1.26×. The win is the
  **fragment-order coalesced `dQ` fold** (`FMHA_BWD_COAL_FO`, default;
  off-switch `FA2SM70_BWD_FO=0`): the store-side split-`dQ` reduction emits in the HMMA accumulator's own
  lane/element order (`addr = warp·kElem·32 + idx·32 + lane`) — contiguous by construction, 4 memory-transaction
  granules per 32-writer group instead of the scattered 16 (a 4× cut in `dQ`-fold L2 write traffic on every
  tile/split; `dQ`-port stall 2.14→0.65 cy/inst) — then a fused bespoke convert kernel (`fo_convert_kernel`)
  transposes fragment-order → `dQ[query,d]` in one coalesced pass. This is what turned the prior `~0.73-0.80×`
  loss into a structural win; the fold's L2 traffic is the one axis where we beat torch's *identical* cutlass
  kernel (which still scatters). A smem-transpose (+12% net-neg) and an in-mainloop warp shuffle both lost to it.
  The R7/fold fast path needs **full 128-query blocks**: a non-128-multiple `Sq` at `d≤128` (and all varlen bwd)
  routes to the exact direct-`dQ` (lock-RMW) path — correct on any `Sq`, ~parity not the fold speedup. Guarded by
  partial-`Sq` cases in `tests/test_backward.py` (the earlier size-64 partials were strip-aligned and missed it).
- **Backward optimization status**: the interleaved QK‖GradK path is retained for its measured d128 multi-batch
  gain (`+4%`); the fragment-order fold rides on top of `FMHA_BWD_R7` (which removed the 255-reg split-`dQ`
  spill). The supported performance claim is the benchmark table in README.
- **d=64 fragment-order fold (`done`)**: extended the coalesced `dQ` fold to `d=64` (was `d=128`-only). The R7 tile
  is 128-wide even at `d=64`, so the fragorder workspace is `Sq·kBI` (not `Sq·d`) with a `kBI`-wide pointer advance
  and a two-stride convert (`out_span=kBI·kMaxK`, `ws_span=kBI²`); a forward perm map avoids the `d<kBI` zero-hole.
  **d=64 isolated backward now beats torch 1.04–1.19×** (was ~parity; fold is ~28% of the d64 wall).
- **d≥256 large-head backward — `done` (beats torch).** `d=256` causal now runs a hand-written block
  mainloop (`fa2_sm70/csrc/volta_bwd_block.cuh`) instead of the cutlass one, and two glue defects that
  dominated the wall time were removed. **Isolated backward vs SDPA-eff: `d=256` 1.27–1.73×, `d=512`
  1.05–1.10×**; reproduce with `python benchmarks/bench_bwd_large.py` (A/B off-switch
  `FA2SM70_BLOCK_BWD=0`, which drops `d=256` back to 1.01–1.16×).
  What actually paid, in the order the measurements found it:
  1. **Workspace zeroing was 27% of the backward.** 352 MB zeroed per call, of which the `gk`/`gv`
     parts (336 MB) never needed it — the kernel STORES them (`gmem_tile.store`, first entry flagged by
     `isFirstQuery`). Only the `gq` tail needs zeros, because `GradQTempStorage` interleaves `lock` and
     `counter` with the payload. Zeroing just that tail flipped `d=256` from 0.81× to 1.08×.
  2. **`r7_dqf` was allocated by a build flag, not by geometry** — a 16.8 MB fp32 buffer at `d=256`,
     where the R7 path (`kMaxK <= kBI`) cannot run at all.
  3. **The block mainloop itself** (playbook §14–§16): 8 warps at a 32×16 warp tile, K/V resident,
     persistent CTAs on an atomic ticket, and a vectorised global→shared refill. That refill alone cut
     integer instructions 6665M → 2460M, *below* cutlass's 2629M: the scalar copy, not the math, was
     the cost. Tensor-instruction count is identical to cutlass's (89,128,960) — we were simply
     executing 1.74× more instructions around it.
  Measured-dead on the way, do not re-try without a new reason: fragment-iterator addressing (nvcc
  already hoists it), `+8` row padding and the XOR swizzle (both lose to register pressure), fp16
  accumulation (`2e-2` error against the current `3e-4`), a second CTA per SM (the register file is
  already fully committed: 16 warps × 128 reg = 65536).
  **Still open:** `d=512` keeps the cutlass path — resident K/V would want 111 KB of shared memory
  against the 96 KB limit; it needs either streamed K/V (measured −30% in the prototype) or a smaller
  key tile. Non-causal, ragged and `Sq != Sk` shapes also stay on the cutlass path.
- `dropout_p=0` is a bit-exact no-op vs the no-dropout path.
- Same seed reproduces the output and `dK`/`dV` BIT-IDENTICALLY (proves the backward replays the
  forward mask exactly -- a different mask would move grads by O(1)); `dQ` matches up to split-K
  atomic-reduction noise (~3e-5, inherent to the parallel-key backward, not the mask).
- The `1/(1-p)` scaling is unbiased: `mean_masks(O_drop) -> O_nodrop` with a residual that is pure
  `1/sqrt(N)` Monte-Carlo noise (verified `relL2*sqrt(N)` is constant).
- Head_dim <= 128 (the differentiable training path), MHA/GQA/MQA, causal/non-causal.

## P1 — Batch efficiency and serving integration

### Variable-length / packed attention

`done` (forward + differentiable forward+backward, head<=512)

`attention_varlen(Q, K, V, cu_seqlens_q, cu_seqlens_k, max_q, max_k, causal, scale)` (inference) and
`flash_attn_varlen_func(...)` (differentiable). Token-packed `Q [Tq,H,D]`, `K/V [Tk,Hkv,D]`, int32
`cu_seqlens [nseq+1]`; each sequence attends its own Q/K ranges with per-sequence bottom-right causal.
Uses the cutlass FMHA `seqstart_q/k` (forward) and `cu_seqlens_q/k` (backward) native paths -- no
padding materialized.

**Done — evidence**

- Independent `Sq != Sk` per sequence, GQA/MQA, causal + non-causal: forward `relL2 ~2.5-3e-4` vs a
  per-sequence fp32 reference; backward `dQ/dK/dV relL2 ~3-4e-4` (`tests/test_varlen.py`). LSE is
  `[nseq, H, lse_dim]` and is shared forward->backward. delta is packed `[H, Tq]` (kernel/torch agree).
  Split-K occupancy fill works under `cu_seqlens`.
- Forward and the differentiable backward both support head_dim <= 512 (verified d256 ragged varlen
  bwd, `relL2 ~3-4e-4`), consistent with the fixed-length backward (Gemma-4 d=256/512).

### Stateful KV cache

`done` (fp16 `KVCache` + int8/int4 `QuantizedKVCache`)

`fa2_sm70.KVCache(B, Hkv, max_len, D, device)`: `.append(k, v)` writes new tokens (O(n), not O(length))
and maintains the deferred-U running-max scan incrementally; `.decode(q, scale)` attends over the valid
prefix. Native MQA/GQA (no per-step K/V expansion). Cache-update (append) is cleanly separated from
attention (decode).

**Done — evidence**

- Cached decode is BIT-EXACT (relL2 0.0) vs a direct full decode over the same prefix, and the
  incremental `k_absmax` exactly matches a full rescan (err 0.0), across MQA/GQA/MHA.
- ZERO-COPY: `.decode` passes the preallocated buffer to the kernel with a `kv_len` param (the
  deferred kernels loop the valid prefix but index with the physical `max_len` stride) -- no per-step
  copy. Measured 1.4-3.7x faster than copy-then-decode (3.7x for GQA/large caches). `test_decode` ALL OK.
- `QuantizedKVCache(B, Hkv, max_len, D, device, bits=8|4)`: keeps the QUANTIZED bytes across steps
  (~2x int8 / ~4x int4 less cache memory + HBM read). Per-token symmetric quantization (the 2-level
  per-channel scale is a global-over-tokens statistic, not incremental). `.append` is O(n) + an
  incremental deferred-U scan over dequantized keys; `.decode` is zero-copy over the valid prefix.
  BIT-EXACT (0.0) vs a one-shot `quantize_kv(per_channel=False)` + `decode_quantized`; int8 ~0.7% vs
  fp16 (`tests/test_quant_cache.py`).

### Paged KV cache

`done` (block-table layout + per-sequence variable lengths)

`decode_paged(Q, Kcache, Vcache, block_table, block_size, kv_len)`: K/V live in a block pool
`[num_blocks, block_size, Hkv, D]`; `block_table [B, max_blocks]` (int32) maps each sequence's logical
block to a physical block, so physically non-contiguous blocks are attended in logical order. Added a
`PAGED` template to the two deferred decode kernels (`split_defer_kernel`, `split_defer_mqa_kernel`)
via a `kvb_off<PAGED>` address helper; paged launchers instantiate fp16 (KVB=16) only.

**Done — evidence** (`tests/test_paged.py`)

- Logical order across **shuffled (non-contiguous)** physical blocks is BIT-EXACT vs the contiguous
  decode over the same gathered KV (`vs_decode = 0.0`), and ~2e-7 vs an fp32 softmax oracle.
- Partial final blocks (`kv_len` not a multiple of `block_size`) covered; GQA (fused GF>1), MHA
  (GF<=1), MQA, `V==K`, and all head dims {64,128,256,512}. Non-paged decode unchanged.
- The deferred-U bound is gathered through the table (a serving engine keeps it incrementally like
  `KVCache`). Per-sequence variable lengths supported via an optional `seqlens [B]` (int32): each
  sequence attends its own length, splits past it are empty (contribute 0 to the associative sum);
  ragged `seqlens=[37,128,5,100]` matches fp32 ~2e-7. `kv_len` sizes the split grid (= max length).

### Native grouped-query prefill

`done`

Prefill maps each query head to its KV head via `kv_head_ratio = H/Hkv` (a Params field consumed in
`advance_to_block`: `key_ptr += (head_id / kv_head_ratio) * k_strideH`). Covers MHA (ratio 1), GQA
(1 < Hkv < H), and MQA (Hkv = 1) with no `repeat_interleave` and no repeated K/V materialization.

**Done — evidence**

- Exact vs the expanded reference: `relL2 ≈ 2.6e-4` across g = 1/2/4/16, causal + windowed, and
  non-power-of-two head counts (14/7, 12/3); `tests/test_attention.py` PREFILL ALL OK; Gemma-4 E2E 40/40.
- K/V stay `[B, Hkv, Sk, d]` (no `[B, H, Sk, d]` copy) → lower peak memory for `Hkv < H`.

## P2 — Attention modifiers and API compatibility

### General local-attention window

`done` (causal past-window + two-sided bidirectional, all head dims)

`attention(..., window=W)` = causal past-window (keys `[i-W+1, i]`); `attention(..., window_right=R,
causal=False)` makes it TWO-SIDED (query `i` attends `[i-(W-1), i+R]`). Params `window_left`/
`window_right` (-1 = unbounded on that side); the mask does both edges (+ causal) and runs for a
non-causal window too; the block loop skips leading AND trailing out-of-window blocks. The
flash-attn-compat layer accepts `window_size=(left, right>0)`.

**Root cause of the old "d=256 NaN" (it was un-normalized WRONG values, not NaN):** the
output-accumulator path (`d > 128`) flags the last block for the normalizing epilogue via
`iter_key_start + kKeysPerBlock >= num_keys`, but `window_right` shrinks the loop to `iter_key_end <
num_keys`, so the last processed block was never "last" -> the divide-by-sum epilogue never fired ->
un-normalized output (`relL2 ~1-256`). The RF path (`d <= 128`) normalizes once after the loop, hence
the d=256-specific symptom. Fix: `kIsLast` uses `iter_key_end`.

**Done — evidence** (`tests/test_window.py`, in the 19-suite matrix): two-sided (symmetric / asymmetric /
edge `wl=0,wr=0`) and causal past-window (incl. `Sq != Sk`), GQA/MQA, head dims 64/128/256/512, all
`relL2 ~2.9e-4`, no NaN.

### ALiBi and soft-capping

`done` (prefill): both soft-capping and per-head ALiBi.

- **Soft-capping**: `attention(..., softcap=cap)` applies `cap*tanh(scale*logit/cap)` per element in the
  kernel's log2 domain, preserving `-inf` masks; relL2 ~2.6e-4, `softcap=0` unchanged.
- **ALiBi**: `attention(..., alibi_slopes=[H])` adds `slope_h*(key_pos - query_pos)` inline in the
  forward softmax loop (an unconditional `iterateRows`, no materialized `[Sq,Sk]` bias); relL2 ~2.5e-4
  vs an fp32 `slope*(key-query)` reference across MHA/GQA/MQA, causal/non-causal, `Sq != Sk`;
  `alibi_slopes=None` is a bit-exact no-op (`tests/test_alibi.py`). Composes with causal/windowed.

**Done — evidence**: both match an FP32 reference alone and with causal/windowed attention; each defaults
off (bit-exact no-op) so unsupported combinations never silently change results.

**Decode path** (`done`): `decode(..., softcap=cap, alibi_slopes=[H])` applies both at decode. They ride
the ONLINE split-KV path (running max) -- the deferred path's Cauchy-Schwarz U-bound does not bound
softcapped/ALiBi-biased scores (exp underflow risk), so `decode()` routes modifiers there and absorbs
any bounded bias exactly. Verified `relL2 ~2e-7` vs fp32 (softcap / ALiBi / both), MHA/GQA/MQA, all head
dims, no NaN; modifiers-off is a bit-exact no-op (`tests/test_decode_mod.py`). int8/int4 + modifiers
raises (fp16 path only).

### FlashAttention-compatible convenience APIs

`done`

`fa2_sm70.flash_attn_interface`: `flash_attn_func`, `flash_attn_qkvpacked_func`,
`flash_attn_kvpacked_func`, `flash_attn_varlen_func` in the flash-attn `[B, S, H, D]` layout (seq
before heads), transposing onto our `[B, H, S, D]` kernels. Forwards `window_size=(left,0)` (causal
past-window), `softcap`, `alibi_slopes`, and `dropout_p`; preserves autograd on the plain causal path.
Unsupported combos (two-sided/non-causal window, varlen dropout) raise `NotImplementedError` rather
than silently emulate. Verified vs an fp32 `[B,S,H,D]` reference (`tests/test_fa_interface.py`,
relL2 ~2.5e-4) for all forms; plain-path grads present and finite.

## P3 — Reliability and release engineering

### Test matrix and CI

`done`

`tests/run_all.py` runs all 20 correctness suites (MXFP4, attention, decode, backward, varlen, dropout,
  paged, quant-cache, decode modifiers, ALiBi, window, fa-interface, HMMA map, d=256 forward, GPTQ
  dequant, W8A16 GEMV, vLLM int8 gather/pool, byte-KV prefill, direct e4m3 prefill) as subprocesses and prints a pass/fail matrix (non-zero exit on failure) for the
`sm_70` GPU runner. `tests/test_smoke_cpu.py` is a no-GPU import + API-surface + signature check.
`.github/workflows/ci.yml` is two-tier: a CPU tier (`compileall` syntax + smoke) runs anywhere, the
GPU tier (full matrix) on a self-hosted `sm70` runner. Verified: CPU smoke ALL OK, GPU matrix 20/20 PASS.

### Packaging and build reproducibility

`done`

`fa2_sm70/_ext.py` runs a one-time preflight (CUDA build of torch + visible device, device arch vs
sm_70, `nvcc` presence/version, host-compiler-vs-nvcc gcc compatibility, bundled cutlass headers) and
wraps every JIT `load()` so a build failure surfaces an ACTIONABLE hint (too-new gcc, missing nvcc,
no sm_70, read-only build dir) plus the known-good recipe -- not a raw nvcc dump. README documents the
pinned/tested matrix (sm_70, CUDA 12.x nvcc, gcc ≤ 14, torch 2.5-2.7 +cu12x) and the three extensions.
Verified: preflight is clean on the reference env, does not fire on `import`, and the full GPU matrix
still passes (20/20) after the loader refactor.

### Benchmark protocol

`done`

`benchmarks/_common.py` enforces the protocol: 20 warm-up iters, single sync + `it` timed iters + final
sync, inputs allocated once outside the timed closure. `attn_flops()` + `roofline_pct()` give FLOP
accounting (% of the 125-TFLOPS V100 roofline; forward = `4·B·H·Sq·Sk·d`, backward = 2.5x), `gpu_state()`
reports SM clock / power / temp (clocks throttle -- a latency number needs them). `benchmarks/README.md`
documents the method, the kernel-only-vs-end-to-end split, and the SDPA (default + efficient) baselines.

## Research backlog

| Item | Status | Decision criterion |
|---|---|---|
| Match/beat torch's backward at `d≤128` | `done` | BHSD-direct layout handling plus the fragment-order `dQ` fold produce measured isolated speedups of 1.04–1.19× at `d=64` and 1.16–1.31× at `d=128`. Full 128-query tiles use the fast fold; partial and varlen shapes retain the exact direct-`dQ` path. |
| Large-head `d≥256` backward | `done` (`d=256`), `next` (`d=512`) | `d=256` causal runs the hand-written block mainloop (`fa2_sm70/csrc/volta_bwd_block.cuh`): **1.27–1.73× vs SDPA-eff**, `relL2 4.4e-04` against an fp32 reference, `tests/test_backward.py` green, reproduced by `benchmarks/bench_bwd_large.py` with an `FA2SM70_BLOCK_BWD=0` A/B. Two glue defects removed on the way carried most of the win: the split workspace was zeroed in full (352 MB, 27% of the backward) although only the `gq` tail needs it, and `r7_dqf` was allocated by a build flag rather than by geometry. `d=512` still routes to cutlass at **1.05–1.10×** — resident K/V need 111 KB of shared memory against the 96 KB limit. |
| Prefill scheduling under a causal mask | `done` | **The vendor kernel schedules causal work naively; we do not.** Under a causal mask query tile `i` scans `ceil(((i+1)*kQ+off)/kK)` key tiles, so weights run `1..16` at `S=2048`. Two exact fixes, both shipped: (a) the dispatcher hands blocks out by ascending `blockIdx`, putting the HEAVIEST tiles in the ragged tail wave -- reversing the mapping is longest-processing-time-first and gives **+19.6% at S=4096, +29.8% at S=8192** on the isolated kernel; (b) a schedule cannot be shorter than its longest task, and at `B*H=1, S=2048` the heaviest tile is 16 against an ideal of 544/240 = 2.27 -- a **7x imbalance no ordering can fix**, measured as 130 us/head against 31 us/head at `H=32`. Split-K over the causally-trimmed key range (fp32 partials, LSE-weighted combine, exact by §2.2's identity) takes the under-filled shapes to **1.69-3.26x vs SDPA-eff**; `1,1,2048,64` was a **0.96x loss** before it. The planner declines to split when `blocks > 0.9*slots` AND `avg tile weight < 16` -- that threshold is from measurement, not model: two shapes with the same block count and near-equal imbalance went +37% and -4%, separated only by average tile weight (32.5 vs 8.5), because the combine costs `(s+1)*B*H*Sq*d*4` bytes regardless of tile weight. Reproduced by `benchmarks/bench_prefill.py`, which measures the `no-split` column in a child process. |
| Off-chip-latency prefetch | `research` (not part of release) | The current backward is bandwidth-latency limited at low occupancy. A separate prefetch pipeline remains experimental and is deliberately excluded from the supported kernel. |
| Head dimension above 512 | `done` (prefill, `D <= 1024`) | The cutlass output-accumulator config compiles on `sm_70` for `kMaxK` 768/1024 (smem ~33 KB << 96 KB) -- the earlier "fails to compile" note was wrong. Wired `attention` to `D <= 1024`; verified d640/768/1024 causal+non-causal `relL2 ~2.9e-4` vs fp32 (`tests/test_attention.py`). There is no torch-SDPA fallback on Volta for `D > 256` anyway, so this is the only path. Backward/decode `> 512` is a follow-up (niche). |
| Moment-form decode | `research` (accuracy frontier characterized) | The O(1)-per-query moment form (README §2.5) needs `M ~= 1.086*R` moments for score range `R`; viable only for bounded-score heads (`R <= ~1.3` at M=2). Real LLM attention after scaling generally exceeds this (Gemma-4 global measured `B ~= 49`, i.e. large `R`), so the target workload does not benefit -- it stays a documented prototype, not a kernel. |
| Lower-bit KV formats beyond INT4 | `out of scope` (quality rejected) | Studied: on outlier-channel data, per-token+per-channel `relL2` is int4 ~0.15, **int3 ~0.28, int2 ~0.84** -- unacceptable, and the byte saving is marginal (int3 = 0.19x vs int4's 0.25x) with non-byte-aligned packing. int4 (per-channel + top-1 control point -> <1%) is the practical floor on V100. |
| Paged decode: emulated fp8 dequant | `done` (`fd5a84e`) | **Half the long-context ~10%, taken.** The kernel's #1 cost was not the reduction (shuffle latency lands in `short_scoreboard`, measured 0.07) but `e4m3_to_float`: Volta has no fp8 hardware, so it picked the byte apart by hand with a denormal branch, ~12 ops on the load→convert→FMA chain — 16 conversions/lane/token × 262144 token-visits × 12 = 50.3M of the measured 58.9M ALU instructions. Replaced by a branch-free bit trick (fp16 is E5M10/bias-15, e4m3 is E4M3/bias-7: drop `EEEEMMM` at fp16 bit 7, multiply by 2⁸; denormals fall out exactly because an fp16 denormal `0.MMM·2⁻¹⁴` × 2⁸ *is* `0.MMM·2⁻⁶`). **1.50–1.81× (122/388/750/1344 → 81/223/414/795 µs at ctx 8K/32K/64K/128K), KV bandwidth 104 → 169 GB/s**, ncu ALU 58.9M → 37.1M and DRAM 15.1% → 23.6%. It also fixed a silent numerical bug: relL2 vs SDPA over the dequantised pool 2.6e-3 → **2.6e-7** (the old value was flat across every shape — a systematic conversion error, not accumulation noise). |
| Paged decode occupancy (the rest of the ~10%) | `next` | **Still open after `fd5a84e`.** `ncu` now reads DRAM **23.6%** of peak, SM 39.9%, warps active **17.2%** — the kernel moved toward memory-bound but occupancy is untouched, and that is the remaining lever: 166 registers/thread allow only 3 blocks/SM (12 warps of 64). Below 128 registers it would be 4 blocks. Historic ncu figures before the dequant fix, for reference: DRAM **15.1%**, SM 37.3%, warps active **18.1%**. The kernel sustains 104–137 GB/s against **820–840 GB/s** measured achievable on this V100 (`copy_` fp32 819, `sum` fp32 841) — it is latency-bound at low occupancy, *not* bandwidth-bound. Block size is not the cause (74 vs 76 us at bs=1568 vs 128). Two levers: (1) occupancy — more splits already give −13% at 65K and −22% at 128K (ns=64→256), but the split count is static under a captured CUDA graph and ns=256 costs **+30%** at 4K, so it needs a length-aware choice at capture time, not a constant; (2) instructions per byte — each token pays a 5-step cross-lane `__shfl_xor` reduction times GF=6 fused heads, which is what fills the time rather than the KV read; giving each lane more of `d` (larger EPT, fewer lanes per token, several tokens per warp) cuts the reduction depth. Worth ~10% of the whole decode step at long context: at ctx=65K our 16 full-attention layers are 11.8 ms of an 86.2 ms step (13.7%). Gate any claim on end-to-end server tok/s, not kernel microseconds. |
| Decode on the QUANTISED shapes (int4 / int8 / GPTQ) | `next` | **The same check that paid twice, not yet run on these.** The e4m3 pair is done (read `fd5a84e` 1.50-1.81x, write `5ab11a9` 1.12x and now bit-exact vs torch), and both were found by one metric: `smsp__inst_executed_pipe_alu` against the memory instructions, with ALU/LSU > ~3 marking a candidate. Still unmeasured: (1) `load_raw` KVB==4 -- int4 unpacks nibbles and subtracts the zero point, ~3 ALU/element plus the per-token scale multiply; (2) KVB==8 -- the conversion is a single I2F, but the per-token scale lookups may not be; (3) `gptq_w8_dequant_kernel` -- measured at **545 GB/s against the ~820 GB/s this card gives on a plain read**, i.e. ~1.5x of headroom, and it runs on EVERY linear layer of the int8 model, so it is the largest single target left on the serving path; (4) the fp16-pool decode path and `reduce_defer_out`. NB Volta has no hardware for any of these formats, which is exactly why emulation keeps landing on the critical path. |
| Async scheduling vs prefix caching (vLLM-side) | `research` | vLLM logs `Async scheduling is not compatible with prefix caching for Mamba models` → `Asynchronous scheduling is disabled` for this hybrid model (48 linear + 16 full attention). Async overlaps the scheduler with GPU execution and typically pays a few % at small batch; prefix caching measured a 79% hit rate on real traffic. Genuinely uncertain trade — decide by A/B on the live server (one restart each), not by argument. |
| Decode headroom at SHORT context | `out of scope` (near the hardware floor) | Measured: the decode step is 28.6 ms/token at ctx=4K while reading 17.7 GB of int8 weights per GPU per token, i.e. 21.6 ms at the 820 GB/s achievable — **75% of the weight-bandwidth roofline**. Total headroom in the entire step is therefore ~24%, of which our attention is only 1.18 ms (4.1%). Chasing 10% here means fighting the scheduler, sampling, all-reduce and the 48 linear-attention layers, not attention. |
| Sparse or arbitrary masks | `out of scope` | Dense arbitrary masks conflict with the project’s flash-memory objective; use a specialized kernel. |
| Byte-KV prefill format | `done` (format 8 shipped) | Two moves, both at the QUANTISER and both bit-neutral (`torch.equal` vs the shipped int8 path on every shape): a **biased byte** (`u = v+128`, so the sign XOR leaves the kernel) and an **interleaved scale table** (`(k_scale, v_scale)` adjacent per key position, so one `.64` load brings both from one address). Measured **1.061–1.064×** over the shipped int8 path at `Sk` 8K/32K/128K, and parity-to-1.02× against the e5m2 route **while carrying int8 accuracy** (relL2 to fp32 7.0e-3 vs e5m2's 5.4e-2, i.e. 7.8× better). Public API `quantize_kv_prefill` + `prefill_bytes`; `tests/test_prefill_bytes.py`. The table never cost volume (0.8% of traffic) — it cost a 64-bit address recomputation per access: +14 `LDG` but +36 `IMAD` and +11 `IADD3` in the SASS transcript. |
| Issue-slot accounting: where it stops predicting | `done` (doctrine corrected) | We had written "the byte kernel is past the issue wall (ν=1.71 > 1.00), every extra instruction costs a cycle". A direct falsifier (an idle `LOP3`, zero mask from constant memory, bit-identical answer) prices a slot at **0.22 cycles on computing warps and 0.30 on hauling warps** — the ROLE ratio from the model survives, the ABSOLUTE cost does not. And an idle instruction turns out to give the LOWER BOUND, not the price: the same +128 issues placed on the operand-expansion chain (the real sign XOR — formats 0 and 6 differ by exactly it) cost **+4.24% instead of +1.93%**, i.e. **0.49 cycles, 2.2x more for an identical instruction count**. Both reproduced independently. The law is therefore about PLACEMENT, not about the kernel: calibrate with the instruction you intend to add, at the position you intend to add it. The kernel runs at 39.7% of tensor peak against a 65.2% slot ceiling, i.e. **61% of its own counting ceiling**. Phase falsifiers (one phase removed per build; only the time is read) name the real limiter: 1st GEMM **35.1%**, 2nd GEMM **19.7%**, softmax 15.2%, hauling **5.7%**, rendezvous 2.0%. Both GEMMs carry identical tensor work (256 HMMA/warp/tile) and identical aux issues (208) yet differ by 795 cycles/tile — exactly their shared-memory traffic difference (768 predicted). Four edits that cut the instruction count did **not** cut the time; one (16-byte hauling transfers, −35% of the role's issues) made it **4.8% worse**. Consequence for anyone reusing this: **price one slot with a falsifier before optimising for slots.** `docs/VOLTA_SM70.md` §3b. |
| Direct `e4m3` prefill kernel | `done` (`nosc=11`, `tests/test_fwd_e4m3.py`) | `fp8_e4m3` is the format vLLM's KV pool actually uses, so a fast direct path removes the private-int8-format mine entirely (a private format is legal only while every reader is ours — on Gemma-4, 40 of 48 layers would read our int8 as e4m3). The blocker on record is the bit move: the sign travels 8 positions while the other seven bits travel 7, which a byte permute cannot express, measured at 1.698× the no-expansion baseline. **But that number was taken on a dependent chain**, and the row above says a dependent-chain microbenchmark is the wrong lens for in-kernel cost — the same 1–3 extra instructions may land in idle slots. **Both halves of the old verdict were wrong, and measurement said so.** (a) The expansion is **6 instructions per 4 bytes**, not the recorded 8 minimum: `prmt`'s sign-replication mode (the high bit of a selector nibble replicates that byte's sign) lets `[b, sign(b)]` in adjacent bytes plus a shift-left-7 land the data in bits 13..7 and the sign in bit 15, leaving one mask; `__byte_perm` does not expose that mode, it needs inline PTX. The x256 rebias folds rank-1 and is exact. (b) In the kernel it costs **1.162x e5m2**, against the 1.698x the isolated dependent chain predicted — a 46% overprice. Net: **1.10x faster than the shipped int8 path**, relL2 to fp32 2.68e-2 (2x better than e5m2, 3.8x worse than int8+scale), bitwise against the frozen e5m2 reference on data both formats hold exactly, expansion exact on all 254 finite byte values. Independently re-measured on card 0 (0.868/0.862/0.862 vs e5m2 at Sk 8K/32K/128K). Keep int8 as the default where every reader is ours; e4m3 is the route for a foreign pool. **The 6-instruction expansion is a PROVEN MINIMUM, not a best effort**: posed as algebra over Z/2^32 with five permitted operations, the answer is that 6 is attained and 5 does not exist — a residue-mod-8 invariant (neither a byte permutation nor a bitwise boolean moves a bit out of its residue class, so a transport is mandatory in every output word's ancestry), a window-capacity lemma (one transport serves only one byte-parity pair; all four need 33 bits of span), and three position classes against lop3's two. The rank-1 bypass is also closed: the free corrections force an affine substitute psi(x)=(x-C)/alpha, alpha IS the constant already spent on the bias difference, and C != 0 changes accuracy but not the transport structure. The numeric trace was reproduced independently on all 254 finite byte values (F/E = 2^-8 exactly). |
| Bigger warp tile (`MB=2`, `BQ=64`) at `d=256` | `research` (blocked on shared memory) | The only lever the phase decomposition leaves: it halves the Q traffic per unit of work (the `mma.884` fragment map hands each operand value to TWO lanes, so 128 useful bytes cost 2 cycles) and takes ν to ≈1.07. It does not fit: sQ 33 280 + sP 8 704 + double-buffered K/V 69 632 = **111 616 B against the 98 304 limit**. Freeing 13 KB needs single-buffered K feeding with a third rendezvous — a separate build, and no longer about instruction counting. `SHFL` is not an alternative: on Volta it runs on the same MIO at the same throughput, so it is a one-for-one swap. |
| A first-class vLLM backend (`FA2_SM70`) instead of the module substitution | `done` (steps 1-3) | The old integration substituted the `flash_attn_v100` package from outside. It works, but keeping somebody else's signatures means we can never honestly declare what we support — and it produced a real defect: the KV store and the attention backend ended up gated by DIFFERENT conditions, so a private byte pool could be read by a foreign reader as `e4m3` (on Gemma-4, 40 layers of 48). The replacement declares its own capabilities in one place and **refuses at start-up** instead of failing mid-forward. Serves Llama-3.2-1B and **Gemma-4-12B** with output **character-identical** to the vendor backend; `head_dim` 64/128/256/512; sliding window 1024; chunked prefill; KV pool fp16 / `fp8_e4m3` / `int8_per_token_head`. **Prefill 1.88x at L=1024 and 3.12x at L=4096** vs the fork's Triton backend (reproducible to 0.2% over three passes) — the mechanism is that **Triton emits no HMMA on `sm_70`** and computes attention on the FMA pipe. Byte pool gives **x2.00 context** (11 261 -> 22 523 tokens for Gemma-4 on one V100-32GB); decode parity on fp16, up to 1.42x on bytes. On Volta ours is the only backend that serves an `e4m3` pool at all (Triton: `type fp8e4nv not supported in this architecture`). |
| Byte KV pool: which format to default to | `done` (`fp8_e4m3`), and the reason is NOT accuracy | int8-with-per-position-scale is 3x more accurate (quantisation cost 7.3-10.2e-3 vs 2.5-2.9e-2; dNLL +0.000324 vs +0.000733) but **does not boot on Gemma-4**: the engine unifies pages only by an integer multiple of block_size, and int8 gives 66560/1032 = **64.50**. fp16 and e4m3 give exactly 64. So `int8_per_token_head` stays declared for homogeneous geometry, and e4m3 is the default. The private-format mine is closed by REFUSAL rather than hope — `supports_kv_connector() -> False` plus a format mark ON THE CACHE that every reader verifies. Full inventory of pool readers: only attention backends interpret the VALUES; prefix caching hashes tokens, swap/offload/disk are byte copies, spec-decode never indexes it. |
| Sliding window in the paged DECODE kernel | `done` (`kv_start`) | The decode kernel had no window at all. Added a `kv_start` int32 `[B]` giving the left bound per sequence (`nullptr` reproduces the old behaviour bit-for-bit). Chosen by measurement over two alternatives on the worst case (window edge off a block boundary): in-kernel bound relL2 **2.1e-4** at base cost; block-table truncation the same cost but **1.2e-1**; exact gather-then-dense 3.3-4.4x more expensive. Without any bound the kernel is 5.3x slower at T=16K **and wrong** — the engine has already freed the pages left of the window. Honest caveat: truncation produced text identical to the vendor backend on the 4 test prompts, so the choice rests on the number, not on an observed failure. |
| Two silent defects found in OUR OWN code during the port | `done` | (a) The paged-decode dispatcher ended in a default branch, so an unsupported `head_size` returned **wrong results instead of refusing** — and that path was exactly the one the vLLM backend used. Now `TORCH_CHECK(d in {64,128,256,512})`. (b) The backend's Python wrapper cost **ten times** the kernels it served (13-16 ms against 1.3-4.0 ms per step) because it recomputed per layer what is shared per KV group; fixing it is what moved decode from "7-14% slower than Triton" to parity. The seam again cost more than the work. |
| FP8 attention | `out of scope` | Volta has no FP8 tensor-core path; this belongs to newer architectures. |

## Summary

Development priority: **backward → varlen → stateful/paged KV cache → native GQA/MQA prefill →
dropout → ALiBi/softcap and an extended window API**. A task counts as finished only when it has
FP32-reference tests, a reproducible benchmark, and its `sm_70` limitations stated explicitly.

## Сделано в этой сессии (прямой проход, head_dim=256)

* **Ядро с разделением варпов по ролям** (`fa2_sm70/csrc/volta_fwd_ws.cuh`): один блок, 12 варпов --
  четыре везут плитку K/V, восемь считают, связь именованными барьерами, одна встреча на плитку.
  Замерено x1.03-1.36 к нашему fp16-ядру и x1.86-2.30 к SDPA; проверено 16 тестами pytest и 15
  формами через точку входа. Включено по умолчанию, откат `FA2SM70_WS=0`.
* Байтовый путь прошёл x0.65 -> x1.03-1.36 за счёт КОМПОНОВКИ, а не ускорения операций: покупка
  занятости дроблением по запросам, прямая адресация, возвращённый разворот, роли варпов.
* Полная аппаратная матрица в `fa2_src/fmha_kernel/gemm/volta_prims.h` (§0 и §4e-§4kk): что даром,
  что складывается, что замерено мёртвым.

### Осталось по этому ядру

* Подключить формат 8 (смещённый байт + чередующаяся таблица) к обвязке vLLM: ядро и питон-API
  готовы (`quantize_kv_prefill` / `prefill_bytes`), но КВАНТОВАТЕЛЬ движка пока пишет знаковый байт и
  две раздельные таблицы. Выигрыш 1.06× замерен на стенде, а не в сквозном прогоне.
* Прямое ядро `e4m3` (см. строку в backlog): снимает необходимость в приватном формате целиком.
* Бо́льшая плитка варпа `MB=2` при `d=256` -- единственный оставшийся рычаг по разложению фаз,
  упирается в 111 616 Б разделяемой памяти при пределе 98 304.
* Открытая ветка по оси времени: структурное расщепление считающих по ЧЁТНОСТИ плиток (цена посчитана
  наперёд -- 64 регистра вместо 32 на половину, место отказа известно: слияние состояний m, l, O).
* Перемерить `d=512` с той же структурой: прежняя оценка "потолок 19.5%" относится к пути cutlass.

## Проход 31.07.2026: перемер, формат 8, и опровержение собственной доктрины

Все числа README и графики перемерены на свободной карте с частотами 1530 МГц (было 1380), с печатью
числа чужих процессов до и после каждого прогона. Что изменилось по существу, а не по частоте:

* **Отгружен формат 8** байтового префилла -- 1.061-1.064x к прежнему int8-пути, побитово тот же
  ответ, точность int8 при скорости e5m2. Публичный API + тест.
* **Закрыт плавающий NaN** в отгруженном fp16-ядре `d=256`: хвостовая плитка ключей читала за конец
  тензора, и хотя `P` этих столбцов ровно ноль, `0 * inf = NaN` портил всю строку выхода через второе
  умножение. Свип 64/180 -> 0/180, `compute-sanitizer` 900 недопустимых чтений -> 0, скорость 1.0031.
  Зажим адреса вместо предикатной загрузки: предикат замерен как -7.7%.
* **Опровергнута наша же формулировка** «ядро за стеной выдач»: слот стоит 0.22-0.30 такта, ядро на
  61% собственного счётного потолка, связывает пропускная разделяемой памяти. Раздел §3b в
  `docs/VOLTA_SM70.md` дополнен двумя подразделами -- границей применимости и разложением по фазам.
* Добавлен каталог ядер: [`docs/KERNELS.ru.md`](docs/KERNELS.ru.md) / [`docs/KERNELS.md`](docs/KERNELS.md).
