# Gemma-4-12B, head-to-head — our sm70 kernel (V100) vs FlashAttention-2 (A100/PG199)

Same model (`google/gemma-4-12B-it`, bf16, byte-identical `model.safetensors`; `config.json` md5 matches).
Our kernel runs on a **Tesla V100 (sm_70)**; the reference is a real prebuilt **FlashAttention-2
(flash_attn 2.8.3)** on an **A100 (sm_80, reported "DRIVE-PG199-PROD", GPU3)**. Kernels compared directly on
the model's own attention shapes (`benchmarks/kbench_sliding.py`, `examples/compare_run.py`).

> **Setup & provenance** (all numbers below were produced under exactly this; they are point measurements,
> not guarantees — re-run to reproduce). Measured **2026-07-16**, batch 1, greedy, one GPU each unless noted.
> - **V100 side**: Tesla V100-SXM2-32GB, driver 580.159.03, `torch 2.7.0+cu126`, `transformers 5.14.1`,
>   `TORCH_CUDA_ARCH_LIST=7.0`; package at repo commit `f94f6a2` (`git rev-parse --short HEAD`).
> - **A100 side**: A100 (cc 8.0), `torch 2.9`, `flash_attn 2.8.3`, `transformers 5.14.1`, cuda-12.9 nvcc.
> - **Commands**: `benchmarks/kbench_sliding.py` (kernel head-to-head, `ATTN=fa2_sm70` / `ATTN=flash`),
>   `examples/{compare_run,e2e_throughput}.py` (whole-model). Default power/clocks (no locked SM clock).
> - Raw console logs are the scripts' own stdout; nothing here is averaged across machines.

## 1. Kernel head-to-head on the sliding layers (`d=256`, GQA 16/8, causal window 1024)

These are the shapes FlashAttention-2 *can* run. Same inputs (fp16), same window/scale.

| seqlen | ours V100 (µs) | FA2 A100 (µs) | FA2 faster | ours V100 (TFLOP/s) | FA2 A100 (TFLOP/s) |
|--:|--:|--:|--:|--:|--:|
| 512   | 151   | 83   | 1.8× | 28 | 52 |
| 1024  | 458   | 142  | 3.2× | 38 | 121 |
| 2048  | 1262  | 277  | 4.6× | 27 | 124 |
| 4096  | 2965  | 543  | 5.5× | 23 | 127 |
| 8192  | 6490  | 1092 | 5.9× | 21 | 126 |
| 16384 | 13620 | 2177 | 6.3× | 20 | 126 |

**Numerically equivalent.** On identical inputs, our output vs the FA2 output is `relL2 = 2.07e-4`, and
against an fp32 oracle **both** kernels sit at `2.48e-4` — i.e. our sm70 kernel is *exactly as accurate as
FlashAttention-2* (fp16 level), it just casts bf16→fp16 first (sm_70 has no bf16 tensor cores).

**Reading the speed gap honestly.** FA2 on the A100 is ~1.8–6.3× faster (growing with context). That
splits into two independent factors, neither a bug in our kernel:
- **~2.5× hardware.** A100 fp16 tensor peak ≈ 312 TFLOPS + ~1.5 TB/s HBM; V100 ≈ 125 TFLOPS + 0.9 TB/s.
  Both are **data-sheet** peaks, compared like with like. The `%roofline` figures elsewhere in this
  repo use the **measured** V100 ceiling of 93.6 TFLOP/s instead, so a percentage here and a
  percentage there are not the same denominator — this is the only file that quotes the spec one.
- **~2× kernel efficiency on its native arch.** FA2 reaches **~40 % of the A100 peak** (126/312); our
  kernel reaches **~20 % of the V100 peak** (25/125) — which is the Volta cutlass-FMHA ceiling for
  `d=256` (the output-accumulator path). FA2 exploits Ampere-only features (async copy, 3rd-gen tensor
  cores) that Volta simply does not have.

So on equal footing (fraction of the GPU's own peak) FA2 is ~2× more efficient; the rest is a stronger
> **Correction (measured later).** "Practical ceiling for Volta" describes the cutlass-class
> path. At `d=256` the warp-role kernel reaches 24.2-33.1% of the 125-TFLOPS roofline against
> 17.8-31.8% for that path, so the ceiling belongs to the path rather than to the hardware.

GPU. Our kernel is at the practical ceiling *for Volta* — and it is the reference kernel that is missing
from that platform, not ours.

## 2. FlashAttention-2 runs the sliding layers, but **not** the `d=512` global layers (hybrid)

Gemma-4-12B alternates the `d=256` sliding layers above (40 of 48 layers) with **global layers (every 6th)
of `head_dim = 512`, MQA 16/1**. `flash_attn` supports head_dim **up to 256 only** — both mainline
`flash_attn 2.8.3` *and* the vLLM fork (its README: "all head dimensions up to 256"; there are no d512
FA kernels). So FA2 runs the 40 sliding layers fine, but the 8 global layers raise:

```
RuntimeError: FlashAttention forward only supports head dimension at most 256
  (the model passes q(1,8,16,512) k(1,8,1,512) on the global layers)
```

So on the A100 the model is served by a **hybrid**: real FlashAttention-2 on every `d≤256` sliding layer
(the majority) + torch SDPA on the 8 `d=512` global layers (the only backend there that does d512). Our
sm70 kernel needs no split — it does head_dim up to 1024 (d512 via the output-accumulator path) in one
kernel, so it runs the **whole** model on a V100, the older card that never had FlashAttention at all.

## 3. End-to-end pp / decode, whole model (every layer through the backend under test)

Single GPU, bf16, prompt = 1024 tokens, 128 decoded tokens:

| backend | GPU | prefill (pp) | decode |
|---|---|--:|--:|
| **ours** (sm70, all layers, incl. d512) | V100-32GB (sm_70) | 437 tok/s | 12.9 tok/s (77 ms/tok) |
| **FA2 hybrid** (real FA2 on the 40 d256 layers + SDPA on the 8 d512) | A100 (sm_80) | 4624 tok/s | 18.0 tok/s (56 ms/tok) |
| SDPA (all layers) | A100 (sm_80) | 4527 tok/s | 18.0 tok/s (56 ms/tok) |

Two things stand out. **(a)** On the A100, using real FlashAttention-2 for the sliding layers (hybrid)
is within ~2 % of plain SDPA and *identical* on decode (18.0 tok/s) — for a 12 B model the attention
backend barely moves end-to-end throughput; the projection/MLP matmuls dominate. **(b)** The A100 leads
pp by ~10× (compute-bound: A100 ~312 vs V100 125 TFLOPS peak) but decode by only ~1.4× (memory-bound:
per token both stream the full weights, and A100's ~1.5 TB/s HBM is only ~1.7× the V100's 0.9 TB/s).

**Bit-identical decisions.** Greedy generation, first 24 tokens of four prompts — our V100 run matched the
A100 run **token-for-token on every prompt** (21/21, 24/24, 24/24, 24/24). Same answers verbatim
(*"…is Paris, and the river … is the Seine."*, the one-line Fibonacci, Rayleigh scattering, 17×24).

## 4. Does our sm70 kernel even build on the A100? Yes — but it is Volta-native at the ISA level

Out of curiosity we built the package **on the A100** (`TORCH_CUDA_ARCH_LIST=8.0`, cuda-12.9, torch 2.9).
It compiles clean (98 s) and runs **correctly**: our A100 output vs our V100 output `relL2 = 2.4e-5`, and
both sit at `2.48e-4` vs the fp32 oracle — identical accuracy. It even runs the **d512 global** layer on
the A100 (`relL2 2.48e-4`), the one native FA2 refuses. So the port is not V100-locked in *source*.

But it is **~6× slower on the A100 than on the V100**, and ~30× slower than native FA2 on the same card:

| d256 sliding, seqlen | ours @V100 | ours @A100 | A100/V100 | FA2 @A100 | ours@A100 / FA2@A100 |
|--:|--:|--:|--:|--:|--:|
| 2048  | 1262 µs | 7966 µs | 6.3× slower | 277 µs | 29× |
| 8192  | 6490 µs | 35334 µs | 5.4× | 1092 µs | 32× |
| 16384 | 13620 µs | 71924 µs | 5.3× | 2177 µs | 33× |

### Why it is slower on the *stronger* chip — the tensor-core ISA, with SASS evidence

The cause is not tuning; it is that **the two GPUs implement different tensor-core instructions**, and our
kernel is written against the Volta one. Disassembling the two builds (`cuobjdump -sass`) makes it exact:

| build | arch | tensor-core MMA (`HMMA`) | CUDA-core FMA (`HFMA2`) |
|---|---|--:|--:|
| V100 (`fa2sm70_prefill.so`) | sm_70 | **3328 × `HMMA.884`** | 0 |
| A100 (same source, `TORCH_CUDA_ARCH_LIST=8.0`) | sm_80 | **0** | 168 × `HFMA2.MMA` |

- **Volta (sm_70)** tensor cores execute `HMMA.884` — the warp-level MMA of shape **`m8n8k4`** (contract
  only `k=4` per issue). That is Volta's *native* tensor instruction, and it is the fast path: on the V100
  our kernel issues 3328 of them and runs on the tensor cores (~20–37 TFLOP/s of the 125 TFLOPS peak).
- **Ampere (sm_80)** redesigned the tensor core around a **much wider `m16n8k16`** MMA (`HMMA.16816`,
  `k=16`). The wider tile is the whole point: it amortises each operand register-load over 4× more
  contraction work, which is how Ampere reaches 312 TFLOPS. **Ampere's ISA does not contain the Volta
  `m8n8k4` tensor opcode at all** — it was not carried forward as a hardware instruction, only kept as
  *compilable* PTX for source compatibility.
- So when the `Sm70`-tagged cutlass code (which emits `mma.sync.m8n8k4` PTX) is compiled for sm_80,
  `ptxas` has **no Ampere tensor-core instruction to map it to**. It falls back to **emulating the matmul
  on the general CUDA cores** with `HFMA2` (half2 fused-multiply-add) — note the A100 build has **zero**
  `HMMA`. The attention matmuls leave the tensor cores entirely and run as scalar/vector FP16 FMA with all
  the `m8n8k4` operand-shuffling done in software.

That is why the newer, faster card is *slower* here: on the V100 this is a **tensor-core** kernel; on the
A100 the very same source becomes a **CUDA-core** kernel (no tensor cores used), so it lands at ~4 TFLOP/s
— below even the V100's tensor-core rate, and ~30× under native FA2 (which uses `HMMA.16816`). It stays
bit-exact only because FP16 arithmetic gives the same result whichever unit runs it.

**A real Ampere port** would flip the cutlass arch tag `Sm70 → Sm80` (and the tile shapes), so cutlass
emits `mma.m16n8k16` → `HMMA.16816` SASS → the native Ampere tensor cores. That is a different kernel; the
point of this project is the V100, where `HMMA.884` *is* the fast path.

## Reproduce
`benchmarks/kbench_sliding.py` (kernel head-to-head + numeric agreement via `SAVE`/`LOAD`; set
`TORCH_CUDA_ARCH_LIST=8.0` to build on Ampere) and `examples/compare_run.py` (whole-model tok/s +
token-id diff). Env in `REPRODUCE.md`.
