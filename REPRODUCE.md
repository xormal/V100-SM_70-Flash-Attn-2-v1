# Reproducing every result in this repo

A self-contained recipe so an independent researcher can rebuild the kernels, re-run the tests and
benchmarks, and run a real Gemma model end-to-end on the same hardware. All numbers below were measured
on the machine described in "Hardware".

## Hardware

- **4× NVIDIA V100-SXM2-32GB** (`sm_70`, Volta, HMMA.884 tensor cores, no bf16 tensor cores).
- CUDA 12.x driver; the CUDA **12.8** toolkit provides `nvcc` for the JIT build.
- All GPUs used one at a time for the kernel benches; the Gemma-12B demo shards over 1–4 GPUs.

## 1. Environment

The three CUDA extensions (prefill / decode / backward) **JIT-compile on first import** — no manual build
step, but `nvcc` must be on `PATH` and match the torch CUDA version.

```bash
# a venv with a CUDA-12.x torch (this repo was validated on torch 2.7.0+cu126, Python 3.12)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or your matching cuXX wheel

# put nvcc on PATH (adjust to your toolkit location); must be the SAME major CUDA as torch
export PATH=/path/to/cuda-12.8/bin:$PATH
export CUDA_HOME=/path/to/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0                 # Volta

# import once to trigger the JIT build (first call takes ~1–2 min, then cached under fa2_sm70/_build)
PYTHONPATH=$(pwd) python -c "import fa2_sm70, torch, math; \
  q=torch.randn(1,8,64,128,device='cuda',dtype=torch.float16); \
  print(fa2_sm70.attention(q,q,q,causal=True,scale=1/math.sqrt(128))[0].isfinite().all())"
# -> tensor(True, device='cuda:0')
```

For the **Gemma end-to-end demo** you additionally need `transformers` (+ `accelerate` for multi-GPU):

| Gemma family | `transformers` | notes |
|---|---|---|
| Gemma-2 / Gemma-3 | `>= 4.42` | `gemma2`, `gemma3` model types |
| **Gemma-4** (`gemma4_unified`) | **`>= 5.10`** (validated on `5.14.1`) | multimodal; text head_dim = 256 |

```bash
pip install "transformers>=5.10" accelerate sentencepiece
```

### Optional: fixing a broken `accelerate` import (`ERR ... DEFAULT_CIPHERS`)

If `import accelerate` fails with
`ImportError: cannot import name 'DEFAULT_CIPHERS' from 'urllib3.util.ssl_'`, an old `botocore`
(imported transitively via `accelerate → sagemaker → boto3`) is incompatible with `urllib3 >= 2`.
Fix by upgrading `botocore` (does **not** require pinning urllib3 down):

```bash
pip install -U botocore boto3 s3transfer      # botocore >= 1.31 no longer imports DEFAULT_CIPHERS
```

## 2. Correctness (all exact / bit-close to an fp32 oracle)

```bash
python tests/run_all.py         # 20 suites: MXFP4, attention, decode, backward, varlen, dropout, paged,
                                # alibi, window, quant_cache, decode_mod, fa-interface, HMMA map,
                                # d=256 forward, GPTQ dequant, W8A16 GEMV, vLLM int8 gather/pool, byte-KV prefill,
                                # direct e4m3 prefill -> "20/20"
```

## 3. Kernel benchmarks (vs torch SDPA)

```bash
python benchmarks/bench_prefill.py         # prefill vs SDPA / EFFICIENT_ATTENTION
python benchmarks/bench_decode.py          # split-KV decode vs EFFICIENT_ATTENTION
python benchmarks/bench_decode_lengths.py  # decode tok/s vs context length, ours vs SDPA-eff
python benchmarks/bench_train.py           # fwd vs fwd+bwd vs SDPA-eff (d=64/128/256/512)
python benchmarks/bench_bwd_large.py       # ISOLATED backward, d=256/512, with a relL2 check
python benchmarks/bench_modes.py           # window / two-sided / ALiBi / softcap (no SDPA fused path)
```

Expected (V100, causal, `speedup = SDPA_time / ours`): prefill **1.07–1.13×** where many heads already
fill the machine and **1.69–3.26×** where they do not (small batch / few heads — the causal split-K
path); decode **2.3–12.4×** (split-KV vs the parallelism-starved prefill kernel at `Sq=1`); modes SDPA
can't fuse are **13–27×** faster than the materialized-mask MATH fallback (softcap has no SDPA route at
all). Isolated backward: **1.04–1.19×** at `d=64`, **1.16–1.31×** at `d=128`, **1.27–1.73×** at `d=256`,
**1.05–1.10×** at `d=512`.

### Verifying the prefill scheduling levers independently

`bench_prefill.py` prints a `no-split` column that it measures **in a child process** with split-K
switched off, so a single command reproduces both sides of the A/B. Both levers also have off-switches:

```bash
python benchmarks/bench_prefill.py                       # ours (both levers on) + no-split column
FA2SM70_FWD_SPLITS=1 python benchmarks/bench_prefill.py  # split-K off (block-order reversal still on)
FA2SM70_FWD_REV=0 FA2SM70_FWD_SPLITS=1 python benchmarks/bench_prefill.py   # both off = vendor schedule
```

Measured here (V100-SXM2-32GB, **clocks locked at 1530 MHz**, re-measured 2026-07-31 on an idle
card) — note the second column: at `1,1,2048,64` the unsplit kernel is *slower than SDPA-eff*
(0.156 vs 0.148 ms), so scheduling is what turns that shape from a loss into the table's largest win:

| B, H, Sk, d | ours | no-split | SDPA-eff | ours/eff |
|---|--:|--:|--:|--:|
| 1, 32, 2048, 128 | 0.947 ms | 0.944 | 1.031 | 1.09× |
| 1, 32, 8192, 128 | 13.697 ms | 13.614 | 15.071 | 1.10× |
| 1, 1, 2048, 128 | **0.061 ms** | 0.127 | 0.121 | **2.00×** |
| 1, 1, 4096, 128 | **0.140 ms** | 0.262 | 0.269 | **1.92×** |
| 1, 1, 2048, 64 | **0.047 ms** | 0.156 | 0.148 | **3.15×** |
| 1, 2, 4096, 64 | **0.164 ms** | 0.340 | 0.374 | **2.28×** |

Correctness is the same gate as everywhere else: `relL2` vs the fp32 oracle is printed next to every row
and is unchanged by splitting (the partial `O` buffers are fp32), and `tests/run_all.py` stays 20/20.

### Verifying the d=256 block mainloop independently

`d=256` causal runs a hand-written block mainloop instead of the cutlass one. `bench_bwd_large.py`
prints `relL2` against SDPA on every row, and the routing has an off-switch, so the claim can be
checked in one A/B on your own card:

```bash
python benchmarks/bench_bwd_large.py                      # block mainloop (default)
FA2SM70_BLOCK_BWD=0 python benchmarks/bench_bwd_large.py  # forced onto the cutlass path
```

Measured here (V100-SXM2-32GB, 1530 MHz, min-of-6) — note `d=512` is identical in both runs because it
does not route (resident K/V would need 111 KB of shared memory against the 96 KB limit):

| B, H, S, d | block mainloop | cutlass path |
|---|--:|--:|
| 1, 1, 2048, 256 | **1.73×** | 1.03× |
| 1, 8, 512, 256 | **1.27–1.31×** | 1.16× |
| 2, 8, 1024, 256 | **1.35×** | 1.06× |
| 1, 8, 2048, 256 | **1.49×** | 1.09× |
| 1, 32, 2048, 256 | **1.46×** | 1.01× |
| 1, 8, 2048, 512 | 1.05× | 1.04× (same path) |

The `relL2` column differs between the two paths (`4.4e-04` vs `4.0e-05`): the block mainloop keeps its
`dV/dK` accumulators in fp32 registers and converts once at the epilogue, the cutlass path reduces
through an fp32 gmem workspace. Both are well inside the `3e-3` gate that `tests/test_backward.py`
enforces against an fp32 reference.

### Decode vs context length (single-stream, `Sq=1`, `decode(q,K,V)` — the path the model calls)

Our split-KV flash-decoding fills the 80 SMs; SDPA-EFFICIENT (the same cutlass FMHA) has ~1 CTA/head at
`Sq=1`, so it starves — the gap grows with context. `tok/s = 1 / step-latency`; exact (`relL2 ~1e-7`).
(The online decode path had an ~80 µs fixed cost from a separate `logsumexp` launch on the split LSEs;
that LSE is now written by the reduce kernel itself, so a `L=512` step is ~28 µs instead of ~105 µs — the
short-context numbers below reflect that.)

**Gemma-4-12B decode shape** — `H16 / Hkv8 (GQA-2) / d256`, batch 1:

| ctx len | ours µs | ours tok/s | KV GB/s | SDPA-eff µs | SDPA-eff tok/s | speedup |
|--:|--:|--:|--:|--:|--:|--:|
| 512   | 28  | 35955 | 151 | 113  | 8871 | 4.1× |
| 2048  | 65  | 15411 | 259 | 405  | 2470 | 6.2× |
| 4096  | 87  | 11494 | 386 | 735  | 1361 | 8.4× |
| 8192  | 128 | 7840  | 526 | 1460 | 685  | 11.4× |
| 16384 | 235 | 4259  | 572 | 2894 | 346  | 12.3× |
| 32768 | 483 | 2070  | 556 | 5782 | 173  | **12.0×** |

**Llama-style MHA** — `H32 / Hkv32 / d128`, batch 1:

| ctx len | ours µs | ours tok/s | KV GB/s | SDPA-eff µs | SDPA-eff tok/s | speedup |
|--:|--:|--:|--:|--:|--:|--:|
| 512   | 27  | 36945 | 310 | 48   | 21044 | 1.8× |
| 2048  | 71  | 14158 | 475 | 151  | 6621  | 2.1× |
| 8192  | 190 | 5277  | 708 | 581  | 1720  | 3.1× |
| 32768 | 653 | 1531  | 822 | 2306 | 434   | **3.5×** |

We now beat SDPA-EFFICIENT at **every** context length and both configs (down from a `0.5×` loss at
`L=512` before the overhead fix). The speedup still grows with context — SDPA's 32 CTAs starve the 80 SMs
while our split-KV fills them, and GQA wins more (SDPA reads the KV `H/Hkv`×-expanded; we read the shared
KV once). At long context the decode kernel itself approaches the ~800 GB/s HBM ceiling (`KV GB/s` column).

## 4. Gemma end-to-end (this kernel as the attention backend)

```bash
# 12B fits on ONE 32 GB V100 in bf16; use device_map=auto to shard a larger model
MODEL=/path/to/gemma-4-12B-it CUDA_VISIBLE_DEVICES=0,1 \
  PYTHONPATH=$(pwd) python examples/run_gemma_fa2_sm70.py
```

> **Two different entry points — do not conflate them** (both register the name `fa2_sm70`, but route
> differently on purpose):
> - **`examples/run_gemma_fa2_sm70.py`** (this section): a self-contained *kernel exerciser / benchmark*
>   that routes **every layer — all 48 — through the sm70 kernel** (sliding via `attention(window=1024)` /
>   `decode`, global via the same entry points at `d=512`). Use it to measure the kernel on the whole
>   model end-to-end; the tok/s below are from it.
> - **`integrations/gemma4_transformers.py`** (used by `sample/run_gemma4.py`): the *minimal correct
>   serving* backend. By **default it routes only the 8 `d=512` global layers to the kernel** (the tested
>   Gemma d512 MQA layout has no working SDPA route on Volta) and leaves the 40 `d=256` sliding layers on
>   **SDPA** (most precise while it fits); env flags `FA2SM70_SLIDING_PREFILL=1` /
>   `FA2SM70_DECODE_SLIDING=1` move the sliding layers onto
>   our kernel too (needed for long context, where SDPA's sliding mask OOMs). See `sample/README.md` for
>   its routing table.

Per `config.json` the 48 decoder layers are **40 `sliding_attention`** (`head_dim 256`, GQA `16/8`,
`sliding_window 1024`) + **8 `full_attention`/global** (`global_head_dim 512`, MQA `16/1`, V ≡ K).
Measured on **google/gemma-4-12B-it** sharded over 4× V100 with `examples/run_gemma_fa2_sm70.py` (all 48
layers on the kernel):

```
PREFILL : ~262 tok/s     DECODE : ~9.7 tok/s (~103 ms/tok)
Q: What is the capital of France, and what river runs through it?
A: The capital of France is Paris, and the river that runs through it is the Seine.
Q: Write a one-line Python function that returns the n-th Fibonacci number.
A: def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)
Q: In two sentences, explain why the sky is blue.
A: ... Rayleigh scattering, where shorter (blue) wavelengths scatter more ...
```

### Multi-GPU note (important)

With `device_map="auto"`, each decoder layer can live on a different GPU. The kernel entry points take a
`c10::cuda::CUDAGuard(Q.device())` so every launch happens on the **tensors' own device**; without it,
`getCurrentCUDAStream()` would return the default device's stream and a layer on `cuda:1` would silently
read `cuda:0` memory → `NaN` logits. If you fork the kernels, keep that guard.

## 5. Notes on numerics

- Inputs are cast bf16→fp16 on `sm_70` (no bf16 tensor cores); attention is exact-to-fp16 (`relL2 ~1e-3`
  vs an fp32 oracle, per layer). Gemma activations stay within fp16 range after q/k-norm.
- The decode `deferred` path needs a running `max‖k‖`; a bare `decode(q,K,V)` without one takes the exact
  online path instead (it needs no bound) — see `README.md` §Decode.
