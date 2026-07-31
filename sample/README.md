# Sample: run Gemma-4 on a V100 with `fa2_sm70`

Gemma-4 (`Gemma4UnifiedForConditionalGeneration`) alternates two attention shapes: *sliding* layers
(16/8 GQA, `head_dim=256`, window 1024) and *global* `full_attention` layers (every 6th; **16/1 MQA,
`head_dim=512`, with V ≡ K**). On the tested Volta stack, this **Gemma d512 MQA layout has no working
torch-SDPA route**: flash requires `sm_80+`, while the available fallback materializes too much state
at long context. This package provides a runnable path for those layers; the sample wires it in as a
transformers attention backend and generates text end-to-end.

---

## 1. Requirements

| | |
|---|---|
| GPU | NVIDIA **V100 / any Volta (`sm_70`)**, ideally 32 GB |
| CUDA | a CUDA **12.x** toolkit whose `nvcc` still targets `compute_70` (e.g. 12.4–12.8) |
| Host compiler | a `gcc` your `nvcc` accepts (**nvcc 12.8 rejects gcc > 14** — use gcc 11–13) |
| PyTorch | a CUDA build that includes `sm_70` (e.g. `torch==2.7.0` `cu126`) |
| transformers | **≥ 5.10** (this is the version that ships the `gemma4_unified` architecture) |
| extras | `accelerate`, `safetensors` |

## 2. Install

```bash
# from the repo root (the directory that contains setup.py):
python -m venv .venv && . .venv/bin/activate
pip install "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu126
pip install "transformers>=5.10" accelerate safetensors
pip install -e .                     # installs fa2_sm70; CUDA kernels JIT-build on first use
```

The three CUDA extensions **compile for `sm_70` lazily, each on first use**, and are cached under
`fa2_sm70/_build/`: **prefill** (cutlass FMHA) on the first `attention` call, **decode** (split-KV) on the
first `decode` call, and **backward** on the first autograd/training call (`flash_attn_func` with grad) —
so pure inference never builds it. The prefill/backward kernels build from the bundled CUTLASS sources in
`fa2_src/` — no network needed. Each first call takes ~1–2 min; then instant.

If your default compiler is too new for `nvcc`, point the build at an older one:

```bash
export CC=/usr/bin/gcc-13 CXX=/usr/bin/gcc-13 CUDAHOSTCXX=/usr/bin/gcc-13
export CUDA_HOME=/path/to/cuda-12.x      # the toolkit whose nvcc supports sm_70
```

## 3. Get the model

Gemma-4-12B-it is ~23 GB (`bfloat16`, one `model.safetensors`). Download it to a local directory (it is
a gated/preview checkpoint — accept the license on its model page first, then):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                                  # once, with a token that has access
huggingface-cli download google/gemma-4-12b-it \
    --local-dir /path/to/gemma-4-12B-it                # ~23 GB
```

(If you already have the folder locally, just point `G4_PATH` at it — no download needed. Prefer a
fast SSD; loading 23 GB from a spinning disk is slow.)

## 4. Run

```bash
export G4_PATH=/path/to/gemma-4-12B-it
export CUDA_VISIBLE_DEVICES=0
python sample/run_gemma4.py                 # generate
python sample/run_gemma4.py --verify        # generate + check greedy token-parity vs eager
```

Expected (single 32 GB V100):

```
loading /path/to/gemma-4-12B-it (attn_implementation=fa2_sm70) ...
loaded in ~11s
[fa2_sm70]  ~4s  (~10 tok/s)
  The V100 GPU is architecturally distinct from newer generations because it utilizes the Volta ...
[verify] greedy token match vs eager: 40/40  identical=True
```

`--verify` runs a pure-eager reference from the *same* weights (eager is the only numerically-correct
`head_dim=512` MQA reference on the tested Volta stack, since SDPA raises for that layout) and confirms
the kernel output is **token-identical**. Per-layer accuracy vs an fp32 oracle is `relL2 ≈ 2.2e-4`
prefill / `≈ 1e-6`
decode (see `../tests/`).

## 5. What the backend does (`../integrations/gemma4_transformers.py`)

`register()` adds `attn_implementation="fa2_sm70"` to transformers' attention registry, then routes:

| layer | phase | path |
|---|---|---|
| **global** `d=512` MQA (V≡K) | prefill | `fa2_sm70.attention` (cutlass FMHA; **the only Volta backend for d=512**) |
| **global** `d=512` MQA (V≡K) | decode  | `fa2_sm70.decode` (split-KV; auto-detects **V≡K → reads KV once**, and **MQA-fuses** heads) |
| **sliding** `d=256` GQA (win 1024) | prefill | SDPA by default (most precise while it fits); set `FA2SM70_SLIDING_PREFILL=1` for our banded-causal `fa2_sm70.attention(window=1024)` (`O(Sk·W)`, **required for long context** — SDPA's sliding mask OOMs there) |
| **sliding** `d=256` GQA | decode | SDPA (works for `d≤256`; set `FA2SM70_DECODE_SLIDING=1` to use our kernel) |

## 6. Memory / long context

The 23 GB model on one 32 GB V100 leaves ~9 GB → good for **~8k context** on a single card. For longer
context spread the model across GPUs **and** enable our windowed sliding-prefill kernel (SDPA's sliding
mask OOMs at length — a 16k sliding layer wants a 15 GB score matrix):

```bash
export CUDA_VISIBLE_DEVICES=0,1
export FA2SM70_SLIDING_PREFILL=1                  # our O(Sk·window) banded kernel for the d=256 layers
python sample/run_gemma4.py --device auto         # device_map="auto"; confirmed working at 16k / 24k
```

At these lengths the cost is dominated by the 8 `d=512` global `full_attention` layers doing genuine
`O(Sq²)` causal attention — exact softmax attention at ~19.5% of the 125-TFLOPS data-sheet roofline.

> **Correction (measured later).** That figure is the ceiling of *that path*, not of the hardware. The
> warp-role kernel at `d=256` (see `../README.md`, "Forward at head_dim=256") reaches **24.2–33.1%** of
> the same roofline where the cutlass-class path reaches 17.8–31.8%, by splitting warps into hauling
> and computing groups. The `d=512` number above has not been re-measured with that structure, so read
> it as "the ceiling of the cutlass-class path at `d=512`" rather than as a property of Volta.
