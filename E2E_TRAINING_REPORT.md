# End-to-End Training Validation

This report records an external validation of the repository's differentiable attention path in
compact GPT-3-style and Llama-3-style causal language models. The objective was integration and
numerical validation: prove that the custom SM70 forward/backward kernels can replace PyTorch
SDPA-efficient attention inside a real optimizer loop without changing convergence.

## Result

**PASS** — both model families completed forward, causal attention backward, full-model gradient
clipping, AdamW updates, and validation for every tested head dimension: `64`, `128`, `256`, and
`512`.

Across all eight architecture/head-dimension combinations:

- maximum initial loss difference: `2.86e-6`;
- maximum logits relative L2 error: `8.74e-5`;
- maximum full-model gradient relative L2 error: `1.15e-4`;
- maximum validation-loss difference after training: below `4.6e-4`;
- maximum validation byte-accuracy difference: `0.078` percentage points;
- every run reduced training loss from approximately `5.50-5.71` to `2.78-3.03` in 30 steps.

These results establish end-to-end feasibility and convergence parity. They do not establish a
universal performance win: at the tested sequence length of 256, attention is sub-millisecond and
is only a small part of the complete model step.

## Provenance

This is a historical integration-validation report. Its timing conclusions apply only to the source
snapshot recorded below; use the current README and `benchmarks/bench_train.py` for release-facing
kernel-performance numbers.

| Component | Value |
|---|---|
| Date | 2026-07-21 |
| Source snapshot | `c3b9005e978b2cd99d9e876d39d611ce02e696f5` |
| GPU | NVIDIA Tesla V100-SXM2-32GB (`sm_70`) |
| Driver | `580.159.03` |
| PyTorch | `2.7.0+cu126` |
| PyTorch CUDA runtime | `12.6` |
| Dataset | `Salesforce/wikitext`, `wikitext-2-raw-v1` |
| Reference backend | `SDPBackend.EFFICIENT_ATTENTION`, forced explicitly |

The source snapshot, dataset, build cache, training harness, and raw JSON outputs were kept in a
separate validation workspace. No generated data, model checkpoints, local paths, or build products
were copied into this repository. Repository-native kernel measurements remain reproducible with
`benchmarks/bench_train.py`.

## Experimental design

Both backends received the same initialized state dictionary and the same deterministic training and
validation batches within each comparison. Parameters, residual paths, MLPs, normalizations, logits,
loss, and AdamW state used FP32. Q/K/V were cast to FP16 at the attention boundary.

Common configuration:

- hidden size: `512`;
- decoder layers: `4`;
- sequence length: `256`;
- batch size: `2` (`512` byte tokens per step);
- vocabulary: `256`, direct UTF-8 byte tokenization;
- optimizer: AdamW, learning rate `3e-4`, betas `(0.9, 0.95)`, weight decay `0.1`;
- global gradient clipping: `1.0`;
- training steps: `30`, with the first five excluded from timing;
- validation: five deterministic batches;
- timing: three complete runs per architecture; isolated attention used five warmups and 30 measured
  forward-plus-backward iterations per run.

The Llama-3-style model uses RoPE, RMSNorm, SwiGLU, tied token/output weights, and GQA where possible.
The GPT-3-style model uses learned positional embeddings, LayerNorm, GELU, tied token/output weights,
and MHA.

| Head dim | Llama Q/KV heads | Llama parameters | GPT Q/KV heads | GPT parameters |
|---:|---:|---:|---:|---:|
| 64 | 8 / 4 | 12,718,592 | 8 / 8 | 12,872,960 |
| 128 | 4 / 2 | 12,718,592 | 4 / 4 | 12,872,960 |
| 256 | 2 / 1 | 12,718,592 | 2 / 2 | 12,872,960 |
| 512 | 1 / 1 | 13,767,168 | 1 / 1 | 12,872,960 |

For the Llama GQA reference, K/V were expanded to query-head count before the forced
SDPA-efficient call. The custom API received the native GQA layout; its current training wrapper
performs the corresponding expansion before backward and reduces `dK`/`dV` to K/V-head count
afterward. Both wrapper costs are included and matter when interpreting small-shape timings.

## Numerical correctness

Before optimization, one complete model forward/backward pass compared loss, all logits, and one
flattened vector containing every model gradient.

| Architecture | Head dim | Absolute loss difference | Logits relative L2 | Full gradients relative L2 |
|---|---:|---:|---:|---:|
| Llama-3-style | 64 | 1.43e-6 | 5.93e-5 | 1.12e-4 |
| Llama-3-style | 128 | 2.86e-6 | 8.74e-5 | 1.14e-4 |
| Llama-3-style | 256 | 0 | 8.63e-5 | 1.15e-4 |
| Llama-3-style | 512 | 1.43e-6 | 8.16e-5 | 9.57e-5 |
| GPT-3-style | 64 | 0 | 3.20e-5 | 6.75e-5 |
| GPT-3-style | 128 | 4.77e-7 | 4.00e-5 | 7.50e-5 |
| GPT-3-style | 256 | 9.54e-7 | 1.99e-5 | 6.78e-5 |
| GPT-3-style | 512 | 9.54e-7 | 3.64e-5 | 7.12e-5 |

The differences are consistent with FP16 operation-order differences and did not cause material
training divergence.

## Training and validation

The table contains the canonical first full run. Quality values are comparable between backends
within a row, not across head dimensions: each head dimension used a different deterministic
initialization and batch seed.

| Architecture | d | Initial loss, FA2 / SDPA | Final loss, FA2 / SDPA | Validation loss, FA2 / SDPA | Validation PPL, FA2 / SDPA | Byte accuracy, FA2 / SDPA |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3-style | 64 | 5.707933 / 5.707934 | 2.783708 / 2.783644 | 2.745279 / 2.745219 | 15.5690 / 15.5680 | 26.7969% / 26.7969% |
| Llama-3-style | 128 | 5.693637 / 5.693640 | 2.951103 / 2.951140 | 2.692767 / 2.692762 | 14.7725 / 14.7724 | 28.2422% / 28.2422% |
| Llama-3-style | 256 | 5.601137 / 5.601137 | 2.940590 / 2.940519 | 2.882477 / 2.882472 | 17.8585 / 17.8584 | 28.4766% / 28.4766% |
| Llama-3-style | 512 | 5.498905 / 5.498903 | 2.932456 / 2.932126 | 2.812731 / 2.812638 | 16.6553 / 16.6538 | 24.9219% / 25.0000% |
| GPT-3-style | 64 | 5.556232 / 5.556232 | 2.842605 / 2.842578 | 2.773684 / 2.773698 | 16.0175 / 16.0178 | 26.7969% / 26.7969% |
| GPT-3-style | 128 | 5.668872 / 5.668872 | 2.995888 / 2.995782 | 2.740142 / 2.740149 | 15.4892 / 15.4893 | 27.1875% / 27.1484% |
| GPT-3-style | 256 | 5.580097 / 5.580098 | 3.031020 / 3.030828 | 2.925116 / 2.925389 | 18.6364 / 18.6415 | 26.6797% / 26.6797% |
| GPT-3-style | 512 | 5.555705 / 5.555704 | 2.904803 / 2.904556 | 2.865437 / 2.865261 | 17.5567 / 17.5536 | 25.8984% / 25.8594% |

Perplexity here is byte-level perplexity, not a standard subword-token language-model score. Thirty
steps are sufficient for integration and convergence validation, not for a model-quality claim.

## Isolated attention timings

Each latency is the median of three per-run mean causal forward-plus-backward times. Speedup is
`SDPA time / FA2 time`, so values above `1.0x` favor FA2. The interval is the minimum-to-maximum paired
speedup across the three runs.

| Architecture | d | FA2, ms | SDPA-efficient, ms | Paired speedup median (range) |
|---|---:|---:|---:|---:|
| Llama-3-style GQA | 64 | 0.518 | 0.552 | 1.064x (0.960-1.081x) |
| Llama-3-style GQA | 128 | 0.531 | 0.592 | 1.114x (1.108-1.754x) |
| Llama-3-style GQA | 256 | 0.703 | 0.646 | 0.966x (0.867-1.274x) |
| Llama-3-style MHA | 512 | 0.689 | 0.500 | 0.729x (0.716-0.771x) |
| GPT-3-style MHA | 64 | 0.394 | 0.398 | 0.969x (0.898-1.015x) |
| GPT-3-style MHA | 128 | 0.342 | 0.358 | 1.134x (1.010-1.207x) |
| GPT-3-style MHA | 256 | 0.525 | 0.465 | 0.886x (0.848-0.890x) |
| GPT-3-style MHA | 512 | 0.686 | 0.493 | 0.722x (0.710-0.800x) |

The repeatable result for this historical snapshot at this sequence length is that `d=128` was
competitive or faster, while its `d=512` backward path was slower than SDPA-efficient.
Sub-millisecond results at the remaining shapes are too sensitive to clocks and host scheduling for a
release-level performance claim.

## End-to-end step timings

The synchronized step timer includes host-to-device batch copies, forward, cross-entropy, backward,
global gradient clipping, and AdamW update.

| Architecture | d | FA2 step, ms | SDPA-efficient step, ms | Paired speedup median (range) |
|---|---:|---:|---:|---:|
| Llama-3-style | 64 | 29.61 | 29.86 | 1.023x (1.008-1.190x) |
| Llama-3-style | 128 | 27.87 | 26.97 | 0.968x (0.808-1.124x) |
| Llama-3-style | 256 | 29.46 | 26.10 | 0.882x (0.798-1.046x) |
| Llama-3-style | 512 | 26.76 | 26.41 | 1.075x (0.987-1.097x) |
| GPT-3-style | 64 | 15.66 | 16.44 | 1.152x (0.873-1.152x) |
| GPT-3-style | 128 | 18.07 | 17.04 | 0.943x (0.753-1.159x) |
| GPT-3-style | 256 | 15.95 | 16.73 | 1.049x (0.788-1.407x) |
| GPT-3-style | 512 | 13.39 | 14.23 | 0.828x (0.795-1.251x) |

Peak allocated VRAM was `259-279 MiB` for the Llama-3-style runs and approximately `262 MiB` for
the GPT-3-style runs. This confirms that these are integration models, not saturation benchmarks.

## Conclusions and follow-up

1. The custom SM70 path is viable for real training: every tested shape completed a full autograd and
   optimizer loop.
2. Loss, logits, complete gradients, convergence, validation perplexity, and validation byte accuracy
   agree closely with forced SDPA-efficient attention.
3. The short-context experiment does not support a universal speedup claim. MLP, normalization,
   optimizer work, launch overhead, and host scheduling dominate the complete step.
4. The clearest backward optimization target is `d=512`, which was consistently slower than the
   reference in the isolated comparison.
5. A release-grade performance comparison should use fixed application clocks, randomized or
   interleaved backend order, longer sequences, and the repository's allocation-free timing protocol.

## Summary

The check passes: the FA2 SM70 kernel trains compact GPT-3-style and Llama-3-style models at
`d=64/128/256/512`. Agreement with a forced SDPA-efficient baseline is confirmed on loss, logits,
the gradient vector, validation perplexity and byte accuracy. The residual differences do not
affect convergence of these runs.

At the short context `S=256` there is no basis for claiming a general speed win: the per-step times
are under a millisecond and the end-to-end time depends heavily on the rest of the model and on
background load. The clear place for further optimisation is the backward pass at `d=512`.
