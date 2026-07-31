# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Wire the fa2_sm70 kernels into HuggingFace transformers as an attention backend, for the
# Gemma-4 unified model on V100 (sm_70). Why this is needed: the model's *global* (full_attention)
# layers use head_dim=512 MQA, and torch SDPA on Volta has NO backend for d=512 (flash=sm80+,
# mem-efficient rejects d>256, only MATH remains -> O(Sq*Sk) score materialisation -> OOM at long
# context). fa2_sm70's decode/prefill kernels are the only runnable path there. The *sliding* layers
# use head_dim=256, which SDPA can serve, so we route ONLY d=512 to our kernel and delegate the rest
# to SDPA (with its correct sliding-window mask) -- a minimal, correct integration that makes the
# model actually serveable on V100.
#
# Usage:
#   from integrations.gemma4_transformers import register
#   register()                                   # adds "fa2_sm70" to ALL_ATTENTION_FUNCTIONS
#   model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="fa2_sm70", ...)
import os
import torch
import fa2_sm70

# When set, the d=512 global layers use pure-PyTorch eager attention instead of our kernel. eager is
# the only *correct* reference for d=512 on Volta (SDPA has no backend there -> it raises), so this
# lets one loaded model produce an apples-to-apples reference for the kernel path. Off in real serving.
REF_EAGER = bool(int(os.environ.get("FA2SM70_REF_EAGER", "0")))
REF_SDPA = bool(int(os.environ.get("FA2SM70_REF_SDPA", "0")))    # route ALL layers to SDPA (reference)
DECODE_SLIDING = bool(int(os.environ.get("FA2SM70_DECODE_SLIDING", "0")))  # sliding d256 decode -> our kernel
# Sliding (d256) PREFILL: default -> SDPA (fp32-softmax, most precise; fine while it fits memory). Set
# =1 to use our banded-causal windowed kernel -- REQUIRED for long context (SDPA's sliding mask forces
# the O(Sq^2) MATH path -> 15 GB score @16k -> OOM); our kernel is O(Sk*window) and O(n) memory.
SLIDING_PREFILL = bool(int(os.environ.get("FA2SM70_SLIDING_PREFILL", "0")))


def _sdpa(module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                  scaling=scaling, sliding_window=sliding_window, **kwargs)


def _eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    # Self-contained eager attention (matches transformers' eager_attention_forward exactly): the only
    # numerically-correct d=512 reference on Volta, since SDPA has no d=512 backend there.
    g = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(g, dim=1) if g > 1 else key
    v = value.repeat_interleave(g, dim=1) if g > 1 else value
    attn = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None and torch.is_tensor(attention_mask):
        attn = attn + attention_mask[..., : k.shape[-2]]
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous()       # [B,Sq,Hq,D]
    return out, attn


def _fa2_sm70_attention_forward(module, query, key, value, attention_mask, scaling=None,
                                dropout=0.0, sliding_window=None, **kwargs):
    """transformers attention-interface contract: query [B,Hq,Sq,D], key/value [B,Hkv,Sk,D]
    (NOT pre-repeated for GQA -- our kernel consumes Hkv<Hq natively). Returns
    (attn_output [B,Sq,Hq,D], attn_weights=None)."""
    D = query.shape[-1]
    Sq = query.shape[2]
    if scaling is None:
        scaling = D ** -0.5
    if REF_SDPA:                                                # reference: every layer via SDPA
        return _sdpa(module, query, key, value, attention_mask, scaling, dropout, sliding_window, **kwargs)
    # SLIDING d=256 layers. PREFILL (Sq>1) -> our banded-causal (sliding-window) kernel: exact,
    # O(Sk*window) not O(Sk^2), O(n) memory -> long-context works where SDPA falls to MATH and OOMs.
    # DECODE (Sq==1) -> optionally our kernel via a KV slice to the window (default SDPA; window=1024
    # gives no throughput headroom). The d=512 GLOBAL layers have no efficient Volta backend -> ours.
    if D != 512:
        if SLIDING_PREFILL and Sq > 1 and sliding_window is not None:
            O, _ = fa2_sm70.attention(query.half(), key.half(), value.half(), causal=True,
                                      scale=float(scaling), window=int(sliding_window))
            return O.to(query.dtype).transpose(1, 2).contiguous(), None
        if DECODE_SLIDING and Sq == 1:
            k, v = key, value
            if sliding_window is not None and key.shape[2] > sliding_window:
                k, v = key[:, :, -sliding_window:, :], value[:, :, -sliding_window:, :]
            same = k.data_ptr() == v.data_ptr()
            kf = k.half(); vf = kf if same else v.half()
            O, _ = fa2_sm70.decode(query.half(), kf, vf, scale=float(scaling))
            return O.to(query.dtype).transpose(1, 2).contiguous(), None
        return _sdpa(module, query, key, value, attention_mask, scaling, dropout, sliding_window, **kwargs)
    if REF_EAGER:
        return _eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
    # V IS K on these layers (attention_k_eq_v): keep them the SAME fp16 tensor so the decode kernel
    # detects V==K and reads the KV once (VEQK, up to ~1.7x). A separate value.half() would break that.
    same_kv = key.data_ptr() == value.data_ptr()
    qf = query.half(); kf = key.half(); vf = kf if same_kv else value.half()
    if Sq == 1:                                                # decode step
        O, _ = fa2_sm70.decode(qf, kf, vf, scale=float(scaling))
    else:                                                      # prefill (causal); single-seq, no padding
        O, _ = fa2_sm70.attention(qf, kf, vf, causal=True, scale=float(scaling))
    return O.to(query.dtype).transpose(1, 2).contiguous(), None


def register(name="fa2_sm70"):
    """Register the backend so `attn_implementation="fa2_sm70"` selects it."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[name] = _fa2_sm70_attention_forward
    return name
