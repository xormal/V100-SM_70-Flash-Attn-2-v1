# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# End-to-end demo: run Gemma-4 (Gemma4Unified) text generation on a V100 with the fa2_sm70 kernels
# serving the head_dim=512 global (full_attention) MQA layers. The tested Gemma MQA layout has no
# working SDPA route on Volta at long context. The demo verifies that greedy output is token-identical
# to a pure-eager reference computed from the same weights. Requires transformers>=5.10 (which has
# gemma4_unified) plus this package.
#
#   G4_PATH=/path/to/gemma-4-12B-it python integrations/run_gemma4.py
import os, sys, time, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrations import gemma4_transformers as G
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = os.environ.get("G4_PATH")
if not PATH:
    raise SystemExit("Set G4_PATH to a local Gemma-4 model directory")
NTOK = int(os.environ.get("G4_NTOK", "40"))
G.register()
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, attn_implementation="fa2_sm70",
    # [fp16, А НЕ bf16 -- ЭТО ЗАМЕР, А НЕ ВКУС]
    # На sm_70 cuBLAS отдаёт bf16 SIMT-ядру magma_sgemmEx: sm__inst_executed_pipe_tensor = 0 на ВСЕХ
    # 36 замерах, плитка 64x64, трафик DRAM до 40.5x идеального. Цена -- x8.0...x8.8 на КАЖДОМ
    # умножении модели. Прежние «437 ток/с» на этом пути объясняются этим полностью.
    # И fp16 здесь не компромисс, а улучшение по ОБЕИМ осям: на Gemma-4 замерено relL2 0.0044 против
    # 0.2657 у bf16 (в 60 раз точнее), при максимуме активации 262 против диапазона fp16 до 65504 --
    # запас x250, переполнение недостижимо.
                                             dtype=torch.float16, device_map="cuda:0").eval()
msgs = [{"role": "user", "content": "In one sentence, what makes the V100 GPU architecturally different from newer GPUs?"}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = enc["input_ids"].to("cuda:0")


@torch.no_grad()
def gen():
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(ids, max_new_tokens=NTOK, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    return out[0, ids.shape[1]:], time.time() - t


G.REF_EAGER = False
new_ours, t_ours = gen()
print(f"[fa2_sm70] {t_ours:.2f}s {NTOK/t_ours:.1f} tok/s :: {tok.decode(new_ours, skip_special_tokens=True)!r}")
G.REF_EAGER = True
new_ref, _ = gen()
n_eq = int((new_ours == new_ref).sum())
print(f"[eager ref] greedy token match {n_eq}/{NTOK}  identical={n_eq == NTOK}")
