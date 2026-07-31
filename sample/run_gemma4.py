# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# End-to-end sample: run Gemma-4 (Gemma4Unified) text generation on a V100 (sm_70) using the fa2_sm70
# kernels for the head_dim=512 global (full_attention) layers. The tested Gemma MQA layout has no
# working SDPA route on Volta at long context. Without these kernels the model is effectively
# unserveable on a V100.
#
#   pip install -e ..                 # from this repo (builds the CUDA kernels on first use)
#   pip install "transformers>=5.10"  # provides the gemma4_unified architecture
#   G4_PATH=/path/to/gemma-4-12B-it python run_gemma4.py            # generate
#   G4_PATH=/path/to/gemma-4-12B-it python run_gemma4.py --verify   # + check token-parity vs eager
#
# See README.md in this directory for the full download/build/run recipe.
import os, sys, time, argparse, torch

# make the sibling `integrations/` package importable when run from a source checkout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrations import gemma4_transformers as G
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--path", default=os.environ.get("G4_PATH"))
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--ntok", type=int, default=40)
ap.add_argument("--prompt", default="In one sentence, what makes the V100 GPU architecturally different from newer GPUs?")
ap.add_argument("--verify", action="store_true", help="also generate a pure-eager reference and report greedy token match")
args = ap.parse_args()
if not args.path:
    ap.error("provide --path or set G4_PATH to a local Gemma-4 model directory")

G.register()   # adds attn_implementation="fa2_sm70" to transformers' attention registry
print(f"loading {args.path} (attn_implementation=fa2_sm70) ...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(args.path)
model = AutoModelForCausalLM.from_pretrained(
    args.path, attn_implementation="fa2_sm70", dtype=torch.bfloat16, device_map=args.device).eval()
in_dev = model.get_input_embeddings().weight.device   # handles both "cuda:0" and "auto" (multi-GPU)
print(f"loaded in {time.time()-t0:.1f}s  (input device {in_dev})", flush=True)

enc = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = enc["input_ids"].to(in_dev)


@torch.no_grad()
def generate():
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(ids, max_new_tokens=args.ntok, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    return out[0, ids.shape[1]:], time.time() - t


G.REF_EAGER = False
new, dt = generate()
print(f"\n[fa2_sm70]  {dt:.2f}s  ({args.ntok/dt:.1f} tok/s)")
print("  " + tok.decode(new, skip_special_tokens=True).replace("\n", " "))

if args.verify:
    # eager is the validated reference for this d=512 MQA layout on Volta
    G.REF_EAGER = True
    ref, _ = generate()
    n_eq = int((new == ref).sum())
    print(f"\n[verify] greedy token match vs eager: {n_eq}/{args.ntok}  identical={n_eq == args.ntok}")
