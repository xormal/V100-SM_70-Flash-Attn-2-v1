"""Per-GPU sustained attention throughput, to expose clock/thermal asymmetry between cards.

Why this exists: on this box the two serving GPUs were measured to differ by 21% on the SAME
attention work inside the server, and a 131K prefill varied 104 s vs 167 s between identical runs.
That looked like a superlinear algorithmic effect and was not -- it is clock behaviour. Run this
BEFORE trusting any long-context timing, and prefer a fixed power cap (e.g. `nvidia-smi -pl 220`)
so the numbers are reproducible.

    CUDA_VISIBLE_DEVICES=<i> python percard_attn.py
"""
import torch, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import fa2_sm70
dev = "cuda"; torch.manual_seed(0)
H, Hkv, d = 12, 2, 256                      # per-GPU shape at TP=2 for Qwen3.6-27B
q = (torch.randn(1, H, 15680, d, device=dev, dtype=torch.float16) * 0.4)
k = (torch.randn(1, Hkv, 65536, d, device=dev, dtype=torch.float16) * 0.4)
v = (torch.randn(1, Hkv, 65536, d, device=dev, dtype=torch.float16) * 0.4)
sc = d ** -0.5
fa2_sm70.attention(q, k, v, causal=True, scale=sc); torch.cuda.synchronize()
ts = []
for _ in range(40):
    t0 = time.perf_counter(); fa2_sm70.attention(q, k, v, causal=True, scale=sc)
    torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
first = sum(ts[:5]) / 5; last = sum(ts[-5:]) / 5
clk = os.popen("nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw "
               "--format=csv,noheader -i " + os.environ.get("CUDA_VISIBLE_DEVICES", "0")).read().strip()
print(f"GPU{os.environ.get('CUDA_VISIBLE_DEVICES','?')}: first5={first*1e3:7.1f} ms  "
      f"last5={last*1e3:7.1f} ms  decay={100*(last/first-1):+5.1f}%   [{clk}]")
