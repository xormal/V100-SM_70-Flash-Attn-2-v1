"""Sustained attention on ONE GPU until it reaches thermal/clock equilibrium, at a fixed power cap.

The 40-iteration version of this test (percard_attn.py) runs ~18 s and MISSES the collapse: GPU2 on
this box holds ~1357 MHz for the first half-minute and only later falls to ~307 MHz, which is why a
131K prefill (100+ s) swung 60% between runs while the short test looked merely 7% off. Run long.

    CUDA_VISIBLE_DEVICES=<i> SECONDS_RUN=150 python power_sweep.py
"""
import torch, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import fa2_sm70
dev = "cuda"; torch.manual_seed(0)
H, Hkv, d = 12, 2, 256
RUN = float(os.environ.get("SECONDS_RUN", "150"))
g = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
q = (torch.randn(1, H, 15680, d, device=dev, dtype=torch.float16) * 0.4)
k = (torch.randn(1, Hkv, 65536, d, device=dev, dtype=torch.float16) * 0.4)
v = (torch.randn(1, Hkv, 65536, d, device=dev, dtype=torch.float16) * 0.4)
sc = d ** -0.5
fa2_sm70.attention(q, k, v, causal=True, scale=sc); torch.cuda.synchronize()
ts = []; t_start = time.perf_counter()
while time.perf_counter() - t_start < RUN:
    t0 = time.perf_counter(); fa2_sm70.attention(q, k, v, causal=True, scale=sc)
    torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
n = len(ts)
first = sum(ts[:5]) / 5
last = sum(ts[-5:]) / 5
worst = max(ts[5:]) if n > 10 else max(ts)
st = os.popen("nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,power.limit "
              "--format=csv,noheader -i " + g).read().strip()
print(f"pl={os.environ.get('PL','?'):>4}W GPU{g}: n={n:3d} first={first*1e3:6.1f} "
      f"steady={last*1e3:6.1f} worst={worst*1e3:6.1f} ms  decay={100*(last/first-1):+6.1f}%  "
      f"thr={1/last:5.2f}/s  [{st}]", flush=True)
