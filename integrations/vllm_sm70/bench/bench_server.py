"""Prefill/decode throughput against a running vLLM server.

Uses UNIQUE prompts on purpose: with `--enable-prefix-caching` a repeated prompt is served from
cache and reports impossible prefill rates (that is where the bogus "9677 tok/s" for this model came
from -- it would be 520 TFLOP/s against a 2xV100 peak of 250).

    python bench_server.py <port> <tag> 4096,16384,65536
"""
import json, os, time, sys, urllib.request, random
PORT=int(sys.argv[1]); TAG=sys.argv[2]; SIZES=[int(x) for x in sys.argv[3].split(",")]
B=os.environ.get("SERVER_URL", f"http://localhost:{PORT}").rstrip("/")
M=json.loads(urllib.request.urlopen(B+"/v1/models").read())["data"][0]["id"]
random.seed(20260725)
BASE=("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut "
      "labore et dolore magna aliqua enim minim veniam quis nostrud exercitation ullamco laboris "
      "nisi aliquip ex ea commodo consequat").split()
VOCAB=[BASE[random.randrange(len(BASE))] for _ in range(700000)]
def stream(prompt, max_tokens):
    body={"model":M,"prompt":prompt,"max_tokens":max_tokens,"temperature":0,"stream":True,
          "stream_options":{"include_usage":True}}
    req=urllib.request.Request(B+"/v1/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); ttft=None; n=0; usage=None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line=line.strip()
            if not line.startswith(b"data: "): continue
            p=line[6:]
            if p==b"[DONE]": break
            d=json.loads(p)
            if d.get("usage"): usage=d["usage"]
            ch=d.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None: ttft=time.perf_counter()-t0
                n+=1
    return ttft, time.perf_counter()-t0, n, usage
print(f"[{TAG}] model={M}")
_t,_tot,_n,_u = stream(" ".join(VOCAB[:2000]), 1)
TPW=_u["prompt_tokens"]/2000.0
print(f"  (calibration: {TPW:.3f} tokens/word)")
print(f"{'prompt_tok':>10} {'TTFT s':>8} {'prefill tok/s':>14} {'decode tok/s':>13}")
off=2000
for want in SIZES:
    words=int(want/TPW)
    prompt=" ".join(VOCAB[off:off+words]); off+=words
    # WARM UP THIS SIZE FIRST. The very first sized request after a boot pays graph capture, kernel
    # JIT and allocator growth, and it does so ONLY on the first size measured. Evidence: the same
    # server measured 16K twice gave 1728 then 3627/4457 tok/s -- the cold shot understated its own
    # kernel by 2x, while the second size in the same run was already stable to 0.2%. Both our path
    # and the incumbent showed it (incumbent 3269 -> 3970), so it is not a property of either kernel.
    # A separate prompt is used so no prefix cache can be involved.
    warm = " ".join(VOCAB[off:off + max(64, words // 8)]); off += max(64, words // 8)
    stream(warm, 8)
    ttft, tot, n, usage = stream(prompt, 48)
    pt = usage["prompt_tokens"] if usage else want
    if ttft is None:
        print(f"{pt:10d} {'no tokens returned':>38}", flush=True); continue
    dec = (n-1)/max(tot-ttft,1e-9) if n>1 else float("nan")
    print(f"{pt:10d} {ttft:8.2f} {pt/ttft:14.0f} {dec:13.1f}", flush=True)
print("BENCH_DONE")
