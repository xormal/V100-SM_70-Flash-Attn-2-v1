import json, os, time, sys, urllib.request, random, threading
PORT=int(sys.argv[1]); CTX=int(sys.argv[2]); BATCH=[int(x) for x in sys.argv[3].split(",")]
B=os.environ.get("SERVER_URL", f"http://localhost:{PORT}").rstrip("/")
M=json.loads(urllib.request.urlopen(B+"/v1/models").read())["data"][0]["id"]
random.seed(7)
BASE="lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore".split()
VOCAB=[BASE[random.randrange(len(BASE))] for _ in range(900000)]
def run(prompt, ntok, out, i):
    body={"model":M,"prompt":prompt,"max_tokens":ntok,"temperature":0,"stream":True,
          "stream_options":{"include_usage":True}}
    req=urllib.request.Request(B+"/v1/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); ttft=None; n=0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line=line.strip()
            if not line.startswith(b"data: "): continue
            p=line[6:]
            if p==b"[DONE]": break
            d=json.loads(p); ch=d.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None: ttft=time.perf_counter()-t0
                n+=1
    out[i]=(ttft, time.perf_counter()-t0, n)
TPW=1.121; words=int(CTX/TPW); NTOK=int(sys.argv[4]) if len(sys.argv)>4 else 48
print(f"ctx~{CTX} tok, {NTOK} decode tokens each")
print(f"{'batch':>6} {'aggregate decode tok/s':>24} {'per-seq tok/s':>14} {'end-to-end t/s':>14}")
off=0
for bs in BATCH:
    prompts=[]
    for _ in range(bs):
        prompts.append(" ".join(VOCAB[off:off+words])); off+=words
    out=[None]*bs
    ths=[threading.Thread(target=run, args=(prompts[i], NTOK, out, i)) for i in range(bs)]
    t0=time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall=time.perf_counter()-t0
    ok=[o for o in out if o]
    ttft=max(o[0] for o in ok); tot=max(o[1] for o in ok); n=sum(o[2] for o in ok)
    dec=(n-len(ok))/max(tot-ttft,1e-9)
    e2e=n/wall
    print(f"{bs:6d} {dec:24.1f} {dec/bs:14.1f} {e2e:14.1f}", flush=True)
print("BATCH_DONE")
