# tm8 (turbomind s884, боевой слепок) при большом M: КПД тел префилла по вариантам
import torch, importlib.util, time, sys
def load(name):
    so=f"/mnt/d1/alex/VLLM_ch/run/.fa2sm70_build_prod38abl/{name}/{name}.so"
    sp=importlib.util.spec_from_file_location(name, so); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
tm8=load("fa2sm70_tm8")
M=int(sys.argv[1]) if len(sys.argv)>1 else 3072
shapes=[(5120,17408),(5120,8192),(3072,5120),(8704,5120)]
g=128; ПИК=125.0
def мер(fn, сек=0.6):
    fn(); torch.cuda.synchronize(); n=0; t=time.perf_counter()
    while time.perf_counter()-t<сек: fn(); n+=1
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n
print(f"M={M}  (такт см. nvidia-smi)")
for K,N in shapes:
    w=torch.randint(0,256,(N,K),dtype=torch.int16,device="cuda")
    sc=(torch.rand(K//g,N,device="cuda")*0.01+0.001).half(); z=torch.full((K//g,N),128.0,device="cuda").half()
    tw,ts,meta=tm8.prepare(w,sc,z,g); k_ld,q_ld=int(meta[0]),int(meta[1])
    x=(torch.randn(M,K,device="cuda")*0.1).half(); y=torch.empty(M,N,device="cuda").half()
    wf=(torch.randn(K,N,device="cuda")*0.01).half()
    dt_cb=мер(lambda: torch.mm(x,wf)); tf_cb=2*M*N*K/dt_cb/1e12
    res=[]
    for var in range(18):
        try: tm8.gemm(y,x,tw,ts,g,k_ld,q_ld,var,1,0); torch.cuda.synchronize()
        except Exception as e: continue
        dt=мер(lambda: tm8.gemm(y,x,tw,ts,g,k_ld,q_ld,var,1,0)); res.append((2*M*N*K/dt/1e12,var,dt))
    res.sort(reverse=True)
    top=" ".join(f"v{v}:{tf:.1f}" for tf,v,dt in res[:5])
    print(f"  K={K:5d} N={N:5d}: cuBLAS fp16 {tf_cb:5.1f} TFLOP/s ({tf_cb/ПИК*100:.0f}%) | tm8 лучшие: {top}  | худший v{res[-1][1]}:{res[-1][0]:.1f}")
