# Тела префилла: tm8 (лучший вариант) против «деквант раз на слой (reconstruct) + cuBLAS fp16»
import torch, importlib.util, time, sys
def load(name):
    so=f"/mnt/d1/alex/VLLM_ch/run/.fa2sm70_build_prod38abl/{name}/{name}.so"
    sp=importlib.util.spec_from_file_location(name, so); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
tm8=load("fa2sm70_tm8"); M=int(sys.argv[1]) if len(sys.argv)>1 else 3072; g=128
def мер(fn, сек=0.6):
    fn(); torch.cuda.synchronize(); n=0; t=time.perf_counter()
    while time.perf_counter()-t<сек: fn(); n+=1
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n
for K,N in [(5120,17408),(5120,8192),(3072,5120),(8704,5120)]:
    w=torch.randint(0,256,(N,K),dtype=torch.int16,device="cuda")
    sc=(torch.rand(K//g,N,device="cuda")*0.01+0.001).half(); z=torch.full((K//g,N),128.0,device="cuda").half()
    tw,ts,meta=tm8.prepare(w,sc,z,g); k_ld,q_ld=int(meta[0]),int(meta[1])
    qz=torch.full((K//g,N//4),0x80808080-2**32,dtype=torch.int64,device="cuda").to(torch.int32)  # GPTQ v2: нули как есть
    x=(torch.randn(M,K,device="cuda")*0.1).half(); y=torch.empty(M,N,device="cuda").half()
    Wt=torch.empty(N,K,device="cuda").half()
    best=None
    for var in (2,9,17,8,1,10):
        try: tm8.gemm(y,x,tw,ts,g,k_ld,q_ld,var,1,0); torch.cuda.synchronize()
        except Exception: continue
        dt=мер(lambda: tm8.gemm(y,x,tw,ts,g,k_ld,q_ld,var,1,0));
        if best is None or dt<best[0]: best=(dt,var)
    try:
        tm8.reconstruct(Wt,tw,qz,sc,N,K,True); torch.cuda.synchronize()
        dt_r=мер(lambda: tm8.reconstruct(Wt,tw,qz,sc,N,K,True))
        dt_mm=мер(lambda: torch.mm(x,Wt.t()))
        # сверка: y_tm8 (лучший) против x@Wt^T
        tm8.gemm(y,x,tw,ts,g,k_ld,q_ld,best[1],1,0); ref=torch.mm(x,Wt.t()); rel=float((y.float()-ref.float()).norm()/ref.float().norm())
        fl=2*M*N*K
        print(f"  K={K:5d} N={N:5d}: tm8 v{best[1]} {best[0]*1e3:6.2f} мс ({fl/best[0]/1e12:5.1f} TF) | recon {dt_r*1e3:5.2f} + cuBLAS {dt_mm*1e3:6.2f} = {(dt_r+dt_mm)*1e3:6.2f} мс ({fl/(dt_r+dt_mm)/1e12:5.1f} TF) | x{best[0]/(dt_r+dt_mm):.2f} | relL2 {rel:.1e}")
    except Exception as e:
        print(f"  K={K} N={N}: reconstruct ОТКАЗ {type(e).__name__}: {str(e)[:120]}")
