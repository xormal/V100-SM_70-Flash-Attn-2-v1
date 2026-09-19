import torch, time, importlib.util, math, sys
so="/mnt/d1/alex/VLLM_ch/run/.fa2sm70_build_prod38abl/fa2sm70_prefill/fa2sm70_prefill.so"
spec=importlib.util.spec_from_file_location("fa2sm70_prefill", so); pf=importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
dev="cuda"; D=256; H=12; Hkv=2; scale=1.0/math.sqrt(D)   # TP=2: 12 q-голов, 2 kv на карту
def мер(Sk, сек=1.2):
    Sq=Sk
    Q=torch.randn(1,Sq,H,D,dtype=torch.float16,device=dev).contiguous()
    K=torch.randn(1,Hkv,Sk,D,dtype=torch.float16,device=dev).contiguous()
    V=torch.randn(1,Hkv,Sk,D,dtype=torch.float16,device=dev).contiguous()
    for _ in range(3): pf.attn_fwd_volta(Q,K,V,scale,True)
    torch.cuda.synchronize(); n=0; t=time.perf_counter()
    while time.perf_counter()-t<сек: pf.attn_fwd_volta(Q,K,V,scale,True); n+=1
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/n
    flop=2*2*(Sk*Sk/2)*D*H            # qk+pv, FMA=2, каузал /2, на карту (12 голов)
    return dt, flop/dt/1e12
ПИК=125.0  # V100 fp16 HMMA пик TFLOPS
print(f"  {'Sk':>7} {'мс/слой':>8} {'TFLOP/s':>8} {'КПД':>6}")
пары=[]
for Sk in (4096,8192,16384,32768):
    dt,tf=мер(Sk); пары.append((Sk,dt))
    print(f"  {Sk:>7} {dt*1e3:8.2f} {tf:8.1f} {tf/ПИК*100:5.1f}%")
# квадратичная экстраполяция по последней точке (самая насыщенная)
Sk0,dt0=пары[-1]
for L in (131072,262144):
    dt=dt0*(L/Sk0)**2
    print(f"  экстраполяция Sk={L}: {dt*1e3:.0f} мс/слой -> внимание {dt*16*1e3:.0f} мс = {dt*16:.1f} с на карту (16 слоёв)")
