# Декодное внимание на боевых формах (int8-пул, страница 16384, Sk=262144, H=12/Hkv=2, d=256):
# скалярное ядро (_into, строки порознь) против ХИМЕРЫ (flash_decode_hmma_paged, qGrp строк вместе)
import torch, importlib.util, time, sys, math
import os
def load(name):
    so=os.environ.get("FA2SM70_DEC_SO") or f"/mnt/d1/alex/VLLM_ch/run/.fa2sm70_build_prod38abl/{name}/{name}.so"
    sp=importlib.util.spec_from_file_location(name, so); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
dec=load("fa2sm70_decode")
H,Hkv,D,bs=12,2,256,16384
Sk=int(sys.argv[1]) if len(sys.argv)>1 else 262144
qGrp=int(sys.argv[2]) if len(sys.argv)>2 else 4
ns=int(sys.argv[3]) if len(sys.argv)>3 else 120
nb=Sk//bs; B=qGrp; scale=1/math.sqrt(D)
k=torch.randint(-127,128,(nb,bs,Hkv,D),dtype=torch.int8,device="cuda"); v=torch.randint(-127,128,(nb,bs,Hkv,D),dtype=torch.int8,device="cuda")
ks=(torch.rand(nb*bs*Hkv,device="cuda")*0.004+0.002); vs=(torch.rand(nb*bs*Hkv,device="cuda")*0.004+0.002)
bt1=torch.arange(nb,dtype=torch.int32,device="cuda").view(1,nb); sl1=torch.tensor([Sk],dtype=torch.int32,device="cuda")
btB=bt1.expand(B,nb).contiguous(); slB=sl1.expand(B).contiguous()
q4=(torch.randn(B,H,1,D,device="cuda")).half()
A=torch.zeros(B*H*ns*D,device="cuda"); Bw=torch.zeros(2*B*H*ns,device="cuda")
outH=torch.zeros(B,H*D,device="cuda").half(); outS=torch.zeros(B,H*D,device="cuda").half()
kmax=torch.zeros((max(B,8),Hkv),device="cuda")
байт=2*Sk*Hkv*D  # K+V int8 на карту, полезное чтение
def мер(fn, сек=1.0):
    fn(); torch.cuda.synchronize(); n=0; t=time.perf_counter()
    while time.perf_counter()-t<сек: fn(); n+=1
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n
def химера(): Bw.zero_(); dec.flash_decode_hmma_paged(q4,k,v,bt1,bs,scale,ns,sl1,A,Bw,outH,H*D,ks,vs,qGrp,1)
def скаляр(): dec.flash_decode_defer_mqa_paged_into(q4,k,v,kmax,btB,bs,scale,ns,Sk,slB,False,1.0,1.0,A,Bw,outS,H*D,ks,vs,None)
print(f"Sk={Sk} qGrp={qGrp} ns={ns} страница={bs} полезно {байт/1e6:.0f} МБ на слой")
for имя,fn,out in (("скаляр(_into, строки порознь)",скаляр,outS),("ХИМЕРА(hmma, строки вместе)",химера,outH)):
    try:
        dt=мер(fn)
        print(f"  {имя:32s}: {dt*1e3:7.2f} мс/слой  = {байт/dt/1e9:6.0f} ГБ/с полезных  -> 16 слоёв {dt*16*1e3:6.1f} мс/шаг")
    except Exception as e:
        print(f"  {имя}: ОТКАЗ {type(e).__name__}: {str(e)[:160]}")
try:
    a=outS.float(); b=outH.float(); rel=float((a-b).norm()/a.norm()); print(f"  relL2(химера vs скаляр) = {rel:.2e}")
except Exception as e: print("  сверка не удалась", e)
