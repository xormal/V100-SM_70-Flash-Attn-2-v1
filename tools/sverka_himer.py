import torch, importlib.util, math, sys
def load(path):
    sp=importlib.util.spec_from_file_location("fa2sm70_decode", path); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
prod=load("/mnt/d1/alex/VLLM_ch/run/.fa2sm70_build_prod38abl/fa2sm70_decode/fa2sm70_decode.so")
import os; dev=load(os.environ.get("FA2SM70_DEC_SO","/tmp/claude-1000/-mnt-d1-alex-VLLM-fa2/44a942db-7eb7-4d06-ad86-a66ac5ce5f04/scratchpad/build_dev/fa2sm70_decode/fa2sm70_decode.so"))
H,Hkv,D,bs=12,2,256,16384; Sk=int(sys.argv[1]) if len(sys.argv)>1 else 65536; qGrp=4; ns=120; nb=Sk//bs; B=qGrp; scale=1/math.sqrt(D)
torch.manual_seed(1)
k=torch.randint(-127,128,(nb,bs,Hkv,D),dtype=torch.int8,device="cuda"); v=torch.randint(-127,128,(nb,bs,Hkv,D),dtype=torch.int8,device="cuda")
ks=(torch.rand(nb*bs*Hkv,device="cuda")*0.004+0.002); vs=(torch.rand(nb*bs*Hkv,device="cuda")*0.004+0.002)
bt=torch.arange(nb,dtype=torch.int32,device="cuda").view(1,nb); sl=torch.tensor([Sk],dtype=torch.int32,device="cuda")
q4=torch.randn(B,H,1,D,device="cuda").half()
outs=[]
for m in (prod,dev):
    A=torch.zeros(B*H*ns*D,device="cuda"); Bw=torch.zeros(2*B*H*ns,device="cuda"); out=torch.zeros(B,H*D,device="cuda").half()
    m.flash_decode_hmma_paged(q4,k,v,bt,bs,scale,ns,sl,A,Bw,out,H*D,ks,vs,qGrp,1); torch.cuda.synchronize(); outs.append(out.clone())
a,b=outs; d=(a.float()-b.float())
print(f"Sk={Sk}: max|prod-dev|={float(d.abs().max()):.3e}  relL2={float(d.norm()/a.float().norm()):.3e}  побайтово={'ДА' if torch.equal(a,b) else 'нет'}")
