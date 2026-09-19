# Межслойные дельты KV по дампу первой страницы (16384 ткн) всех слоёв внимания. Аргумент: каталог kv_dump.
# Для соседних слоёв (a -> b): косинус; остаток после лучшей линейной карты P (МНК по половине токенов, проверка на другой);
# битность: ошибка квантования int4/int3/int2 остатка (поблочные шкалы по 64 ткн) против прямого int4 K_b и против int8 (эталон хранения).
import torch, glob, sys, math
d=sys.argv[1]; files=sorted(glob.glob(d+"/kv_*.pt")); print(f"слоёв: {len(files)}")
def deq(o, kind):
    x=o[kind].float(); sc=o[kind+"s"].float()          # x [bs, Hkv, D]; sc [bs*Hkv]
    bs,Hkv,D=x.shape; return x*sc.view(bs,Hkv,1)
def quant_err(x, bits, blk=64):
    # симметричное квантование по блокам [blk, D] на голову: относительная L2-ошибка
    bs,Hkv,D=x.shape; nb=bs//blk; xb=x[:nb*blk].view(nb,blk,Hkv,D)
    qmax=2**(bits-1)-1; s=xb.abs().amax(dim=(1,3),keepdim=True).clamp_min(1e-12)/qmax
    q=(xb/s).round().clamp(-qmax,qmax)*s
    return float(((q-xb).norm()/xb.norm()))
layers=[torch.load(f) for f in files]
print(f"  {'пара':7s} {'вид':2s} {'cos':>6s} {'ост/P %':>8s} | ошибка: {'K int8':>7s} {'K int4':>7s} {'ост int4':>8s} {'ост int3':>8s} {'ост int2':>8s}")
for kind in ("k","v"):
    for i in range(len(layers)-1):
        A=deq(layers[i],kind); Bt=deq(layers[i+1],kind)   # [bs,Hkv,D]
        bs,Hkv,D=A.shape; cos=[]; res_rel=[]; R_all=[]
        for h in range(Hkv):
            a=A[:,h,:]; b=Bt[:,h,:]
            cos.append(float((a*b).sum(-1).div(a.norm(dim=-1)*b.norm(dim=-1)+1e-9).mean()))
            n=bs//2; P=torch.linalg.lstsq(a[:n], b[:n]).solution        # [D,D]: b ≈ a P
            R=b[n:]-a[n:]@P; res_rel.append(float(R.norm()/b[n:].norm())); R_all.append(R)
        R=torch.stack(R_all,1)                                           # [bs/2,Hkv,D]
        Bh=Bt[bs//2:]
        print(f"  {i:2d}->{i+1:<2d} {kind:2s} {sum(cos)/len(cos):6.3f} {100*sum(res_rel)/len(res_rel):8.1f} | "
              f"{100*quant_err(Bh,8):7.2f} {100*quant_err(Bh,4):7.2f} {100*quant_err(R,4):8.2f} {100*quant_err(R,3):8.2f} {100*quant_err(R,2):8.2f}")
print("  (ост/P -- энергия остатка после линейной карты, % от ||K_b||; ошибки -- относительная L2 в %; хранение сейчас int8)")
