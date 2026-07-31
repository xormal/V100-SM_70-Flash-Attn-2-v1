import torch, glob, os, sys
D=sys.argv[1]; dev="cuda"
kvs=sorted(glob.glob(os.path.join(D,"kv_layer*.pt")))
seen=set(); sel=[]
for f in kvs:
    tag=os.path.basename(f).split("_")[2]; li=os.path.basename(f).split("_")[1]
    if li in seen: continue
    seen.add(li); sel.append((li,f,os.path.join(D,f"q_{tag}")))
S=64
KEEP=[0.5,0.2,0.1,0.05,0.02]
print(f"ORACLE block selection (upper limit of ANY skipping scheme), s={S}: rank blocks by their TRUE mass (unimplementable;")
print("it is the ceiling), keep the top fraction, read only those. Error vs exact fp32.")
print("Reference points: e4m3 KV quantisation itself is 3.6e-4; fp16 attention is ~2.5e-4.")
hdr=" ".join(f"{int(k*100):>7}%" for k in KEEP)
print(f"{'layer':>7} | {hdr}   (relative L2 error of the head output)")
agg=[]
for li,fk,fq in sel:
    if not os.path.exists(fq): continue
    dk=torch.load(fk, map_location=dev, weights_only=False); dq=torch.load(fq, map_location=dev, weights_only=False)
    K=dk["k"].float(); V=dk["v"].float(); Q=dq["q"].float()
    T,Hkv,d=K.shape; nq,H,_=Q.shape; a=d**-0.5; grp=H//Hkv; nb=T//S
    errs={k:[] for k in KEEP}
    for hq in range(0,H,3):
        h=hq//grp
        Kh=K[:nb*S,h].contiguous(); Vh=V[:nb*S,h].contiguous()
        Kb=Kh.view(nb,S,d); mu=Kb.mean(1)
        r=(Kb-mu[:,None,:]).norm(dim=-1).amax(1)
        for qi in range(0,nq,32):
            q=Q[qi,hq]
            sc=a*(Kh@q); m=sc.max(); w=(sc-m).exp(); Z=w.sum()
            o=(w[:,None]*Vh).sum(0)/Z
            mass=w.view(nb,S).sum(1)
            order=torch.argsort(mass, descending=True)   # ORACLE ranking: true per-block mass
            for k in KEEP:
                nk=max(1,int(round(nb*k)))
                kept=torch.zeros(nb,dtype=torch.bool,device=dev); kept[order[:nk]]=True
                kk=kept.repeat_interleave(S).float()
                w2=w*kk; s2=w2.sum()
                o2=(w2[:,None]*Vh).sum(0)/s2 if s2>0 else torch.zeros_like(o)
                errs[k].append(float((o2-o).norm()/o.norm()))
    row=[sum(errs[k])/len(errs[k]) for k in KEEP]
    agg.append(row)
    print(f"{li:>7} | " + " ".join(f"{x:8.2e}" for x in row), flush=True)
    del K,V,Q; torch.cuda.empty_cache()
print("\nMEAN over layers:")
m=[sum(c)/len(c) for c in zip(*agg)]
for k,x in zip(KEEP,m):
    print(f"  keep {int(k*100):3d}% of BEST blocks -> read {1/k:5.1f}x less, relative error {x:.2e}")
print("ORACLE_DONE")
