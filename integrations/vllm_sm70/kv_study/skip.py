import torch, glob, os, sys, math
D=sys.argv[1]; dev="cuda"
kvs=sorted(glob.glob(os.path.join(D,"kv_layer*.pt")))
seen=set(); sel=[]
for f in kvs:
    tag=os.path.basename(f).split("_")[2]          # cache ptr, identifies (layer, worker)
    li=os.path.basename(f).split("_")[1]
    if li in seen: continue
    seen.add(li); sel.append((li,f,os.path.join(D,f"q_{tag}")))
print(f"layers: {len(sel)}")
S=512
EPS=[1e-3, 1e-2]
print(f"block s={S}; skip fraction sigma (higher = less store read); error = ||o-o_exact||/||o_exact||")
print(f"{'layer':>5} | {'orac e-3':>9} {'cert e-3':>9} {'err e-3':>9} | {'orac e-2':>9} {'cert e-2':>9} {'err e-2':>9}")
agg={}
for li,fk,fq in sel:
    if not os.path.exists(fq): continue
    dk=torch.load(fk, map_location=dev, weights_only=False)
    dq=torch.load(fq, map_location=dev, weights_only=False)
    K=dk["k"].float(); V=dk["v"].float()            # [T,Hkv,d]
    Q=dq["q"].float()                               # [nq,H,d]
    T,Hkv,d=K.shape; nq,H,_=Q.shape
    a=d**-0.5; grp=H//Hkv
    nb=T//S
    row={}
    for eps in EPS:
        oracs=[]; certs=[]; errs=[]
        for hq in range(0, H, 3):                   # sample query heads
            h=hq//grp
            Kh=K[:nb*S,h].contiguous(); Vh=V[:nb*S,h].contiguous()
            Kb=Kh.view(nb,S,d); Vb=Vh.view(nb,S,d)
            mu=Kb.mean(1)                           # [nb,d]
            r=(Kb-mu[:,None,:]).norm(dim=-1).amax(1) # [nb]
            for qi in range(0, nq, 64):             # sample queries
                q=Q[qi,hq]
                sc=a*(Kh@q)                         # [nb*S] exact scores
                m=sc.max()
                w=(sc-m).exp(); Z=w.sum()
                o=(w[:,None]*Vh).sum(0)/Z
                mass=w.view(nb,S).sum(1)            # exact per-block mass (unnormalised)
                # ---- oracle: skip every block whose TRUE mass <= eps*Z
                oracs.append(float((mass <= eps*Z).float().mean()))
                # ---- certified: bound U_i = a(<q,mu_i> + ||q|| r_i); mass_i <= S*exp(a U_i - m)
                U=a*(mu@q + q.norm()*r)
                bnd=S*torch.exp(U-m)
                order=torch.argsort(bnd, descending=True)
                # keep the largest-bound blocks until the REMAINING bounded mass <= eps * kept EXACT mass
                kept=torch.zeros(nb, dtype=torch.bool, device=dev)
                acc=0.0; rem=float(bnd.sum())
                for idx in order.tolist():
                    if rem <= eps*acc and acc > 0: break
                    kept[idx]=True; acc+=float(mass[idx]); rem-=float(bnd[idx])
                certs.append(float((~kept).float().mean()))
                if kept.all(): errs.append(0.0)
                else:
                    kk=kept.repeat_interleave(S)
                    w2=w*kk.float()
                    o2=(w2[:,None]*Vh).sum(0)/w2.sum()
                    errs.append(float((o2-o).norm()/o.norm()))
        row[eps]=(sum(oracs)/len(oracs), sum(certs)/len(certs), sum(errs)/len(errs))
        agg.setdefault(eps, []).append(row[eps])
    print(f"{li:>5} | {row[1e-3][0]:9.3f} {row[1e-3][1]:9.3f} {row[1e-3][2]:9.2e} | "
          f"{row[1e-2][0]:9.3f} {row[1e-2][1]:9.3f} {row[1e-2][2]:9.2e}", flush=True)
    del K,V,Q; torch.cuda.empty_cache()
print("\nMEAN over layers:")
for eps in EPS:
    m=[sum(c)/len(c) for c in zip(*agg[eps])]
    print(f"  eps={eps:g}: oracle skip {m[0]:.3f}   CERTIFIED skip {m[1]:.3f}   rel.err {m[2]:.2e}"
          f"   -> read {1/max(1-m[1],1e-9):.2f}x less" if m[1]<1 else "")
print("SKIP_DONE")
