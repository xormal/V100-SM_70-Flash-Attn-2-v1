import torch, glob, os, sys
D=sys.argv[1]; dev="cuda"
kvs=sorted(glob.glob(os.path.join(D,"kv_layer*.pt")))
seen=set(); sel=[]
for f in kvs:
    tag=os.path.basename(f).split("_")[2]; li=os.path.basename(f).split("_")[1]
    if li in seen: continue
    seen.add(li); sel.append((li,f,os.path.join(D,f"q_{tag}")))
EPS=1e-2
print(f"eps={EPS}. sigma = fraction of blocks a CERTIFIED bound can skip. Two bounds:")
print("  ball      : <q,k_j> <= <q,mu> + ||q||*r          (r = block radius)")
print("  ellipsoid : <q,k_j> <= <q,mu> + ||P^T q||*rho + ||q||*e   (P = top-32 block basis)")
print(f"{'s':>5} | {'oracle':>7} {'ball':>7} {'ellip':>7} | {'read x less (ellip)':>20}")
for S in (64, 128, 256, 512):
    O=[];Bl=[];El=[]
    for li,fk,fq in sel[:8]:
        if not os.path.exists(fq): continue
        dk=torch.load(fk, map_location=dev, weights_only=False); dq=torch.load(fq, map_location=dev, weights_only=False)
        K=dk["k"].float(); Q=dq["q"].float()
        T,Hkv,d=K.shape; nq,H,_=Q.shape; a=d**-0.5; grp=H//Hkv; nb=T//S
        for hq in range(0,H,6):
            h=hq//grp
            Kb=K[:nb*S,h].contiguous().view(nb,S,d)
            mu=Kb.mean(1); C=Kb-mu[:,None,:]
            r=C.norm(dim=-1).amax(1)
            # per-block top-32 basis; rho = max coefficient norm, e = max residual norm
            R=32
            U_,S_,Vh_=torch.linalg.svd(C, full_matrices=False)
            P=Vh_[:,:R,:]                                   # [nb,R,d]
            co=torch.einsum('bsd,brd->bsr', C, P)
            rho=co.norm(dim=-1).amax(1)
            res=C-torch.einsum('bsr,brd->bsd', co, P)
            e=res.norm(dim=-1).amax(1)
            for qi in range(0,nq,128):
                q=Q[qi,hq]
                sc=a*(K[:nb*S,h]@q); m=sc.max(); w=(sc-m).exp(); Z=w.sum()
                mass=w.view(nb,S).sum(1)
                O.append(float((mass<=EPS*Z).float().mean()))
                for name,U in (("ball", a*(mu@q + q.norm()*r)),
                               ("ellip", a*(mu@q + torch.einsum('brd,d->br',P,q).norm(dim=-1)*rho + q.norm()*e))):
                    bnd=S*torch.exp(U-m)
                    order=torch.argsort(bnd, descending=True)
                    kept=torch.zeros(nb,dtype=torch.bool,device=dev); acc=0.0; rem=float(bnd.sum())
                    for idx in order.tolist():
                        if rem<=EPS*acc and acc>0: break
                        kept[idx]=True; acc+=float(mass[idx]); rem-=float(bnd[idx])
                    (Bl if name=="ball" else El).append(float((~kept).float().mean()))
        del K,Q; torch.cuda.empty_cache()
    o=sum(O)/len(O); b=sum(Bl)/len(Bl); el=sum(El)/len(El)
    print(f"{S:5d} | {o:7.3f} {b:7.3f} {el:7.3f} | {1/max(1-el,1e-9):20.2f}", flush=True)
print("SKIP2_DONE")
