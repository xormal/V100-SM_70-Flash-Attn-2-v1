import torch, glob, os, sys
D=sys.argv[1]
files=sorted(glob.glob(os.path.join(D,"kv_layer*.pt")))
seen=set(); files=[f for f in files if (os.path.basename(f).split("_")[1] not in seen and not seen.add(os.path.basename(f).split("_")[1]))]
print(f"layers: {len(files)}")
print(f"{'layer':>5} {'obj':>3} {'blk':>6} | {'r=16':>7} {'r=32':>7} {'r=64':>7} {'r=128':>7}   (residual energy fraction, block-local basis)")
tot={}
for f in files:
    li=int(os.path.basename(f).split("_")[1].replace("layer",""))
    d=torch.load(f, map_location="cpu")
    for name in ("k","v"):
        X=d[name].float()                      # [T, Hkv, d]
        T,Hkv,dim=X.shape
        for s in (512,):                       # candidate block size
            res={r:[] for r in (16,32,64,128)}
            for h in range(Hkv):
                for b0 in range(0, T-s+1, s*4):    # sample every 4th block
                    Bk=X[b0:b0+s,h,:]              # [s, d]
                    Bk=Bk-Bk.mean(0, keepdim=True) # centred, unconstrained baseline
                    sv=torch.linalg.svdvals(Bk)
                    e=(sv**2); tot_e=e.sum()
                    for r in res: res[r].append(float(e[r:].sum()/tot_e))
            row=[sum(res[r])/len(res[r]) for r in (16,32,64,128)]
            print(f"{li:5d} {name:>3} {s:6d} | " + " ".join(f"{x:7.4f}" for x in row))
            tot.setdefault(name, []).append(row)
print("\nMEAN over layers (residual energy after rank r, block s=512, centred):")
for name in ("k","v"):
    m=[sum(c)/len(c) for c in zip(*tot[name])]
    print(f"  {name}: r=16 {m[0]:.4f}  r=32 {m[1]:.4f}  r=64 {m[2]:.4f}  r=128 {m[3]:.4f}")
print("SPECTRA_DONE")
