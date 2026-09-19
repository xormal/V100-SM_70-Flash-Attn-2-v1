# Сверка дампа отбора с позициями игл. Аргументы: каталог дампа, N токенов промпта.
# Для чанков с idx: плитка, содержащая иглу -> доля отобранных блоков и картина ±4 блока вокруг иглы.
import torch, glob, sys, collections
d=sys.argv[1]; N=int(sys.argv[2]); depths=[0.04+i*(0.92/11) for i in range(12)]
pos=[int(dp*N) for dp in depths]; имена="alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
files=sorted(glob.glob(d+"/sel_*.pt")); print(f"дампов: {len(files)}")
доли=[]; карта=collections.defaultdict(list)
for f in files:
    o=torch.load(f); cnt=o["cnt"]; B=o["B"]; TI=o["TI"]; Tg=o["Tg"]; kv0=o["kv0"]
    NBvis_last=(Tg+B-1)//B
    доли.append(float(cnt.float().mean())/max(1,NBvis_last))
    if "idx" not in o: continue
    idx=o["idx"]; tiles,Hkv,NB=idx.shape
    for i,p in enumerate(pos):
        if not (kv0 <= p < Tg): continue
        t=min(tiles-1,(p-kv0)//TI); J=p//B
        for hk in range(Hkv):
            kept=set(idx[t,hk,:int(cnt[t,hk])].tolist())
            окно="".join("#" if (J+o_) in kept else "." for o_ in range(-4,5))
            карта[i].append((float(cnt[t,hk])/max(1,(p//B)+1), окно, (J in kept)))
print(f"  средняя доля отобранных блоков по чанкам: {100*sum(доли)/max(1,len(доли)):.1f} % (min {100*min(доли):.1f}, max {100*max(доли):.1f})")
for i in range(12):
    v=карта.get(i,[])
    if not v: print(f"  {i+1:2d}. {имена[i]:8s} поз {pos[i]:6d}: дампа плитки нет"); continue
    свой=sum(1 for x in v if x[2]); ср=sum(x[0] for x in v)/len(v)
    print(f"  {i+1:2d}. {имена[i]:8s} поз {pos[i]:6d} блок {pos[i]//64:5d}: свой блок выбран {свой}/{len(v)} (слой×kv), доля в плитке {100*ср:4.1f} %, окно ±4: {v[0][1]} {v[len(v)//2][1]}")
