# Доля удержанной массы софтмакса (по 16 зондам плитки): Σ_kept E / Σ_all E. Аргументы: каталог дампа [N префикса]
import torch, glob, sys, collections, statistics
d=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 251580
depths=[0.04+i*(0.92/11) for i in range(12)]; pos=[int(dp*N) for dp in depths]
имена="alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
files=sorted(glob.glob(d+"/sel_*.pt")); всё=[]; игл=collections.defaultdict(list); первый=None
for f in files:
    o=torch.load(f)
    if "stat" not in o: continue
    st=o["stat"]; r=(st[...,0]/st[...,1].clamp_min(1e-30)).flatten().tolist(); всё+=r
    kv0,Tg,TI=o["kv0"],o["Tg"],o["TI"]
    for i,p in enumerate(pos):
        if kv0 <= p < Tg:
            t=min(st.shape[0]-1,(p-kv0)//TI); игл[i]+=(st[t,:,0]/st[t,:,1].clamp_min(1e-30)).tolist()
if not всё: print("дампов с массой нет"); sys.exit()
всё.sort(); n=len(всё)
print(f"плиток×kv-голов с массой: {n}")
print(f"  удержанная масса: медиана {100*всё[n//2]:.1f} %, среднее {100*statistics.mean(всё):.1f} %, 10-й перцентиль {100*всё[n//10]:.1f} %, 1-й {100*всё[n//100]:.1f} %, минимум {100*всё[0]:.2f} %")
for i in range(12):
    v=игл.get(i,[])
    if v: print(f"  {i+1:2d}. {имена[i]:8s}: плитка иглы -- удержано в среднем {100*statistics.mean(v):.1f} %, минимум по слоям {100*min(v):.1f} %")
