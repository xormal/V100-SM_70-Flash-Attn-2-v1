# Кумулятивный индексатор офлайн. Вход: дампы с m,S (чанки с иглами): для одного чанка 16 слоёв x (плитки, kv, NB).
# Правила при равной цене (доля блоков): (а) масса на слое; (б) объединение множеств слоёв; (в) накопленные оценки.
import torch, glob, sys, collections, statistics
d=sys.argv[1]; MASS=float(sys.argv[2]) if len(sys.argv)>2 else 0.90
files=sorted(glob.glob(d+"/sel_*.pt")); groups=collections.OrderedDict()
for f in files:
    o=torch.load(f)
    if "m" in o and "S" in o: groups.setdefault((o["kv0"],o["Tg"]), []).append(o)
print(f"чанков с сырыми оценками: {len(groups)} (слоёв в каждом: {sorted(set(len(g) for g in groups.values()))})")
def E_of(o):
    m,S=o["m"].float(),o["S"].float(); M=m.amax(-1,keepdim=True); return S*torch.exp(m-M)   # [tiles,Hkv,NB]
def mass_select(E, target):   # порог по массе на (плитка,kv): маска блоков
    tot=E.sum(-1,keepdim=True); srt,_=E.sort(-1,descending=True); cum=srt.cumsum(-1)
    k=(cum < target*tot).sum(-1,keepdim=True)+1             # число блоков до target
    thr=srt.gather(-1,(k-1).clamp(max=E.shape[-1]-1)); return E>=thr
res=collections.defaultdict(list)
for key,g in list(groups.items())[:6]:
    Es=[E_of(o) for o in g]; L=len(Es); tiles,Hkv,NB=Es[0].shape
    kv0,Tg=key; NBvis=min(NB,(Tg+63)//64)
    Es=[e[...,:NBvis] for e in Es]
    # (а) на слое
    sets=[mass_select(e,MASS) for e in Es]
    blocks_a=statistics.mean(float(s.float().mean()) for s in sets)
    # (б) объединение
    uni=torch.stack(sets).any(0); blocks_b=float(uni.float().mean())
    mass_b=[float(((e*uni).sum(-1)/e.sum(-1).clamp_min(1e-30)).median()) for e in Es]
    # (в) накопленные нормированные оценки; цена подбирается под (а): тот же средний размер набора
    En=torch.stack([e/e.sum(-1,keepdim=True).clamp_min(1e-30) for e in Es]).sum(0)
    # подбор target для En, дающий blocks ~ blocks_a
    lo,hi=0.5,0.999
    for _ in range(20):
        mid=(lo+hi)/2; s=mass_select(En,mid); b=float(s.float().mean())
        if b<blocks_a: lo=mid
        else: hi=mid
    acc=mass_select(En,lo); blocks_c=float(acc.float().mean())
    mass_c=[float(((e*acc).sum(-1)/e.sum(-1).clamp_min(1e-30)).median()) for e in Es]
    mass_a=[float(((e*s).sum(-1)/e.sum(-1).clamp_min(1e-30)).median()) for e,s in zip(Es,sets)]
    res["a"].append((blocks_a,min(mass_a),statistics.median(mass_a))); res["b"].append((blocks_b,min(mass_b),statistics.median(mass_b))); res["c"].append((blocks_c,min(mass_c),statistics.median(mass_c)))
for k,nm in (("a","(а) масса на слое"),("b","(б) объединение слоёв"),("c","(в) накопленные оценки, та же цена что (а)")):
    v=res[k]; print(f"  {nm:42s}: блоков {100*statistics.mean(x[0] for x in v):5.1f} %  масса по слоям: min {100*statistics.mean(x[1] for x in v):5.1f} %  медиана {100*statistics.mean(x[2] for x in v):5.1f} %")
