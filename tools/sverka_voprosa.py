# Плитка ВОПРОСА: дампы идут по порядку; новый запрос начинается, когда Tg падает. В каждом запросе
# берём чанк с максимальным Tg (последний, с вопросом) и его ПОСЛЕДНЮЮ плитку. Аргументы: каталог, N префикса.
import torch, glob, sys, collections
d=sys.argv[1]; N=int(sys.argv[2]); depths=[0.04+i*(0.92/11) for i in range(12)]
pos=[int(dp*N) for dp in depths]; имена="alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
files=sorted(glob.glob(d+"/sel_*.pt")); reqs=[]; prev=-1; cur=None
for f in files:
    o=torch.load(f)
    if o["Tg"] < prev - 100:            # Tg упал -> новый запрос
        if cur: reqs.append(cur)
        cur=[]
    if cur is None: cur=[]
    cur.append(o); prev=o["Tg"]
if cur: reqs.append(cur)
print(f"запросов (по падению Tg): {len(reqs)}; первый = префикс ({reqs[0][-1]['Tg']} ткн), далее вопросы")
B=reqs[0][0]["B"]
print(f"  {'запрос':10s} " + " ".join(f"{n[:3]:>4s}" for n in имена) + "   доля  Tg")
for k,r in enumerate(reqs[1:],1):
    mx=max(o["Tg"] for o in r); last=[o for o in r if o["Tg"]==mx and "idx" in o]
    if not last: print(f"  вопрос {k}: нет idx (Tg={mx})"); continue
    hits=[0]*12; tot=0; доля=0.0
    for o in last:
        idx=o["idx"]; cnt=o["cnt"]; tiles,Hkv,NB=idx.shape; t=tiles-1
        for hk in range(Hkv):
            kept=set(idx[t,hk,:int(cnt[t,hk])].tolist()); tot+=1; доля+=float(cnt[t,hk])/max(1,(o["Tg"]//B))
            for i,p in enumerate(pos):
                J=p//B
                if J in kept or (J+1) in kept: hits[i]+=1
    print(f"  {('вопрос '+str(k)):10s} " + " ".join(f"{100*h/max(1,tot):4.0f}" for h in hits) + f"   {100*доля/max(1,tot):4.1f}%  {mx}")
print("  (числа = % пар слой×kv-голова, где блок иглы или следующий выбран плиткой вопроса; вопрос k спрашивает иглу k)")
