# Свод последнего профиля rank-0 по именам ядер: мкс и доля. Аргументы: каталог [топ]
import gzip, json, glob, sys, collections, re
p=sorted(glob.glob(sys.argv[1]+"/*rank-0*.json.gz"))[-1]; top=int(sys.argv[2]) if len(sys.argv)>2 else 25
d=json.load(gzip.open(p)); acc=collections.Counter(); cnt=collections.Counter(); tot=0.0
for e in d["traceEvents"]:
    if e.get("cat")!="kernel": continue
    nm=e["name"]; nm=re.sub(r"<.*","",nm)[:60]; acc[nm]+=e["dur"]; cnt[nm]+=1; tot+=e["dur"]
print(f"файл {p.split('/')[-1]}  ядер всего {tot/1e3:.1f} мс")
for nm,us in acc.most_common(top): print(f"  {us/1e3:9.2f} мс {us/tot*100:5.1f}% {cnt[nm]:6d}x  {nm}")
