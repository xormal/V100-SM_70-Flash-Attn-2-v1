import gzip, json, collections, glob, sys
p=sorted(glob.glob(sys.argv[1]+"/*rank-0*.json.gz"))[-1]
d=json.load(gzip.open(p)); яд=collections.Counter(); зов=collections.Counter()
for e in d["traceEvents"]:
    if e.get("cat")=="kernel": яд[e["name"]]+=e.get("dur",0); зов[e["name"]]+=1
N=60; всего=sum(яд.values())
гр={"тело tm8":0,"тело gemv":0,"тело w12":0,"тело cutlass":0,"обмен ar2":0,"NCCL":0,
    "GDN":0,"внимание":0,"i4 черновика":0,"прочее":0}
for имя,т in яд.items():
    к=("тело tm8" if "turbomind" in имя else "тело gemv" if "w8a16" in имя else
       "тело w12" if "w12a16" in имя else "тело cutlass" if "cutlass" in имя else
       "обмен ar2" if имя.startswith("ar2_") else "NCCL" if "nccl" in имя else
       "GDN" if ("gdn" in имя or "conv_upd" in имя) else
       "внимание" if ("hmma" in имя and "w8a16" not in имя) else
       "i4 черновика" if "i4_scores" in имя else "прочее")
    гр[к]+=т
print(f"  ВСЕГО ядер {всего/1000/N:.2f} мс/шаг")
for к,т in sorted(гр.items(),key=lambda x:-x[1]):
    if т: print(f"    {к:16s} {т/1000/N:6.2f} мс  {т/всего*100:5.1f}%")
