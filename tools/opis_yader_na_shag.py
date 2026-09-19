import gzip, json, collections, glob, sys
p = sorted(glob.glob(sys.argv[1] + "/*rank-0*.json.gz"))[-1]
d = json.load(gzip.open(p))
яд = collections.Counter(); зов = collections.Counter()
for e in d["traceEvents"]:
    if e.get("cat") == "kernel":
        яд[e["name"]] += e.get("dur", 0); зов[e["name"]] += 1
# число ШАГОВ декода: ядро словаря (lm_head) зовётся РОВНО раз на шаг цели
кандидаты = [(n, зов[n]) for n in зов if "turbomind" in n or "w8a16" in n]
# шагов = вызовов самого частого тела / число слоёв этого тела; оценим по gate_up (64 слоя)
всего = sum(яд.values())
гр = collections.Counter()
for имя, т in яд.items():
    к = ("тело tm8" if "turbomind" in имя else
         "тело gemv/m4" if "w8a16" in имя else
         "тело w12" if "w12a16" in имя else
         "тело cutlass/torch" if ("cutlass" in имя or "gemm" in имя.lower()) else
         "обмен ar2" if имя.startswith("ar2_") else
         "NCCL" if "nccl" in имя.lower() else
         "GDN" if ("gdn" in имя or "conv_upd" in имя or "chunk" in имя) else
         "внимание" if ("hmma" in имя and "w8a16" not in имя) or "flash" in имя.lower() else
         "i4 черновика" if "i4_scores" in имя else
         "элементарные" if ("elementwise" in имя or "vectorized" in имя or "reduce" in имя) else
         "прочее")
    гр[к] += т
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
print(f"  ядер всего {всего/1000:.1f} мс за прогон; при N={N} шагов -> {всего/1000/N:.2f} мс/шаг")
for к, т in гр.most_common():
    print(f"    {к:20s} {т/1000/N:7.2f} мс/шаг  {т/всего*100:5.1f}%")
print("  --- топ-12 ядер")
for имя, т in яд.most_common(12):
    print(f"    {т/1000/N:7.2f} мс/шаг x{зов[имя]:5d}  {имя[:78]}")
