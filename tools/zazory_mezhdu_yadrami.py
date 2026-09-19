import gzip, json, glob, sys, collections
p = sorted(glob.glob(sys.argv[1] + "/*rank-0*.json.gz"))[-1]
d = json.load(gzip.open(p))
яд = []
for e in d["traceEvents"]:
    if e.get("cat") == "kernel":
        яд.append((e["ts"], e.get("dur", 0), e["name"]))
яд.sort()
# берём участок ДЕКОДА: после последнего крупного ядра префилла (>2000 мкс)
посл_преф = 0
for i, (ts, dur, n) in enumerate(яд):
    if dur > 2000: посл_преф = i
яд = яд[посл_преф + 1:]
print(f"  ядер в участке декода: {len(яд)}")
# длительность участка и сумма ядер
t0 = яд[0][0]; t1 = max(ts + dur for ts, dur, n in яд)
сумма = sum(dur for ts, dur, n in яд)
print(f"  окно {(t1-t0)/1000:.1f} мс, сумма длительностей ядер {сумма/1000:.1f} мс")
# зазоры по ОДНОМУ потоку: склеиваем по перекрытию
пред_кон = None; зазоры = collections.Counter(); зазор_всего = 0.0; n_заз = 0
крупные = []
for ts, dur, n in яд:
    if пред_кон is not None and ts > пред_кон:
        g = ts - пред_кон
        зазор_всего += g; n_заз = n_заз + 1 if False else n_заз
        n_заз += 1
        зазоры[n[:60]] += g
        if g > 50: крупные.append((g, n[:70]))
    пред_кон = max(пред_кон or 0, ts + dur)
print(f"  ЗАЗОРОВ: {n_заз} штук, всего {зазор_всего/1000:.1f} мс ({зазор_всего/(t1-t0)*100:.1f}% окна)")
print(f"  средний зазор {зазор_всего/max(n_заз,1):.1f} мкс")
print("  --- куда уходят зазоры (ядро ПОСЛЕ зазора):")
for n, g in зазоры.most_common(10):
    print(f"    {g/1000:7.2f} мс  перед  {n}")
print(f"  --- крупных зазоров (>50 мкс): {len(крупные)}")
for g, n in sorted(крупные, reverse=True)[:6]:
    print(f"    {g:7.1f} мкс перед {n}")
