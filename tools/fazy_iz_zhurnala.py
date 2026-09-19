import sys, re
print("  период   словарь   ЦЕЛЬ(карта)  черновик(карта)")
for s in sys.stdin:
    def f(k):
        m = re.search(re.escape(k) + r"=([0-9.]+)", s)
        return m.group(1) if m else "-"
    p = f("ПЕРИОД (вход->вход)")
    if p == "-" or float(p) < 30:
        continue
    print(f"  {p:>7s} {f('словарь (logits)'):>9s} {f('[карта] проход цели'):>11s} {f('[карта] проход черновика'):>13s}")
