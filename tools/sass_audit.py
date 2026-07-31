#!/usr/bin/env python3
"""SASS audit for sm_70 kernels: find what ptxas did badly, or did not do at all.

Volta has no cp.async and no hardware prologue overlap, so a global load's latency (~400-800 cy) is
hidden only if ptxas puts enough independent work between the LDG and its first consumer. It very
often does not -- it schedules for register pressure first, and an LDG that is "close enough" to its
use costs the full miss latency every time. That distance is measurable, per instruction, right here.

Usage:  python tools/sass_audit.py <cubin-or-.so> [kernel-name-regex]

Reports, per kernel:
  * instruction mix by opcode class (what the kernel actually spends issue slots on);
  * LDG -> first-consumer distance, as a histogram plus the worst offenders. A distance of 0-4 means
    the warp stalls on that load for its full latency with nothing to overlap;
  * load width usage (LDG.128 vs .64 vs .32) -- narrow loads waste transactions and issue slots;
  * address-arithmetic overhead (IADD3/SHF/LEA per LDG) -- high values mean offsets are recomputed
    instead of folded into the load's immediate;
  * MOV density -- a proxy for register-allocator churn;
  * barriers.
"""
import re, subprocess, sys, collections

CLASSES = [
    ("HMMA",   r"^HMMA"),           ("LDG",  r"^LDG"),   ("STG", r"^STG"),
    ("LDS",    r"^LDS"),            ("STS",  r"^STS"),   ("LDL/STL(spill)", r"^(LDL|STL)"),
    ("FADD/FMUL/FFMA", r"^(FADD|FMUL|FFMA)"),
    ("IADD/IMAD/LEA",  r"^(IADD|IMAD|LEA)"),
    ("SHF/LOP/PRMT",   r"^(SHF|LOP|PRMT|BFE|BFI)"),
    ("MUFU(exp/rcp)",  r"^MUFU"),   ("SHFL", r"^SHFL"),
    ("MOV",    r"^MOV"),            ("BAR",  r"^BAR"),
    ("F2F/I2F/F2I",    r"^(F2F|I2F|F2I|I2I)"),
    ("ATOM/RED",       r"^(ATOM|RED)"),
    ("branch/pred",    r"^(BRA|BSSY|BSYNC|ISETP|FSETP|SEL|PLOP)"),
]

# Предикат перед опкодом (`@P0 LDG.E...`, `@!P3 BRA`) раньше уводил инструкцию в "other" -- из-за
# этого предикатные LDG вообще не считались, и "адресной арифметики на одну LDG" выходило абсурдным.
_PRED = re.compile(r"^@!?P[T0-9]+\s+")

def strip_pred(ins):
    return _PRED.sub("", ins).strip()

def classify(op):
    for name, pat in CLASSES:
        if re.match(pat, op):
            return name
    return "other"

def dump(path, kregex):
    out = subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True).stdout
    kernels, cur, body = {}, None, []
    for line in out.split("\n"):
        m = re.search(r"Function : (\S+)", line)
        if m:
            if cur and body: kernels[cur] = body
            cur, body = m.group(1), []
            continue
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(.*?);", line)
        if m and cur:
            body.append(strip_pred(m.group(1)))
    if cur and body: kernels[cur] = body
    if kregex:
        kernels = {k: v for k, v in kernels.items() if re.search(kregex, k)}
    return kernels

def dest_reg(ins):
    m = re.match(r"@?\S*\s*[A-Z0-9.]+\s+(R\d+)", ins)
    return m.group(1) if m else None

def regs_used(ins):
    # operands after the first comma
    body = ins.split(" ", 1)[1] if " " in ins else ""
    parts = body.split(",")
    return set(re.findall(r"\bR\d+\b", ",".join(parts[1:]))) if len(parts) > 1 else set()

def audit(name, ins_list):
    n = len(ins_list)
    mix = collections.Counter(classify(i.split()[0].lstrip("@!PT ")) for i in ins_list)
    print(f"\n=== {name[:110]}")
    print(f"  инструкций: {n}")
    print("  состав:")
    for k, c in mix.most_common():
        if c * 200 >= n or c > 50:
            print(f"    {k:<18} {c:>7}  {100.0*c/n:>5.1f}%")

    # LDG -> первый потребитель
    dists, worst = [], []
    for i, ins in enumerate(ins_list):
        if not ins.split()[0].lstrip("@!PT ").startswith("LDG"):
            continue
        d = dest_reg(ins)
        if not d: continue
        for j in range(i + 1, min(i + 400, n)):
            if d in regs_used(ins_list[j]):
                dists.append(j - i); worst.append((j - i, i, ins[:70]))
                break
        else:
            dists.append(400)
    if dists:
        buckets = [(0, 4), (5, 16), (17, 64), (65, 200), (201, 10**9)]
        print(f"  расстояние LDG -> первое использование ({len(dists)} загрузок):")
        for lo, hi in buckets:
            c = sum(1 for x in dists if lo <= x <= hi)
            tag = "  <-- задержка НЕ спрятана" if lo == 0 and c else ""
            print(f"    {lo:>4}-{hi if hi < 10**9 else '400+':<5} {c:>6}  {100.0*c/len(dists):>5.1f}%{tag}")
        print(f"    медиана {sorted(dists)[len(dists)//2]}")
    # ширина загрузок
    w = collections.Counter()
    for ins in ins_list:
        m = re.match(r"@?\S*\s*LDG\.E(\.\w+)?(\.128|\.64)?", ins)
        if ins.split()[0].lstrip("@!PT ").startswith("LDG"):
            w["128" if ".128" in ins else ("64" if ".64" in ins else "32")] += 1
    if w:
        print(f"  ширина LDG: " + ", ".join(f"{k}бит={v}" for k, v in sorted(w.items())))
    ldg = mix.get("LDG", 0)
    if ldg:
        addr = mix.get("IADD/IMAD/LEA", 0) + mix.get("SHF/LOP/PRMT", 0)
        print(f"  адресной арифметики на одну LDG: {addr/ldg:.1f}")
    print(f"  MOV на 100 инструкций: {100.0*mix.get('MOV',0)/n:.1f}   "
          f"спиллов (LDL/STL): {mix.get('LDL/STL(spill)',0)}   барьеров: {mix.get('BAR',0)}")

if __name__ == "__main__":
    path = sys.argv[1]
    kregex = sys.argv[2] if len(sys.argv) > 2 else None
    ks = dump(path, kregex)
    if not ks:
        print("ядра не найдены"); sys.exit(1)
    for name, body in sorted(ks.items(), key=lambda kv: -len(kv[1]))[:4]:
        audit(name, body)
