#!/usr/bin/env python3
# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# СЧЁТЧИК ВЫДАЧ для стендов глубины тени. Бюджет тени меряется в ИНСТРУКЦИЯХ (EPILOG §1a.0), значит
# надо знать, сколько инструкций РЕАЛЬНО вставил компилятор на одну порцию наездника -- источник
# не годится, потому что ptxas сворачивает, размножает и пускает арифметику через чужие конвейеры.
#
#   python tools/shadow_sass.py <бинарник> [регвыр-ядра]
#
# Печатает по каждому ядру: всего инструкций, и разбивку по конвейерам sm_70
# (FMA / ALU / XU / LSU-глобальная / LSU-разделяемая / тензорный / управление).
import os, re, shutil, subprocess, sys, collections

CUOBJ = os.environ.get("CUOBJDUMP") or shutil.which("cuobjdump")

PIPE = [
    ("tensor", re.compile(r"^(HMMA|IMMA)")),
    ("fma",    re.compile(r"^(FFMA|FADD|FMUL|FSET|FSEL|FCHK|DADD|DMUL|DFMA)")),
    ("xu",     re.compile(r"^MUFU")),
    ("lsu_g",  re.compile(r"^(LDG|STG|RED|ATOM|LDL|STL|LD\b|ST\b)")),
    ("lsu_s",  re.compile(r"^(LDS|STS|ATOMS|SHFL|LDSM)")),
    ("alu",    re.compile(r"^(IADD|IMAD|LOP|SHF|PRMT|SEL|ISETP|I2F|F2I|F2F|I2I|MOV|BFE|FLO|POPC|VOTE|PLOP|LEA|SGXT|BREV|IABS|IMNMX|FMNMX|CS2R|S2R|R2P|P2R)")),
    ("ctrl",   re.compile(r"^(BRA|BRX|BSSY|BSYNC|EXIT|RET|CALL|NOP|BAR|MEMBAR|DEPBAR|YIELD|SSY|SYNC|PBK|BRK|JMP|WARPSYNC|ERRBAR)")),
]

def classify(op):
    for name, rx in PIPE:
        if rx.match(op):
            return name
    return "other"

def main():
    if not CUOBJ:
        raise SystemExit("cuobjdump not found; set CUOBJDUMP or add it to PATH")
    binpath = sys.argv[1]
    pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
    sass = subprocess.run([CUOBJ, "-sass", binpath], capture_output=True, text=True).stdout
    cur, per = None, collections.OrderedDict()
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            if pat and not pat.search(cur):
                cur = None
            elif cur:
                per[cur] = collections.Counter()
            continue
        if cur is None:
            continue
        # предикат отделять ЯВНО: шаблон "@?!?P?\d?" съедал букву P у самой инструкции (PRMT -> RMT)
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m:
            op = m.group(1).split('.')[0]
            per[cur]["TOTAL"] += 1
            per[cur][classify(op)] += 1
            per[cur]["op:" + op] += 1
    order = ["TOTAL", "fma", "alu", "xu", "lsu_g", "lsu_s", "tensor", "ctrl", "other"]
    print("%-58s %s" % ("ядро", "  ".join("%7s" % o for o in order)))
    for k, c in per.items():
        print("%-58s %s" % (k[:58], "  ".join("%7d" % c[o] for o in order)))
    return per

if __name__ == "__main__":
    main()
