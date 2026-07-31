#!/usr/bin/env python3
# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ПРОВЕРКА КОМПИЛЯТОРА ОДНОЙ КОМАНДОЙ. Правило сессии: смотреть SASS СРАЗУ после сборки, а не после
# того, как замер разочаровал. За эту сессию компилятор трижды сделал не то, что просили, и каждый раз
# это было видно в SASS и невидимо в исходнике:
#   1. получив 26 свободных регистров, ужался до 128 ради 4 блоков/SM и добавил 24 Б разлива (-7%);
#   2. при `минимум блоков = 2` на другой геометрии — снова 128 регистров и 32 Б разлива;
#   3. гистограмма опкодов ввела в заблуждение: 127 из 184 IMAD оказались MOV/SHL/IADD, пущенными
#      через конвейер IMAD ради второго порта выдачи, то есть НЕ арифметикой.
#
#   python tools/sass_check.py <файл.o|бинарник> [--kernel РЕГВЫР]
#
# Печатает по каждому ядру: регистры/разлив (если есть в cuobjdump -res-usage), долю HMMA, IMAD С
# РАЗБОРОМ ПО СУФФИКСАМ, ширины LDS/STS, барьеры — и ЯВНО отмечает известные патологии.
import re, subprocess, sys, collections, shutil, os

CUOBJ = shutil.which("cuobjdump")
if not CUOBJ:
    cuda_home = os.environ.get("CUDA_HOME", "")
    candidate = os.path.join(cuda_home, "bin", "cuobjdump") if cuda_home else ""
    if candidate and os.path.isfile(candidate):
        CUOBJ = candidate

def dump(path, flag):
    try:
        return subprocess.run([CUOBJ, flag, path], capture_output=True, text=True).stdout
    except FileNotFoundError:
        sys.exit("cuobjdump не найден; добавьте его в PATH или задайте CUDA_HOME")

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__ or "укажите файл")
    path = sys.argv[1]
    kre = None
    if "--kernel" in sys.argv:
        kre = re.compile(sys.argv[sys.argv.index("--kernel") + 1])
    sass = dump(path, "-sass")
    res = dump(path, "-res-usage")
    # регистры/разлив по имени ядра
    usage = {}
    cur = None
    for ln in res.splitlines():
        m = re.search(r"Function\s+([^:]+):", ln)          # " Function _Z...:"
        if m: cur = m.group(1).strip()
        m = re.search(r"REG:(\d+)\s+STACK:(\d+)", ln)     # "  REG:142 STACK:0 SHARED:512 ..."
        if m and cur: usage[cur] = (int(m.group(1)), int(m.group(2)))
    bad = 0
    for f in re.split(r"Function : ", sass)[1:]:
        name = f.split("\n", 1)[0].strip()
        if kre and not kre.search(name): continue
        ops = re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\w+\s+)?([A-Z][A-Z0-9_.]*)", f)
        if not ops: continue
        base = collections.Counter(o.split(".")[0] for o in ops)
        if base.get("HMMA", 0) == 0 and base.get("IMMA", 0) == 0: continue
        tot = sum(base.values())
        imad = collections.Counter(o for o in ops if o.startswith("IMAD"))
        real = imad.get("IMAD", 0) + sum(v for k, v in imad.items() if "WIDE" in k)
        fake = sum(imad.values()) - real
        lds = collections.Counter(o for o in ops if o.startswith("LDS"))
        sts = collections.Counter(o for o in ops if o.startswith("STS"))
        reg, stack = usage.get(name, (None, None))
        print(f"\n{name[:78]}")
        print(f"  всего {tot}, HMMA {base.get('HMMA',0)} ({100*base.get('HMMA',0)/tot:.1f}%)"
              + (f", регистров {reg}, разлив {stack} Б" if reg is not None else ""))
        print(f"  IMAD всего {sum(imad.values())}: НАСТОЯЩИХ умножений {real}, "
              f"замаскированных MOV/SHL/IADD {fake}")
        print(f"  LDS {dict(sorted(lds.items()))}  STS {dict(sorted(sts.items()))}  BAR {base.get('BAR',0)}")
        # известные патологии
        if stack:
            print(f"  !! РАЗЛИВ {stack} Б — компилятор выбрал цель по блокам сам."
                  f" Задайте __launch_bounds__(потоки, блоков_на_SM) руками.")
            bad += 1
        narrow_st = sts.get("STS", 0)
        if narrow_st > sum(sts.values()) * 0.5 and sum(sts.values()) > 4:
            print(f"  !! {narrow_st} узких STS (2 Б) из {sum(sts.values())} — записи не упакованы.")
            bad += 1
        if base.get("LDS", 0) > 2 * base.get("HMMA", 1):
            print(f"  !! LDS {base['LDS']} против HMMA {base['HMMA']} — подача дороже математики.")
            bad += 1
    print("\nОК: явных патологий не найдено" if bad == 0 else f"\nНАЙДЕНО ПАТОЛОГИЙ: {bad}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
