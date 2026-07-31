#!/usr/bin/env python3
# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ПОИСК КОЛЕНА в выводе стендов глубины тени. Колено определяется ОБЪЕКТИВНО и одинаково для всех
# хозяев: наибольшая порция наездника, при которой время хозяина ещё в пределах допуска от базы.
# Допуск -- не "красивое число", а разброс самого стенда (медиана по 9-11 чередующимся проходам).
#
#   python tools/shadow_knee.py <файл-вывода> [допуск, по умолчанию 0.015]
#
# Печатает по каждому ВИДУ наездника: колено (в порциях и в инструкциях), время на этом колене,
# и НАКЛОН после колена -- цену одной инструкции в наносекундах хозяина, чтобы наездника за коленом
# можно было считать деньгами, а не отвергать.
import re, sys, collections

def parse(path):
    rows = []
    for ln in open(path):
        if ln.startswith('#') or not ln.strip():
            continue
        m = re.match(r"(\S+(?:\s\S+)*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?\s*$", ln.rstrip())
        if m:
            rows.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)), float(m.group(4)),
                         float(m.group(5)) if m.group(5) else None))
    return rows

def main():
    path = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.015
    rows = parse(path)
    if not rows:
        print("нет строк"); return
    base = rows[0][1]
    groups = collections.OrderedDict()
    for nm, us, c3, c4, c5 in rows[1:]:
        kind = re.sub(r"[\dx]+$", "", nm.split('/')[0] if '/' in nm else nm.split()[0]).strip()
        amount = None
        m = re.search(r"R=(\d+)", nm) or re.search(r"x(\d+)", nm) or re.search(r"(\d+)$", nm)
        if m: amount = int(m.group(1))
        tag = nm.split('/')[1] if '/' in nm else ''
        ilp = re.search(r"ilp(\d)", nm)
        key = kind + (('/ilp' + ilp.group(1)) if ilp else '')
        groups.setdefault(key, []).append((amount, us))
    print(f"# база {base:.3f} мкс, допуск {tol*100:.1f}%")
    print(f"{'вид наездника':<20}{'колено, порций':>15}{'время на колене':>17}{'наклон за коленом':>20}")
    for k, v in groups.items():
        v = sorted(x for x in v if x[0] is not None)
        knee, knee_us = 0, base
        for a, us in v:
            if us <= base * (1 + tol): knee, knee_us = a, us
            else: break
        after = [(a, us) for a, us in v if a > knee]
        slope = ""
        if len(after) >= 2:
            (a1, u1), (a2, u2) = after[0], after[-1]
            if a2 > a1: slope = f"{(u2-u1)/(a2-a1)*1000:.1f} нс/порцию"
        elif len(after) == 1:
            slope = f"{(after[0][1]-knee_us)/(after[0][0]-knee)*1000:.1f} нс/порцию" if after[0][0] > knee else ""
        print(f"{k:<20}{knee:>15}{knee_us:>15.3f} мкс{slope:>20}")

if __name__ == "__main__":
    main()
