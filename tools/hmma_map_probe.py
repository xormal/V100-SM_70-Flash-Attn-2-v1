# ИЗМЕРЕНИЕ раскладки mma.sync.m8n8k4 на Volta: какой (lane, регистр) держит какой матричный элемент.
# Метод: единица в одну ячейку фрагмента A при B=1 подсвечивает ВСЮ строку m в выходе -> так
# определяются и строка каждого слота A, и строка каждого слота D. Симметрично для B и столбцов.
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

root = Path(__file__).resolve().parents[1]
SRC = (root / "tests/test_volta_hmma.py").read_text().split("SRC = r'''")[1].split("'''")[0]
mod = load_inline(name="hmma_map",
                  cpp_sources="torch::Tensor run(torch::Tensor A, torch::Tensor B);",
                  cuda_sources=SRC, functions=["run"], verbose=False,
                  extra_include_paths=[str(root / "fa2_src/fmha_kernel/gemm")],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
                  build_directory=os.environ["PROBE_BUILD"])
dev = "cuda"
ones = torch.ones(32, 4, device=dev, dtype=torch.float16)
zero = torch.zeros(32, 4, device=dev, dtype=torch.float16)

# --- шаг 1: слот A -> строка выхода; и какие слоты D принадлежат одной строке
a_rows = {}      # (lane,i) -> замороженное множество слотов D
for la in range(32):
    for ia in range(4):
        A = zero.clone(); A[la, ia] = 1.0
        D = mod.run(A, ones).cpu()
        hot = frozenset((int(x), int(y)) for x, y in (D != 0).nonzero().tolist())
        a_rows[(la, ia)] = hot
groups = {}
for slot, hot in a_rows.items():
    groups.setdefault(hot, []).append(slot)
print(f"=== слоты A: {len(a_rows)} штук сгруппировались в {len(groups)} различных строк выхода")
sizes = sorted({len(v) for v in groups.values()})
hot_sizes = sorted({len(k) for k in groups})
print(f"    слотов A на строку: {sizes};  слотов D в строке: {hot_sizes}")

# --- шаг 2: слот B -> столбец выхода
b_cols = {}
for lb in range(32):
    for ib in range(4):
        B = zero.clone(); B[lb, ib] = 1.0
        D = mod.run(ones, B).cpu()
        hot = frozenset((int(x), int(y)) for x, y in (D != 0).nonzero().tolist())
        b_cols[(lb, ib)] = hot
gb = {}
for slot, hot in b_cols.items():
    gb.setdefault(hot, []).append(slot)
print(f"=== слоты B: сгруппировались в {len(gb)} различных столбцов;  слотов D в столбце: {sorted({len(k) for k in gb})}")

# --- шаг 3: пересечение строки и столбца задаёт конкретный элемент D
rows = list(groups.keys()); cols = list(gb.keys())
cell = {}
for ri, r in enumerate(rows):
    for ci, c in enumerate(cols):
        inter = r & c
        if len(inter) == 1:
            cell[next(iter(inter))] = (ri, ci)
print(f"=== однозначно определено ячеек D: {len(cell)} из 256")
# --- выгрузка ПОЛНОЙ таблицы: она и есть то, на чём строится мейнлуп
out = os.environ.get("HMMA_MAP_OUT", "")
if out and len(cell) == 256:
    with open(out, "w") as f:
        f.write("# Измеренная раскладка mma.sync.m8n8k4 (sm_70), tools/hmma_map_probe.py\n")
        f.write("# Индексы строк/столбцов — порядковые номера ГРУПП, определённых зондированием.\n")
        f.write("# lane reg -> row col\n")
        for (l, r_), (m, n) in sorted(cell.items()):
            f.write(f"{l:>3} {r_:>3} {m:>4} {n:>4}\n")
        # раскладки операндов: слот -> группа
        f.write("\n# A: lane reg -> row\n")
        for (l, i), hot in sorted(a_rows.items()):
            f.write(f"{l:>3} {i:>3} {rows.index(hot):>4}\n")
        f.write("\n# B: lane reg -> col\n")
        for (l, i), hot in sorted(b_cols.items()):
            f.write(f"{l:>3} {i:>3} {cols.index(hot):>4}\n")
    print(f"    полная таблица записана: {out}")
if len(cell) >= 8:
    ex = sorted(cell.items())[:6]
    print("    примеры  (lane,reg) -> (строка, столбец):")
    for (l, r_), (m, n) in ex:
        print(f"      lane {l:>2}, reg {r_} -> D[{m:>2}][{n:>2}]")
