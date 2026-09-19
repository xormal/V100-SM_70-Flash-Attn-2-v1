"""СКОЛЬКО РАЗРЯДОВ РЕАЛЬНО ЗАНЯТО в уже квантованных int8-весах слепка.

ВОПРОС, КОТОРЫЙ ЭТО РЕШАЕТ. Основная масса сети лежит в GPTQ int8 (группа 128, со смещением).
Переход на int7/int6 обычно считают разменом точности. Но у GPTQ есть свойство, которого нет у
плавающего формата: внутри группы веса уже ЦЕЛЫЕ, и вопрос не «сколько точности потеряем», а
«сколько РАЗЛИЧНЫХ значений группа вообще использует». Если размах (max-min) внутри группы меньше
2^b, то перекодировка в b разрядов со СВОИМ смещением группы -- ТОЧНАЯ, элемент в элемент, ноль
потери. Тогда это не сжатие, а снятие неиспользуемого диапазона.

ЧТО СЧИТАЕТСЯ. По каждой группе: размах занятых значений и минимальная ширина ceil(log2(размах+1)).
Дальше -- доля групп, укладывающихся в 7 / 6 / 5 разрядов ТОЧНО, и средняя ширина (это и есть предел
ленивой перекодировки без единой потери).

    python tools/gptq_zanyatost.py <слепок> [--тензоров 40]
"""
import argparse
import glob
import json
import math
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("слепок")
    ap.add_argument("--тензоров", type=int, default=40)
    a = ap.parse_args()
    from safetensors import safe_open

    инд = json.load(open(glob.glob(os.path.join(a.слепок, "*.index.json"))[0]))["weight_map"]
    цели = sorted(n for n in инд if n.endswith(".qweight") and "visual" not in n
                  and not n.startswith("mtp"))
    шаг = max(1, len(цели) // a.тензоров)
    цели = цели[::шаг][:a.тензоров]
    print(f"тензоров всего {sum(1 for n in инд if n.endswith('.qweight'))}, "
          f"смотрим {len(цели)} (равномерно по сети)\n")
    print(f"{'тензор':<46} {'ширина ср':>9} {'<=7':>7} {'<=6':>7} {'<=5':>7}")

    сум_ш = сум_n = 0
    доли = {7: 0, 6: 0, 5: 0}
    всего_гр = 0
    for имя in цели:
        with safe_open(os.path.join(a.слепок, инд[имя]), framework="pt") as f:
            qw = f.get_tensor(имя)                       # [K/4, N] int32, 4 ключа в слове, bits=8
        K4, N = qw.shape
        u = qw.to(torch.int64) & 0xFFFFFFFF
        # четыре байта слова -- четыре ПОДРЯД идущих ключа одного столбца
        б = torch.stack([(u >> (8 * i)) & 0xFF for i in range(4)], dim=1)   # [K/4, 4, N]
        q = б.permute(0, 1, 2).reshape(K4 * 4, N)                           # [K, N]
        K = K4 * 4
        if K % 128:
            continue
        г = q.view(K // 128, 128, N)
        разм = (г.amax(dim=1) - г.amin(dim=1)).flatten()                    # [K/128 * N]
        ш = torch.ceil(torch.log2((разм + 1).float())).clamp_min(1)
        n = ш.numel()
        сум_ш += ш.sum().item(); сум_n += n; всего_гр += n
        стр = {b: (ш <= b).float().mean().item() for b in (7, 6, 5)}
        for b in доли:
            доли[b] += (ш <= b).sum().item()
        print(f"{имя.replace('model.language_model.layers.', 'сл.').replace('.qweight',''):<46} "
              f"{ш.mean().item():>9.2f} {стр[7]:>7.1%} {стр[6]:>7.1%} {стр[5]:>7.1%}")
        del qw, u, б, q, г, разм, ш

    print(f"\nСРЕДНЯЯ ЗАНЯТАЯ ШИРИНА: {сум_ш/сум_n:.2f} разряда из 8")
    for b in (7, 6, 5):
        print(f"  групп, укладывающихся в {b} разрядов ТОЧНО: {доли[b]/всего_гр:.2%}")
    print("\nЧитать так: доля при ширине b -- это доля весов, которые в b разрядов переносятся"
          "\nБЕЗ ЕДИНОЙ ПОТЕРИ (со своим смещением группы). Остальным нужен либо разряд шире,"
          "\nлибо честное огрубление -- и вот ТОЛЬКО там начинается размен.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
