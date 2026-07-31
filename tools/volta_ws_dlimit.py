# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ПРЕДЕЛ head_dim у ядра ролей (fa2_sm70/csrc/volta_fwd_ws.cuh), отдельно для байтового и для
# fp16-KV. Раскладка взята ИЗ ИСХОДНИКА ядра, а не придумана: проверка sanity ниже обязана
# воспроизвести отгруженную конфигурацию (BQ=32, BK=64, DV=256, 8 считающих -> 90624 Б).
#
#   sQ = BQ*(D+PAD) полуслов, sP = BQ*(BK+PAD) полуслов -- всегда fp16;
#   двойной буфер K/V = 2*BK*((D+16) + (DV+PADV)) ЭЛЕМЕНТОВ, элемент = 1 Б (int8) или 2 Б (fp16).
#
# ОГРАНИЧИТЕЛЬ, который и решает вопрос: буфер K/V растёт ЛИНЕЙНО по d и он двойной, поэтому у
# байтового пути запас ровно вдвое больше по этому слагаемому (но не по всему -- Q и P в fp16 у
# обоих). Второй предел -- накопитель O в регистрах: BQ*DV/(32*nCons) на нить.
#
# Запуск: python tools/volta_ws_dlimit.py
PAD, PADV, SMEM, NPROD = 4, 16, 96 * 1024, 4
ACC_MAX = 96          # регистров под O на нить; замерено 32 при 96-98 всего -> ~64 занято прочим
FRAG = 16             # SM/SN/ON обязаны быть кратны 16 (шаг фрагмента mma.sync)


def smem(BQ, BK, D, DV, esz):
    return (BQ * (D + PAD) + BQ * (BK + PAD)) * 2 + 2 * BK * ((D + 16) + (DV + PADV)) * esz


def configs(D, esz, min_cons):
    out = []
    for BQ in (16, 32, 64, 128):
        for BK in (16, 32, 64, 128):
            for div in (1, 2, 4, 8):          # число проходов по S: DV = D/div
                DV = D // div
                if DV * div != D or DV < FRAG:
                    continue
                s = smem(BQ, BK, D, DV, esz)
                if s > SMEM:
                    continue
                for WM in (1, 2, 4, 8):
                    for WN in (1, 2, 4, 8):
                        nc = WM * WN
                        if nc < min_cons or nc + NPROD > 32:
                            continue
                        SM, SN, ON = BQ // WM, BK // WN, DV // WN
                        if min(SM, SN, ON) < FRAG or SM * WM != BQ or SN * WN != BK or ON * WN != DV:
                            continue
                        acc = BQ * DV // (32 * nc)
                        if acc > ACC_MAX:
                            continue
                        out.append(dict(BQ=BQ, BK=BK, DV=DV, WM=WM, WN=WN, nc=nc, smem=s,
                                        passes=div, acc=acc, work=BQ * BK))
    #  сортировка: сперва МЕНЬШЕ проходов по S (пересчёт S -- самая дорогая цена), затем больше
    #  работы за плитку, затем больше считающих варпов
    out.sort(key=lambda c: (c["passes"], -c["work"], -c["nc"]))
    return out


if __name__ == "__main__":
    ship = smem(32, 64, 256, 256, 1)
    assert ship == 90624, f"раскладка разошлась с ядром: {ship} != 90624"
    print(f"sanity: отгруженная конфигурация = {ship} Б = {ship/1024:.1f} КБ  [OK]\n")
    print(f"{'d':>5} | {'int8 KV (BQ/BK/DV WMxWN, КБ, проходов)':^44} | {'fp16 KV':^44}")
    print("-" * 100)
    for D in (64, 128, 192, 256, 320, 384, 512, 640, 768):
        row = [f"{D:>5}"]
        for esz in (1, 2):
            c = configs(D, esz, 4)
            row.append(f"{c[0]['BQ']}/{c[0]['BK']}/{c[0]['DV']} {c[0]['WM']}x{c[0]['WN']}  "
                       f"{c[0]['smem']/1024:.0f}КБ  x{c[0]['passes']}".center(44)
                       if c else "НЕ ВЛЕЗАЕТ".center(44))
        print(" | ".join(row))
