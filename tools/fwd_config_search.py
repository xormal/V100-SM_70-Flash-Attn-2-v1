# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# МЕТОД (б): перебор конфигураций рукописного forward по ДЕШЁВОМУ СУРРОГАТУ.
#
# Доктрина решателя: когда верный метод упирается во время, подозревай ПОРЯДОК, а его цену задаёт
# ШИРИНА развёртки; тонкий порядок ищется перебором с ДЕШЁВЫМ суррогатом, а не полным прогоном
# каждого кандидата. Здесь суррогат считается в закрытой форме за миллисекунды, а замер одной
# конфигурации стоит минуты -- то самое соотношение, ради которого суррогат и нужен.
#
# СУРРОГАТ ПРОВЕРЕН НА УЖЕ СДЕЛАННЫХ ЗАМЕРАХ (это его единственное оправдание):
#   64/64 2x4 VC=1  загр/mma 1.125 -> 3.663 мс      32/64 1x4      1.125 -> 5.972   (вчетверо меньше варпов)
#   64/128 2x4 VC=4 загр/mma 0.875 -> 3.616         64/64 2x4 VC=4 1.125 -> 4.782   (штраф дробления)
# Он объясняет и порядок величин, и провалы; на большее не претендует.
import itertools, json

D = 256; PAD = 4; SMEM_MAX = 94000; REG_MAX = 255; OVERHEAD = 96
VC_PENALTY = 0.06          # замерено: VC=2 стоит ~5.4%, VC=4 ~30% на неизменной геометрии


def model(BQ, BK, WM, WN, DC, VC):
    SM, SN = BQ // WM, BK // WN
    OM, ON = BQ // WM, D // WN
    if SM % 16 or SN % 16 or ON % 16 or OM != SM:  return None
    if BK % VC or (BK // VC) % 16:                 return None
    warps = WM * WN
    if warps * 32 > 1024 or warps < 2:             return None
    if (BK * DC // 8) % (warps * 32):              return None       # кусок делится на нити нацело
    SMB, SNB, OMB, ONB = SM // 16, SN // 16, OM // 16, ON // 16
    qk_l = (SMB + SNB) * (D // 4);  qk_m = SMB * SNB * (D // 4)
    pv_l = (OMB + ONB) * (BK // 4); pv_m = OMB * ONB * (BK // 4)
    lpm = (qk_l + pv_l) / (qk_m + pv_m)
    regs = SMB * SNB * 8 + OMB * ONB * 8 + OVERHEAD
    smem = 2 * (BQ * (D + PAD) + BQ * (BK + PAD) + BK * (DC + PAD) + (BK // VC) * (D + PAD))
    # ПОТОЛОК РЕГИСТРОВ ЗАДАЁТ __launch_bounds__, А НЕ ISA. Компилятор обязан уложиться в
    # 65536/(нити*MINB) регистров на нить; при 512 нитях это 128, и он ДОЛИВАЕТ РАЗЛИВ, чтобы
    # уложиться, ничего об этом не сообщая в скорости. Замерено ptxas: 64/256 2x8 -> 128 регистров
    # и 152 байта разлива; 4x4 -> 128 и 8 байт. Без этого ограничения перебор выдаёт заведомо
    # разливающиеся конфигурации в верхушку списка.
    reg_cap = min(REG_MAX, 65536 // (warps * 32))
    if smem > SMEM_MAX or regs > reg_cap:          return None
    blocks = min(98304 // max(smem, 1), 65536 // max(regs * warps * 32, 1))
    if blocks < 1:                                 return None
    # суррогат: загрузки на MMA, штраф за дробление V, штраф за слишком мало варпов на SM
    warps_sm = warps * blocks
    penalty = (1 + VC_PENALTY * (VC - 1)) * (1.0 if warps_sm >= 8 else 8.0 / warps_sm)
    return dict(BQ=BQ, BK=BK, WM=WM, WN=WN, DC=DC, VC=VC, warps=warps, lpm=round(lpm, 3),
                regs=regs, smem=smem, blocks=blocks, warps_sm=warps_sm,
                score=round(lpm * penalty, 4))


cand = []
for BQ in (32, 64, 128):
    for BK in (32, 64, 128, 256):
        for WM in (1, 2, 4, 8):
            for WN in (1, 2, 4, 8):
                for DC in (16, 32, 64, 128, 256):
                    if DC > D: continue
                    for VC in (1, 2, 4, 8):
                        m = model(BQ, BK, WM, WN, DC, VC)
                        if m: cand.append(m)
cand.sort(key=lambda x: (x["score"], x["smem"]))
seen, top = set(), []
for c in cand:                                  # по одной лучшей на геометрию (BQ,BK,WM,WN)
    k = (c["BQ"], c["BK"], c["WM"], c["WN"])
    if k in seen: continue
    seen.add(k); top.append(c)
print(f"допустимых конфигураций: {len(cand)}, различных геометрий: {len(top)}")
print(f'{"BQ":>4}{"BK":>5}{"варп":>6}{"DC":>5}{"VC":>4}{"загр/mma":>10}{"рег":>6}{"smem":>8}{"бл/SM":>7}{"варп/SM":>9}{"суррогат":>10}')
for c in top[:16]:
    print(f'{c["BQ"]:>4}{c["BK"]:>5}{c["WM"]}x{c["WN"]:<4}{c["DC"]:>5}{c["VC"]:>4}{c["lpm"]:>10.3f}'
          f'{c["regs"]:>6}{c["smem"]//1024:>7}K{c["blocks"]:>7}{c["warps_sm"]:>9}{c["score"]:>10.3f}')
json.dump(top[:16], open("/tmp/fwd_top.json", "w"), indent=1)
