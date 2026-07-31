# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ПЛАНИРОВЩИК БЛОКА: на вход форма задачи, на выход -- конфигурация с МИНИМАЛЬНЫМ временем и
# раскладка работы по маршрутам метапроцессора и по фазам.
#
# Это исполняемая форма раздела §0 кукбука (fa2_src/fmha_kernel/gemm/volta_prims.h): единица
# проектирования -- ПРИЕХАВШАЯ ПЛИТКА, задача -- распределить работу над ней так, чтобы общее время
# было минимальным. Все коэффициенты ЗАМЕРЕНЫ на V100; источник помечен у каждого.
#
# Модель считает не "сколько флопов", а ВРЕМЯ КАЖДОГО МАРШРУТА и берёт максимум (маршруты идут
# параллельно, их взаимное влияние замерено), плюс то, что НЕ перекрывается: барьеры и пролог.
import sys, math, itertools, json

# =====================================================================================
# ЗАМЕРЕННЫЕ КОНСТАНТЫ. [И] -- замер этой сессии, [П] -- прежние замеры проекта, [С] -- спецификация.
# =====================================================================================
SM            = 80          # [С] V100
CLK           = 1.380e9     # [С] зафиксированная частота
WARP          = 32
SMEM_MAX      = 96 * 1024   # [С] на блок
REG_FILE      = 65536       # [С] регистров на SM
REG_ISA       = 255         # [П] потолок ISA, не ёмкости

# --- тензорный маршрут
HMMA_PEAK     = 0.434       # [И] инструкций варпа/такт/SM, чистый цикл MMA (volta_metablock.cu)
HMMA_MAC      = 1024        # [И] МАС на инструкцию варпа: 4 квад-пары x 8*8*4 (сверено по TFLOPS)
# доля тензорного пика при подключении прочих конвейеров -- ЗАМЕРЕНО, не выведено
T_ALONE       = 1.000       # [И]
T_PLUS_I      = 0.937       # [И] + INT32
T_PLUS_F      = 0.948       # [И] + FP32
T_PLUS_IF     = 0.954       # [И] + оба (ТРИ конвейера дешевле двух)
T_PLUS_IFL    = 0.859       # [И] + LSU (все четыре, metablock4)
T_PLUS_D      = 0.843       # [И] + FP64 -- ломает, в набор не входит

# --- ёмкость нетензорных конвейеров ПОД HMMA (инструкций на одну HMMA)
INT_PER_HMMA  = 2.0         # [И] рабочая точка при 92% пика; плато 1->2 бесплатно (int32_capacity)
FP32_PER_HMMA = 1.0         # [И] из metablock: 0.784 прочих на 0.392 HMMA = 2 всего, поровну
# цена ЗАВИСИМОСТИ: та же работа в цепи против сдвинутой на плитку вперёд
DEP_CHAIN     = 0.886       # [И] доля пика, разворот в цепи (pipelined_expand)
DEP_SHIFTED   = 0.931       # [И] то же со сдвигом фазы

# --- скалярные конвейеры, инструкций варпа на такт на SM
IPC_FP32      = 1.71        # [И] add.f32, длинное тело (volta_gaps2)
IPC_INT32     = 1.64        # [И] lop3.b32
IPC_MIX_IF    = 2.71        # [И] СМЕСЬ: складываются, +58%
IPC_SFU       = 12.66/32    # [И] ex2 12.66 оп/такт/SM -> инструкций варпа
SFU_COST_FP32 = 0.076       # [И] поток fp32 стоит потоку ex2 7.6%

# --- разделяемая память
LDS_BYTES     = 50.4        # [И] байт/такт/SM, БЕСКОНФЛИКТНО (перемер: 48.3/54.0/50.4 при 4/8/16 Б).
                            # Прежние 62.7 сняты при ЧЕТЫРЁХКРАТНОМ конфликте банков -- грязная
                            # константа не имеет права стоять в модели.
# ВНИМАНИЕ НА ЕДИНИЦЫ: это инструкции НИТИ на такт на SM (варповых втрое-с-лишним меньше).
# Домножение их ещё и на 32 занижало стоимость обращений В ДВЕНАДЦАТЬ РАЗ -- на этом модель дважды
# давала коэффициент 0.51 и выглядела "недооценённой", хотя ошибка была арифметическая.
LDS_IPC_32    = 15.69       # [И] инструкций НИТИ/такт/SM при 4-байтовых обращениях
LDS_IPC_128   = 3.91        # [И] при 16-байтовых -- ВЧЕТВЕРО меньше при той же полосе
BAR_WAIT      = 33.4        # [И] тактов, bar.sync (volta_bar_pipeline)
BAR_SIGNAL    = 2.76        # [И] тактов, bar.arrive -- в 12 раз дешевле

# --- межполосные маршруты, тактов на редукцию 8 накопителей
XLANE_SHFL    = 0.115       # [И] 8 сложений + 5 бабочек (volta_tensor_cheap)
XLANE_TENSOR  = 0.069       # [И] умножение на единицы -- x1.66
XLANE_SMEM    = None        # через MIO + барьер, считается отдельно

# --- глобальная память
HBM_GBS       = 841.0       # [П] достижимо
HBM_HINT      = 1.060       # [И] .cs/.lu (volta_load_paths / gaps)
HBM_PREFETCH  = 1.053       # [И] prefetch.global.L2
CONST_IPC     = 13.27       # [И] загрузок/такт/SM, ОДНОРОДНЫЙ адрес, мимо MIO
CONST_COST    = 1.015       # [И] четыре константных обращения на четыре LDS = +1.5% времени
L2_HIT        = 0.977       # [П] ЗАМЕРЕНО ncu: префилл идёт из L2, в HBM уходит 2.3% обращений.
                            # Без этого модель объявляет ограничителем HBM, чего в железе НЕТ.
L2_BYTES      = 6*1024*1024 # [С] ёмкость L2

# --- ТРАНСПОРТ. HBM -- не "сколько байт нужно плитке", а сколько времени уходит, чтобы ПРИВЕЗТИ И
# УВЕЗТИ весь рабочий набор. Отсюда следует не объём, а ТРЕБУЕМАЯ АРИФМЕТИЧЕСКАЯ ПЛОТНОСТЬ:
#   нужно МАС/байт >= (пик МАС/такт) / (байт/такт) = 440 / 7.6 = 58, иначе транспорт не спрятать.
HBM_B_PER_CYC_SM = HBM_GBS * 1e9 / CLK / SM      # 7.6 Б/такт/SM
REQ_INTENSITY = HMMA_PEAK * HMMA_MAC / HBM_B_PER_CYC_SM

# --- загрузка ФРАГМЕНТОВ операндов: её в первой модели не было, а она на порядок больше перекладки
# ЗАГРУЗОК НА MMA -- НЕ КОНСТАНТА, а функция формы плитки варпа: на k-шаг грузится MB фрагментов A
# и NB фрагментов B, а MMA выполняется MB*NB. Формула сверена со свипом в шапке volta_fwd_block.cuh
# ТОЧНО: 32x32 -> 1.00, 32x16 -> 1.50, 16x16 -> 2.00, 16x32 -> 1.50.
def loads_per_mma(MB, NB):
    return (MB + NB) / (MB * NB)

# РЕГИСТРЫ: накопители S и O плюс фрагменты и перекладка. Плитка 32x32 проиграла в свипе НЕ по
# загрузкам (у неё их меньше всех), а разливом при 254 регистрах -- без этой модели планировщик
# уверенно выбирает заведомо худшее.
REG_BASE      = 46          # [П] подогнано по отгруженной точке: 178 при SMB*SNB=2, OMB*ONB=8
SPILL_PENALTY = 1.30        # [П] 32x32: 4.752 против 3.663 при прочих равных -- разлив стоит ~30%
FRAG_BYTES    = 8           # байт на полосу за одно обращение LDS.64

# --- разворот int8 -> fp16 нашим примитивом
EXPAND_INSTR  = 3           # [П] инструкций на 4 байта (§1a volta_prims.h)


class Work:
    """Работа над ОДНОЙ приехавшей плиткой, разложенная ПО ТИПУ (не по типу данных)."""
    def __init__(s):
        s.mac        = 0      # чистые умножения-с-накоплением -> ТОЛЬКО тензор
        s.xlane_red  = 0      # межполосные редукции          -> тензор (x1.66 над сдвигами)
        s.elem_fp    = 0      # поэлементные плавающие        -> FP32
        s.elem_int   = 0      # поэлементные целые/байтовые   -> INT32, переполнение в поля fp32
        s.sfu        = 0      # экспоненты                    -> SFU
        s.smem_bytes = 0      # перемещение через smem        -> самая ШИРОКАЯ инструкция
        s.hbm_bytes  = 0      # трафик HBM
        s.barriers   = 0      # двусторонние ожидания
        s.barriers_1w= 0      # ОДНОСТОРОННИЕ -> сигнал вместо ожидания


def decompose(cfg, shape):
    """Разложить работу над плиткой ключей BK для блока запросов BQ."""
    BQ, BK, D, warps = cfg['BQ'], cfg['BK'], cfg['D'], cfg['warps']
    w = Work()
    # два умножения: S = Q*K^T (BQ x BK x D) и O += P*V (BQ x D x BK)
    w.mac = BQ * BK * D + BQ * D * BK
    # софтмакс: экспонента на элемент S, две межполосные редукции (максимум и сумма)
    w.sfu = BQ * BK
    w.xlane_red = 2 * BQ                       # по строке: максимум и сумма
    # поэлементное плавающее: масштаб S, перемасштабирование накопителя O, поправки
    w.elem_fp = BQ * BK + BQ * D
    # поэлементное целое: адреса, маска, упаковка P
    w.elem_int = BQ * BK // 2 + (BK * D) // 8  # маска на пару + адреса загрузки
    if cfg['int8']:
        w.elem_int += (BK * D // 4) * EXPAND_INSTR   # разворот K и V из int8
    # перемещение: K и V в smem, P между умножениями -- ПЛЮС чтение фрагментов операндов под каждую
    # MMA. Последнее на ПОРЯДОК больше перекладки (1.5 обращения на MMA), и без него модель
    # недооценивает разделяемую память вдвое.
    el = 1 if cfg['int8'] else 2               # байт на элемент
    staging = BK * D * el * 2 + BQ * BK * 2
    n_mma = w.mac / HMMA_MAC                   # инструкций варпа на плитку
    frag = n_mma * cfg['lpm'] * FRAG_BYTES * WARP
    w.smem_bytes = staging + frag
    w.frag_bytes = frag; w.staging_bytes = staging
    w.hbm_bytes  = BK * D * el * 2
    w.barriers    = cfg['bar2']
    w.barriers_1w = cfg['bar1']
    return w


def schedule(w, cfg):
    """Время (в тактах на SM) для одной плитки. Маршруты параллельны; берём максимум + непокрытое."""
    warps = cfg['warps']
    # --- тензор
    t_hmma_instr = w.mac / HMMA_MAC / warps    # инструкций варпа на варп... нормируем на блок
    # доля пика зависит от того, СКОЛЬКО конвейеров подключено
    frac = T_ALONE
    if w.elem_int and w.elem_fp and w.smem_bytes: frac = T_PLUS_IFL
    elif w.elem_int and w.elem_fp:               frac = T_PLUS_IF
    elif w.elem_int:                             frac = T_PLUS_I
    elif w.elem_fp:                              frac = T_PLUS_F
    # и от того, СДВИНУТА ли фаза
    frac *= (DEP_SHIFTED / DEP_CHAIN) if cfg['phase_shift'] else 1.0
    t_tensor = (w.mac / HMMA_MAC) / (HMMA_PEAK * frac)          # тактов на SM
    # --- межполосные редукции: маршрут выбирается ДЕШЕВЕЙШИЙ
    red_shfl   = w.xlane_red / 8 * XLANE_SHFL * 8
    red_tensor = w.xlane_red / 8 * XLANE_TENSOR * 8
    t_xlane = min(red_shfl, red_tensor)
    xlane_route = 'тензор (умножение на единицы)' if red_tensor < red_shfl else 'сдвиги (бабочка)'
    # --- поэлементное: FP32 и INT32 идут ПАРАЛЛЕЛЬНО и под HMMA
    #     сперва вычитаем то, что уезжает БЕСПЛАТНО в свободные слоты
    free_int = t_tensor * HMMA_PEAK * frac * INT_PER_HMMA * WARP
    free_fp  = t_tensor * HMMA_PEAK * frac * FP32_PER_HMMA * WARP
    int_over = max(0.0, w.elem_int - free_int)
    fp_over  = max(0.0, w.elem_fp  - free_fp)
    # переполнение целых уводим в упакованные поля fp32 (§1e: 2 целых на инструкцию)
    if int_over > 0 and fp_over == 0:
        moved = min(int_over, free_fp - w.elem_fp) if free_fp > w.elem_fp else 0.0
        int_over -= moved * 2                    # одна инструкция fp32 = ДВА целых
        int_over = max(0.0, int_over)
    t_int = int_over / (IPC_INT32 * WARP)
    t_fp  = fp_over  / (IPC_FP32  * WARP)
    # --- SFU: перекрывается с FP32, платим 7.6%
    t_sfu = w.sfu / (IPC_SFU * WARP) * (1 + SFU_COST_FP32)
    # --- разделяемая память. ВАЖНО: у чтения фрагментов операнда байты почти бесплатны --
    # полосы одной четвёрки читают ОДИН И ТОТ ЖЕ адрес, и Volta раздаёт это широковещанием.
    # Поэтому в ПОЛОСУ идёт только перекладка, а фрагменты -- только в ИНСТРУКЦИИ (закон MIO).
    t_smem_bw = w.staging_bytes / LDS_BYTES
    # ширина влияет на ИНСТРУКЦИИ, не на полосу: фрагмент операнда всегда 8 байт (LDS.64), а
    # перекладку можно вести по 16 (LDS.128) -- только она и выигрывает от выравнивания шага.
    ipc = LDS_IPC_128 if cfg['wide_smem'] else LDS_IPC_32
    t_smem_ins = (w.staging_bytes / (16 if cfg['wide_smem'] else 8)
                  + w.frag_bytes / FRAG_BYTES) / ipc      # ipc уже в инструкциях НИТИ
    t_smem = max(t_smem_bw, t_smem_ins)
    # --- HBM
    gbs = HBM_GBS * (HBM_HINT if cfg['bus_hint'] else 1.0) * (HBM_PREFETCH if cfg['prefetch'] else 1.0)
    # ТРАНСПОРТ. Попадание в L2 не постоянная величина: оно есть, пока рабочий набор K/V ВЛЕЗАЕТ
    # в L2. При длинном контексте (чанковый префилл) не влезает, и транспорт становится задающим.
    hit = cfg.get('l2_hit', L2_HIT)
    t_hbm = w.hbm_bytes * (1.0 - hit) / (gbs * 1e9 / CLK / SM)
    t_l2  = w.hbm_bytes * hit / (4.0 * gbs * 1e9 / CLK / SM)
    t_hbm = max(t_hbm, t_l2)
    # --- НЕ перекрывается: барьеры
    t_bar = w.barriers * BAR_WAIT + w.barriers_1w * (BAR_SIGNAL if cfg['split_bar'] else BAR_WAIT)
    # ЗАКОН ТРАНСПОРТА: на один элемент ключа приходится 2*BQ*d умножений и 2*el*d байт,
    # значит плотность = BQ/el. Требуется >= REQ_INTENSITY, иначе транспорт не спрятать НИКАКОЙ
    # плиткой ключей -- лечится только УВЕЛИЧЕНИЕМ BQ или сжатием формата.
    el = 1 if cfg['int8'] else 2
    intensity = cfg['BQ'] / el
    routes = {'тензор': t_tensor, 'межполосное': t_xlane, 'INT32': t_int, 'FP32': t_fp,
              'SFU': t_sfu, 'разделяемая': t_smem, 'HBM': t_hbm}
    t_par = max(routes.values())
    return t_par + t_bar, routes, t_bar, xlane_route, intensity


def warp_tile(cfg):
    """Форма плитки варпа и вытекающие из неё загрузки-на-MMA и регистры."""
    BQ, BK, D, warps = cfg['BQ'], cfg['BK'], cfg['D'], cfg['warps']
    WM, WN = cfg['WM'], cfg['WN']
    SMt, SNt = BQ // WM, BK // WN                 # плитка варпа для S
    OMt, ONt = BQ // WM, D // WN                  # для O
    if SMt % 16 or SNt % 16 or OMt % 16 or ONt % 16: return None
    MB, NB = SMt // 16, SNt // 16
    OMB, ONB = OMt // 16, ONt // 16
    lpm = loads_per_mma(MB, NB)
    regs = REG_BASE + 8*(MB*NB) + 8*(OMB*ONB) + 4*(MB+NB)
    return dict(SMt=SMt, SNt=SNt, MB=MB, NB=NB, OMB=OMB, ONB=ONB, lpm=lpm, regs_need=regs)


def occupancy(cfg):
    """Сколько блоков резидентно и не бьёт ли квантование волн."""
    BQ, BK, D, warps = cfg['BQ'], cfg['BK'], cfg['D'], cfg['warps']
    el = 1 if cfg['int8'] else 2
    pad = 8 if cfg['wide_smem'] else 4          # выровненный шаг нужен для 128-битных обращений
    # K подаётся КУСКАМИ по DC (буфер BK x DC, а не BK x D) -- иначе модель завышает на 26 КБ и
    # объявляет отгруженную конфигурацию нерезидентной.
    DC = cfg.get('DC', 64)
    smem = (BQ*(D+pad) + BQ*(BK+pad) + BK*(DC+pad) + BK*(D+16)) * el
    if smem > SMEM_MAX: return 0, smem, 0
    by_smem = SMEM_MAX // smem
    regs = REG_FILE // (warps * WARP * max(1, by_smem))
    if regs > REG_ISA: regs = REG_ISA
    return by_smem, smem, regs


# [СВЕРКА 2] свип формы плитки варпа из шапки volta_fwd_block.cuh (S=8192 H=4, частоты зафиксированы).
# Модель обязана воспроизвести ПОРЯДОК этих четырёх точек, иначе она выберет заведомо худшее.
TILE_SWEEP = [((2,2), 1.00, 254, True,  4.752),
              ((2,1), 1.50, 178, False, 3.663),
              ((1,1), 2.00, 182, False, 4.400),
              ((1,2), 1.50, 204, False, 5.964)]


def calibrate_tiles():
    print("=== СВЕРКА 2: форма плитки варпа (замер против модели, S=8192 H=4) ===")
    print(f"{'плитка':<12}{'загр/mma замер':>16}{'модель':>9}{'регистры':>10}{'мс замер':>10}")
    ok = True
    for (MB,NB), lpm_m, regs_m, spill, ms in TILE_SWEEP:
        lpm_c = loads_per_mma(MB, NB)
        mark = 'ok' if abs(lpm_c - lpm_m) < 1e-9 else 'РАСХОЖДЕНИЕ'
        if mark != 'ok': ok = False
        print(f"{MB*16}x{NB*16:<8}{lpm_m:>16.2f}{lpm_c:>9.2f}{regs_m:>10}{ms:>10.3f}   {mark}")
    print(f"  формула загрузок-на-MMA {'СОВПАДАЕТ со всеми четырьмя замерами' if ok else 'НЕ СОВПАДАЕТ'}")
    print("  порядок времён задаётся НЕ загрузками (32x32 их меньше всех, но она худшая после 16x32),")
    print("  а СОВМЕСТНО загрузками и регистрами -- это трёхсторонний компромисс, не одна ось.\n")
    return ok


# [СВЕРКА] Модель не имеет права предсказывать, пока не воспроизводит ИЗМЕРЕННОЕ. Отгруженная
# конфигурация и её замеры (шапка volta_fwd_block.cuh, частоты зафиксированы):
SHIPPED = dict(BQ=64, BK=64, warps=8, WM=2, WN=4, wide_smem=False, phase_shift=False,
               split_bar=False, bus_hint=False, prefetch=False, bar2=2, bar1=3, int8=False)
# ОСТОРОЖНО С ИСХОДНЫМИ: таблица в шапке volta_fwd_block.cuh снята при H=4, а в bench_prefill_d256
# форма (1,16,4096) -- ДРУГАЯ точка. Первая сверка провалилась именно на этом, а не на модели.
# Проверка арифметикой перед тем, как верить числам: работа H*Sq*Sk у трёх точек ниже
# относится как 1 : 4 : 8, замеры дают 1 : 3.27 : 6.38 -- согласуется. С H=16 не согласовалось бы.
MEASURED = [((1, 4, 4096, 4096, 256, True), 1.076),    # шапка: 1.607 -> 1.076
            ((1, 4, 8192, 8192, 256, True), 3.523),    # после правки PADV
            ((1, 2, 16384, 16384, 256, True), 6.865)]


def calibrate():
    print("=== СВЕРКА МОДЕЛИ С ЗАМЕРОМ (отгруженная конфигурация) ===")
    print(f"{'форма':<34}{'замер, мс':>11}{'модель, мс':>12}{'отношение':>11}")
    rows = []
    for shape, meas in MEASURED:
        B, H, Sq, Sk, D, causal = shape
        cfg = dict(SHIPPED); cfg['D'] = D
        cfg['lpm'] = warp_tile(cfg)['lpm']
        occ, smem, regs = occupancy(cfg)
        if occ == 0: print(f"  {str(shape):<34} НЕ РЕЗИДЕНТНА по модели smem ({smem/1024:.1f} КБ)"); continue
        w = decompose(cfg, shape)
        t_tile, routes, t_bar, xr, inten = schedule(w, cfg)
        ntile = math.ceil(Sk / cfg['BK'])
        if causal: ntile = max(1, ntile / 2)
        nblk = math.ceil(Sq / cfg['BQ']) * H * B
        blocks_per_sm = math.ceil(nblk / SM)
        waves = blocks_per_sm / max(1, occ)
        tail = math.ceil(waves) / waves if waves > 0 else 1.0
        hide = 1.0 if occ >= 2 else 1.04
        pred = t_tile * ntile * blocks_per_sm * tail * hide / CLK * 1e3
        lim = max(routes, key=routes.get)
        rows.append((shape, meas, pred))
        print(f"{str(shape):<34}{meas:>11.3f}{pred:>12.3f}{pred/meas:>11.2f}   ограничитель: {lim}")
    r = [p / m for _, m, p in rows]
    k = sum(r) / len(r)
    spread = max(r) / min(r)
    print(f"\n  средний коэффициент {k:.2f}, разброс {spread:.2f}x")
    print("  РАЗБРОС -- главное число: если он мал, модель верна ПО ФОРМЕ и лечится одним множителем;")
    print("  если велик -- модель неверна по существу и предсказывать ей нельзя.")
    print(f"  Коэффициент {k:.2f} = ЗАПАС отгруженного ядра до идеального расписания (1/{1/k:.2f} = x{1/k:.2f}).")
    print("  ЧЕСТНО: часть этого запаса может быть неучтённой в модели стоимостью, а не только")
    print("  недобором. Поэтому модель применяется ДЛЯ РАНЖИРОВАНИЯ, а абсолют -- с множителем.")
    return k, spread


def compare_shipped(shape, K):
    """Отгруженная конфигурация против найденной планировщиком -- на ОДНОЙ форме."""
    B, H, Sq, Sk, D, causal = shape
    def run(cfg):
        occ, smem, regs = occupancy(cfg)
        if occ == 0: return None
        w = decompose(cfg, shape)
        t_tile, routes, t_bar, xr, inten = schedule(w, cfg)
        ntile = math.ceil(Sk / cfg['BK'])
        if causal: ntile = max(1, ntile / 2)
        bps = math.ceil(math.ceil(Sq/cfg['BQ'])*H*B / SM)
        wv = bps / max(1, occ); tl = math.ceil(wv)/wv if wv > 0 else 1.0
        hd = 1.0 if occ >= 2 else 1.04
        return t_tile*ntile*bps*tl*hd/CLK*1e3/K, routes, occ, smem, regs, inten
    sh = dict(SHIPPED); sh['D'] = D; sh['lpm'] = warp_tile(sh)['lpm']
    a = run(sh)
    best = plan(shape)
    b = (best[0]/K, best[2], best[4], best[5], best[6], best[9])
    print(f"=== {shape}: ОТГРУЖЕННОЕ ПРОТИВ НАЙДЕННОГО (с калибровкой) ===")
    for nm, r, cfg in (("отгружено", a, sh), ("найдено", b, best[1])):
        if r is None: print(f"  {nm}: нерезидентно"); continue
        lim = max(r[1], key=r[1].get)
        print(f"  {nm:<10} {r[0]:7.3f} мс  BQ={cfg['BQ']:<4}BK={cfg['BK']:<4}варпов={cfg['warps']:<3}"
              f"блоков/SM={r[2]}  ограничитель: {lim:<12} плотность {r[5]:.0f}")
    if a: print(f"  -> модель обещает x{a[0]/b[0]:.2f}\n")


def plan(shape, verbose=True):
    B, H, Sq, Sk, D, causal = shape
    best = None
    for BQ in (32, 64, 128):
        for BK in (32, 64, 128):
            for WM, WN in ((1,4),(2,4),(2,2),(4,2),(1,8),(2,8),(4,4)):
                warps = WM * WN
                if warps not in (4, 8, 16): continue
                if BQ % 16 or BK % 16: continue
                for wide in (True, False):
                    for shift in (True, False):
                        for sb in (True, False):
                            cfg = dict(BQ=BQ, BK=BK, D=D, warps=warps, WM=WM, WN=WN, int8=False,
                                       wide_smem=wide, phase_shift=shift, split_bar=sb,
                                       bus_hint=True, prefetch=True, bar2=2, bar1=3)
                            wt = warp_tile(cfg)
                            if wt is None: continue
                            cfg['lpm'] = wt['lpm']
                            occ, smem, regs = occupancy(cfg)
                            if occ == 0 or regs < 64: continue
                            spill = wt['regs_need'] > min(regs, REG_ISA)
                            w = decompose(cfg, shape)
                            t_tile, routes, t_bar, xr, inten = schedule(w, cfg)
                            ntile = math.ceil(Sk / BK)
                            nblk  = math.ceil(Sq / BQ) * H * B
                            if causal: ntile = max(1, ntile / 2)
                            # ЗАНЯТОСТЬ НЕ ДЕЛИТ РАБОТУ. Ограничитель здесь -- пропускная способность
                            # (разделяемая память), а два резидентных блока делят ОДИН И ТОТ ЖЕ MIO:
                            # работа на SM не меняется. Занятость даёт только сокрытие задержек и
                            # ровнее хвост волны. Прежняя формула делила на occ и потому уверенно
                            # выбирала мелкую плитку -- ровно то, что замеренный свип отвергает.
                            blocks_per_sm = math.ceil(nblk / SM)
                            waves = blocks_per_sm / max(1, occ)
                            tail = math.ceil(waves) / waves if waves > 0 else 1.0
                            hide = 1.0 if occ >= 2 else 1.04     # [П] один блок/SM: прятать нечем
                            t_total = t_tile * ntile * blocks_per_sm * tail * hide / CLK * 1e3
                            if spill: t_total *= SPILL_PENALTY
                            cfg['regs_need'] = wt['regs_need']; cfg['spill'] = spill
                            rec = (t_total, cfg, routes, t_bar, occ, smem, regs, xr, waves, inten)
                            if best is None or t_total < best[0]: best = rec
    return best


def show(shape, best):
    B, H, Sq, Sk, D, causal = shape
    t, cfg, routes, t_bar, occ, smem, regs, xr, waves, inten = best
    print(f"=== ЗАДАЧА: B={B} H={H} Sq={Sq} Sk={Sk} d={D} {'причинная' if causal else 'полная'} ===")
    print(f"\nКОНФИГУРАЦИЯ С МИНИМАЛЬНЫМ ВРЕМЕНЕМ: {t:.3f} мс")
    print(f"  плитка BQ={cfg['BQ']} BK={cfg['BK']}, варпов {cfg['warps']}, "
          f"блоков/SM {occ}, smem {smem/1024:.1f} КБ, регистров {regs}, волн {waves}")
    print(f"  широкие обращения к smem: {'ДА' if cfg['wide_smem'] else 'нет'}"
          f" | сдвиг фаз: {'ДА' if cfg['phase_shift'] else 'нет'}"
          f" | барьеры половинками: {'ДА' if cfg['split_bar'] else 'нет'}")
    print(f"\nВРЕМЯ ПО МАРШРУТАМ (тактов на плитку; идут ПАРАЛЛЕЛЬНО, время = максимум):")
    mx = max(routes.values())
    for k, v in sorted(routes.items(), key=lambda x: -x[1]):
        bar = '#' * int(40 * v / mx) if mx > 0 else ''
        print(f"    {k:<14} {v:9.1f}  {bar}{'  <- ОГРАНИЧИТЕЛЬ' if v == mx else ''}")
    print(f"    {'барьеры':<14} {t_bar:9.1f}  (НЕ перекрывается, добавляется сверху)")
    print(f"\nЗАКОН ТРАНСПОРТА: плотность {inten:.0f} МАС/байт при требуемых {REQ_INTENSITY:.0f}"
          f"  -> {'ХВАТАЕТ' if inten >= REQ_INTENSITY else 'НЕ ХВАТАЕТ: транспорт не спрятать, увеличивать BQ или сжимать формат'}")
    print(f"\nЛОГИКА БЛОКА -- что на каком маршруте и в какой фазе:")
    print(f"    тензор  -> плитка N:   S = Q*K^T и O += P*V  (чистые умножения, не разбавлять)")
    print(f"    тензор  -> плитка N:   редукции софтмакса -- {xr}")
    print(f"    INT32   -> плитка N+1: адреса, маска, упаковка P (2 инстр/HMMA даром)")
    print(f"    FP32    -> плитка N-1: перемасштабирование O, поправки (1 инстр/HMMA даром)")
    print(f"    SFU     -> плитка N:   exp2, под неё подложить работу FP32 (стоит 7.6%)")
    print(f"    LSU     -> плитка N+1: загрузка K/V ОДНОЙ широкой инструкцией")
    print(f"    константы -> варп-однородное: масштабы, шаги, длины (мимо MIO, +1.5%)")


if __name__ == '__main__':
    calibrate_tiles()
    K, SPREAD = calibrate()
    print()
    if SPREAD > 1.6:
        print("  !! разброс велик -- модель НЕ КАЛИБРОВАНА, предсказания ниже читать нельзя\n")
    SHAPES = [(1, 16, 4096, 4096, 256, True), (1, 4, 8192, 8192, 256, True),
              (12, 2, 8192, 32768, 256, True), (1, 32, 2048, 2048, 256, True)]
    if len(sys.argv) > 1:
        SHAPES = [tuple(int(x) if x not in ('c','f') else (x=='c') for x in sys.argv[1].split(','))]
    for sh in SHAPES:
        compare_shipped(sh, K)
    print()
    show(SHAPES[1], plan(SHAPES[1]))
