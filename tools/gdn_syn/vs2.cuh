// СГЕНЕРИРОВАНО tools/gdn_syn/mkvar.py -- НЕ ПРАВИТЬ РУКАМИ
// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// ВЫХОД ЧАНКОВОГО GATED-DELTA-NET НА ТЕНЗОРНЫХ ЯДРАХ VOLTA.  Замена chunk_fwd_o из fla/Triton.
//
// ЗАЧЕМ. Замерено (§4qq): Triton при цели sm_70 НЕ ЭМИТИТ тензорных инструкций вообще -- в PTX
// mma.sync 0, wmma 0, hmma 0 и 1024 fma.rn.f32, то есть tl.dot компилируется в цикл FP32-FMA на
// обычных ядрах. Потолок 3.5 TFLOP/s против 83 у cuBLAS и 44 у наших рукописных ядер.
//
// ЧТО СЧИТАЕТСЯ (ровно как в chunk_fwd_kernel_o, чанк BT=64):
//     o[t][v] = scale * ( e^{g_t} * SUM_k q[t][k] h[v][k]
//                       + SUM_{s<=t} e^{g_t - g_s} * (SUM_k q[t][k] k[s][k]) * v[s][v] )
// Три умножения матриц на пару (чанк, голова):
//     G1: b_o[64,128] = q[64,128] x h^T          (h лежит [V][K], k непрерывен)
//     G2: b_A[64, 64] = q[64,128] x k^T          (k лежит [t][k], k непрерывен)
//     G3: b_o += tril(b_A)[64,64] x v[64,128]    (v лежит [t][v], v непрерывен)
//
// =================================================================================================
// ЧЕМ ЭТО ЯДРО ОГРАНИЧЕНО. ПОСЫЛКА "ЗАПАС ДО 93.6 TFLOP/s КРАТНЫЙ" -- НЕВЕРНА
// =================================================================================================
// Профилировщик на этой машине недоступен (нет прав на счётчики), поэтому раскладка снята АБЛЯЦИЕЙ:
// собирается вариант с выключенной фазой, разница по секундомеру и есть её цена. Исходная редакция,
// T=15680, H=4, чередующийся минимум из шести прогонов -- 0.217 мс:
//     всё                    0.217
//     без трёх умножений     0.191   -> вся МАТЕМАТИКА  0.052 (24%)
//     без записи выхода      0.177   -> ЗАПИСЬ          0.066 (30%) при 17% трафика
//     без чтения h           0.216   -> чтение h        0.027
// СЧЁТ РОФЛАЙНА, который надо было сделать первым: работа 4.19 ГФЛОП против 96.6 МБ трафика
// (q 16.1 + k 16.1 + v 16.1 + h 32.1 + o 16.1 МБ) -- это 43 ФЛОП/байт, тогда как машина при
// 93.6 TFLOP/s и 820 ГБ/с требует 114. ПОТОЛОК ЗДЕСЬ -- ПАМЯТЬ, и он равен 96.6 МБ / 820 ГБ/с =
// 0.118 мс = 35.6 TFLOP/s, а не 93.6. Сама арифметика уже шла на 81 TFLOP/s, то есть на 87% пика.
// Поэтому всё ниже -- транспорт и расписание, и ни одной правки в арифметику.
//
// ЧТО СДЕЛАНО И ЧТО ЭТО ДАЛО (T=15680, чередующийся минимум из девяти):
//   0.216  исходное
//   0.187  вынос выхода ЧЕРЕЗ РАЗДЕЛЯЕМУЮ вместо 32 разрозненных STG.E.U16 на нить
//   0.163  расписание: G2 первым, V в регистры в прологе, G1 в разрыв, LDG отделены от STS
//   0.171* та же редакция после парной записи A (*одноблочная опора, см. таблицу ниже)
//   0.142  ДВА РЕЗИДЕНТНЫХ БЛОКА: разделяемая ужата 77 -> 44 КБ (окно + подача h половинами)
//
// ИТОГ ПО СПЕКТРУ (мс, чередующийся минимум из девяти, GPU свободна, 1530 МГц):
//     T        исходное   одноблочная   ЭТО ЯДРО   ускорение   TFLOP/s
//     512       0.014       0.011        0.010      x1.40        13.4
//     2048      0.039       0.031        0.026      x1.50        20.6
//     8192      0.123       0.098        0.080      x1.54        26.8
//     15680     0.216       0.171        0.142      x1.52        28.9
// Относительная ошибка против fp64-эталона на CPU не изменилась ни на одной форме: 3.0-3.4e-04.
//
// АБЛЯЦИЯ ИТОГОВОГО ЯДРА показывает, где оно стоит теперь:
//     всё 0.143 | без математики 0.137 | без записи 0.106
// то есть арифметика спрятана почти целиком (0.006 из 0.143), чтение идёт на 759 ГБ/с = 93% от
// достижимых 820, и остаток -- ЗАПИСЬ. Она и есть следующая цель.
// 28.9 из 35.6 TFLOP/s -- это 81% РЕАЛЬНОГО потолка. Оставшийся запас x1.20, и он весь в записи.






namespace fa2_sm70 {
namespace gdn {
namespace vs2 {

// -------------------------------------------------------------------------------------------------
// ЧТО ПРОБОВАНО И ДАЛО РОВНО НОЛЬ ИЛИ ХУЖЕ (чтобы не повторять; все числа -- T=15680, минимум из 6)
// -------------------------------------------------------------------------------------------------
//  * ПОТОКОВЫЕ ПОДСКАЗКИ ЧТЕНИЯ ld.global.cs / ld.global.nc.cs на всех пяти тензорах: 0.162/0.162/
//    0.163 против 0.162 у базы -- НОЛЬ. Условие §4l было выполнено идеально (переиспользования нет
//    вообще, каждый элемент трогает ровно один блок), то есть это НЕ случай §4v, где .cs вредил.
//    Значит те +6% принадлежат режиму, где подсказка спасает ЧУЖИЕ строки в L2; здесь спасать нечего.
//  * ПОДСКАЗКА ЗАПИСИ st.global.cs / st.global.wt: 0.143/0.143 против 0.143 -- НОЛЬ. Проверено
//    ИМЕННО ПОСЛЕ того, как запись стала ограничителем (правило §15.2: отрицательный результат --
//    утверждение о РЕЖИМЕ). Режим сменился, вывод не изменился.
//  * ПОСТОЯННЫЕ БЛОКИ С ПРЕДПОДАЧЕЙ СЛЕДУЮЩЕГО ЧАНКА В РЕГИСТРЫ (сетка = числу SM, блок берёт
//    задания по кругу): механизм РАБОТАЕТ -- 0.193 -> 0.172 внутри своей же сборки, -- но плитка
//    следующего задания это 20 uint4 = 80 регистров, ЖИВЫХ ЧЕРЕЗ ЗАДНЮЮ ДУГУ ЦИКЛА, отчего ptxas
//    берёт 218 регистров вместо 104, и итог 0.169 против 0.163 у непостоянного варианта -- ХУЖЕ.
//    Перенос предподачи в конец тела не помог (218 регистров те же: диапазон всё равно межвитковый).
//    ВЫВОД: на Volta межчанковый конвейер через РЕГИСТРЫ не окупается; ту же работу бесплатно делает
//    второй резидентный блок, потому что его подача -- это чужие регистры, а не наши.
//  * ВЫНОС ВЫХОДА ПО ВАРПАМ (варп выносит свою плитку, хватает __syncwarp вместо __syncthreads):
//    0.146 против 0.144 -- ХУЖЕ. Барьер в 33 такта дешевле потери сплошности: при поблочном обходе
//    полоса варпа кроет 2 строки по 256 Б подряд, при поварповом -- 8 строк по 64 Б.
//  * ПРЯМАЯ ЗАПИСЬ ПАРАМИ __half2 в глобальную вместо круга через разделяемую: 0.151 против 0.144 --
//    ХУЖЕ. Круг через разделяемую окупается: 4 STG.128 на нить против 16 STG.E.U32, и раздутие
//    секторов вдвое меньше.
//  * ПОРЯДОК БЛОКОВ голова-по-быстрой-оси вместо чанк-по-быстрой (ради сплошности строки выхода
//    между одновременными блоками): 0.143 против 0.143 -- НОЛЬ. L2 в 6 МБ и так это сглаживает.

// РАСКЛАДКИ ОПЕРАНДА B В НАШЕМ API -- ПРОТИВОПОЛОЖНЫ ИНТУИЦИИ ИМЕНИ, и я на этом уже ошибся:
//   kKMajor: B это [N][K], непрерывен K  -> инструкция .row.col, это случай Q*K^T
//   kNMajor: B это [K][N], непрерывен N  -> инструкция .row.row, это случай P*V
// Отсюда: G1 (h лежит [V][K]) и G2 (k лежит [t][K]) -- ОБА kKMajor; G3 (v лежит [t][V]) -- kNMajor.

// =================================================================================================
// РАБОЧЕЕ ЯДРО: ДВА РЕЗИДЕНТНЫХ БЛОКА НА SM
// =================================================================================================
// ПОЧЕМУ ИМЕННО ДВА БЛОКА. Аблация одноблочной редакции: 0.163 всего, из них 0.141 чистый транспорт
// и 0.020 НЕПЕРЕКРЫТАЯ арифметика; транспорт при этом 685 ГБ/с из достижимых 820. Оба остатка лечит
// одно и то же -- пока один блок считает, второй везёт. Регистров у нас уже было 104 (порог 128
// взят), связывала ТОЛЬКО разделяемая память: 77 КБ при пороге 48.
//
// КАК ВЗЯТ ПОРОГ 48 КБ. Три буфера кладутся в ОДНО ОКНО, потому что живут в непересекающиеся моменты:
//     окно = sK (во время G2) -> ПОЛОВИНА sH (во время G1) -> sV (во время G3)
// Одного объединения мало (дало бы 61 КБ). Решает то, что h подаётся ПОЛОВИНАМИ ПО k: G1 -- сумма по
// k=0..127, её можно разбить на две по 64 БЕЗ ЕДИНОГО лишнего умножения и без смены плитки варпа,
// то есть отношение загрузок-на-mma остаётся 1.0. Тогда окно = max(17408, 18432) = 18432:
//     sQ 17408 + sA 9216 + окно 18432 = 45056 Б = 44.0 КБ  ->  ДВА блока в 96 КБ
// Ценой четырёх лишних барьеров (33 такта каждый против 18000 тактов на блок -- 0.7%).
//
// ПОРЯДОК ФАЗ ЗАДАН ЗАВИСИМОСТЯМИ, А НЕ ФОРМУЛОЙ:
//   1. G2 идёт ПЕРВЫМ: A -- единственная величина, которой нужен круг через разделяемую (её строит
//      один варп, а читают четыре), значит она должна быть готова как можно раньше;
//   2. каждая пачка LDG выдаётся за одну-две фазы до своего STS (h1 -- под G2, V -- под вторую
//      половину G1), потому что на Volta нет cp.async, но сама LDG неблокирующая: платит потребитель;
//   3. вся подача читается в РЕГИСТРЫ пачкой и только потом кладётся в разделяемую -- §13 каталога
//      сокрытия: весь выигрыш даёт РАЗНЕСЕНИЕ LDG и STS, а не равномерность выдачи.
template <int BT, int DK, int DV, int WM, int WN>
__global__ __launch_bounds__(WM * WN * 32, 2) void volta_gdn_chunk_o(
    const __half* __restrict__ q,      // [T, Hg, DK]
    const __half* __restrict__ k,      // [T, Hg, DK]
    const __half* __restrict__ v,      // [T, H,  DV]
    const __half* __restrict__ h,      // [NT, H, DV, DK]   состояние на начало чанка
    const float*  __restrict__ g,      // [T, H]            накопленный логарифм гейта
    __half* __restrict__ o,            // [T, H, DV]
    int T, int H, int Hg, float scale) {
  constexpr int PAD = 8;                             // 8 полуслов = 16 Б: держит uint4 выровненным
  constexpr int LDQ = DK + PAD, LDV = DV + PAD, LDA = BT + PAD;
  constexpr int DKH = DK / 2, LDH = DKH + PAD;       // половина k у операнда h
  constexpr int SM = BT / WM, SN = DV / WN, SNA = BT / WN;
  constexpr int MB = SM / 16, NB = SN / 16, NBA = SNA / 16;
  constexpr int nthr = WM * WN * 32;
  static_assert(SM % 16 == 0 && SN % 16 == 0 && SNA % 16 == 0, "плитки кратны шагу фрагмента");
  static_assert(DK == DV, "буфер выхода переиспользует sQ");
  static_assert(DKH % 8 == 0, "половина k кратна ширине uint4");

  extern __shared__ __half smem[];
  __half* sQ = smem;                                 // [BT][LDQ]  -- он же sO в эпилоге
  __half* sA = sQ + BT * LDQ;                        // [BT][LDA]
  __half* sU = sA + BT * LDA;                        // ОКНО: sK -> половина sH -> sV
  __half* sK = sU;                                   // [BT][LDQ]
  __half* sHh = sU;                                  // [DV][LDH]
  __half* sV = sU;                                   // [BT][LDV]
  __shared__ float sG[BT];

  const int it = blockIdx.x, ih = blockIdx.y;
  const int NT = (T + BT - 1) / BT;
  if (it >= NT) return;
  const int t0 = it * BT, rows = min(BT, T - t0), rlast = rows - 1;
  const int ihg = ih / (H / Hg);                     // голова ключа для GQA-подобного отображения
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int ro = wm * SM, co = wn * SN, ca = wn * SNA;
  const int rb = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);   // wmma_acc_row, база полосы
  const int cb = (lane & 2) | ((lane & 4) << 1);                 // wmma_acc_col, база полосы

  constexpr int kQK = BT * (DK / 8) / nthr;          // uint4 на нить: 4
  constexpr int kHH = DV * (DKH / 8) / nthr;         // 4 -- ПОЛОВИНА h
  constexpr int kVR = BT * (DV / 8) / nthr;          // 4
  static_assert(kQK * nthr == BT * (DK / 8) && kHH * nthr == DV * (DKH / 8) &&
                    kVR * nthr == BT * (DV / 8), "плитки делятся на нити нацело");
  const long hbase = ((long)it * H + ih) * DV;

  auto ldg_h = [&](uint4 (&dst)[kHH], int half) {
#pragma unroll
    for (int t = 0; t < kHH; ++t) {
      const int i = tid + t * nthr, r = i / (DKH / 8), c = (i % (DKH / 8)) * 8;
      dst[t] = *reinterpret_cast<const uint4*>(&h[(hbase + r) * DK + half * DKH + c]);
    }
  };
  auto sts_h = [&](const uint4 (&src)[kHH]) {
#pragma unroll
    for (int t = 0; t < kHH; ++t) {
      const int i = tid + t * nthr, r = i / (DKH / 8), c = (i % (DKH / 8)) * 8;
      *reinterpret_cast<uint4*>(&sHh[r * LDH + c]) = src[t];
    }
  };

  // --- ПРОЛОГ. ХВОСТ ЧАНКА -- ЗАЖИМОМ СТРОКИ, А НЕ ПРЕДИКАТОМ: предикат вносит ветвь в каждую
  // итерацию и мешает ptxas собрать загрузки в пачку. Зажатая строка даёт КОНЕЧНЫЙ мусор, а он
  // гасится ровно там же, где и раньше: A[t][s] обнуляется при t >= rows или s >= rows, строки
  // выхода при r >= rows не пишутся, а V умножается на уже обнулённые элементы A.
  uint4 qr[kQK], kr[kQK], h0[kHH], h1[kHH], vr[kVR];
#pragma unroll
  for (int t = 0; t < kQK; ++t) {
    const int i = tid + t * nthr, r = min(i / (DK / 8), rlast), c = (i % (DK / 8)) * 8;
    qr[t] = *reinterpret_cast<const uint4*>(&q[((long)(t0 + r) * Hg + ihg) * DK + c]);
    kr[t] = *reinterpret_cast<const uint4*>(&k[((long)(t0 + r) * Hg + ihg) * DK + c]);
  }
  ldg_h(h0, 0);
  if (tid < BT) sG[tid] = g[(long)(t0 + min(tid, rlast)) * H + ih];
#pragma unroll
  for (int t = 0; t < kQK; ++t) {
    const int i = tid + t * nthr, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
    *reinterpret_cast<uint4*>(&sQ[r * LDQ + c]) = qr[t];
    *reinterpret_cast<uint4*>(&sK[r * LDQ + c]) = kr[t];
  }
  __syncthreads();

  // --- G2 (окно = sK). Вторая половина h запрашивается ЗДЕСЬ: её задержка уйдёт под G2 и под первую
  // половину G1, а живой диапазон её регистров не пересечётся с первой половиной.
  ldg_h(h1, 1);
  WarpMma<MB, NBA, BLayout::kKMajor> ma;
  ma.clear();
  ma.template accumulate<DK>(sQ, LDQ, ro, sK, LDQ, ca, lane);

  // --- гейты и причинная маска ВНУТРИ чанка: b_A[t][s] *= e^{g_t-g_s}, ноль при s > t.
  // Поэлементная поправка по накопителю -- свободный слот FP32 под HMMA (§4p). Перебор (i,j,reg)
  // ЯВНЫЙ, а не через visit: индексы обязаны быть константами компиляции, иначе массивы состояния
  // уезжают в локальную память (§43 плейбука), и это не лечится освобождением регистров.
  // Экспонента берётся по ЧЕТЫРЁМ строкам нити, а не по всем 16 ячейкам: EX2 идёт на четверти темпа.
  {
    float gt[MB][2];
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int s = 0; s < 2; ++s) gt[i][s] = sG[ro + i * 16 + rb + 2 * s];
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NBA; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; rg += 2) {          // пара (rg,rg+1) -- СОСЕДНИЕ столбцы одной строки
          const int sub = (rg >> 1) & 1;
          const int tt = ro + i * 16 + rb + 2 * sub;
          const int s0 = ca + j * 16 + cb + 4 * ((rg >> 2) & 1);
          const float e0 = ma.acc[i][j][rg] * __expf(gt[i][sub] - sG[s0]);
          const float e1 = ma.acc[i][j][rg + 1] * __expf(gt[i][sub] - sG[s0 + 1]);
          const bool ok = tt < rows;
          *reinterpret_cast<__half2*>(&sA[(long)tt * LDA + s0]) = __floats2half2_rn(
              (ok && s0 <= tt && s0 < rows) ? e0 : 0.f,
              (ok && s0 + 1 <= tt && s0 + 1 < rows) ? e1 : 0.f);
        }
  }
  __syncthreads();                                   // sA видна; окно свободно

  // --- G1 ПОЛОВИНАМИ ПО k. Смещение по k у операнда A -- это просто сдвиг базы (sQ + DKH), потому
  // что A k-мажорный: a_base = sQ + (row0 + a_row(lane))*LDQ, и прибавка к указателю двигает k.
  sts_h(h0);
  __syncthreads();
  WarpMma<MB, NB, BLayout::kKMajor> mo;
  mo.clear();
  mo.template accumulate<DKH>(sQ, LDQ, ro, sHh, LDH, co, lane);
  __syncthreads();                                   // окно свободно
  sts_h(h1);
#pragma unroll
  for (int t = 0; t < kVR; ++t) {                    // V запрашивается под вторую половину G1
    const int i = tid + t * nthr, r = min(i / (DV / 8), rlast), c = (i % (DV / 8)) * 8;
    vr[t] = *reinterpret_cast<const uint4*>(&v[((long)(t0 + r) * H + ih) * DV + c]);
  }
  __syncthreads();
  mo.template accumulate<DKH>(sQ + DKH, LDQ, ro, sHh, LDH, co, lane);
  __syncthreads();                                   // окно свободно
#pragma unroll
  for (int t = 0; t < kVR; ++t) {
    const int i = tid + t * nthr, r = i / (DV / 8), c = (i % (DV / 8)) * 8;
    *reinterpret_cast<uint4*>(&sV[r * LDV + c]) = vr[t];
  }

  // --- G3: += tril(A) x V. Накопитель G3 ЗАСЕВАЕТСЯ гейтованным результатом G1 (§2a: операнд C
  // бесплатен), поэтому отдельного накопителя и цикла слияния нет -- это 32 регистра и 32 сложения
  // даром. Раскладка аккумулятора у всех четырёх вариантов инструкции ОДНА И ТА ЖЕ (замерено, см.
  // шапку volta_warp_mma.h), поэтому пересыпать значения между объектами законно.
  WarpMma<MB, NB, BLayout::kNMajor> mv;
  {
    float eg[MB][2];
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int s = 0; s < 2; ++s) eg[i][s] = __expf(sG[ro + i * 16 + rb + 2 * s]);
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; ++rg) mv.acc[i][j][rg] = mo.acc[i][j][rg] * eg[i][(rg >> 1) & 1];
  }
  __syncthreads();                                   // sV видна
  mv.template accumulate<BT>(sA, LDA, ro, sV, LDV, co, lane);

  // --- ЗАПИСЬ ЧЕРЕЗ РАЗДЕЛЯЕМУЮ. Раскладка накопителя даёт РАЗБРОС, а глобальная память любит
  // сплошные 16 Б. Прямая запись стоила 32 STG.E.U16 на нить: полоса варпа кроет 8 строк по 4
  // РАЗРОЗНЕННЫХ полуслова, то есть 8 секторов на инструкцию при 64 полезных байтах -- четырёхкратное
  // раздутие при вчетверо большем числе инструкций. ЗАМЕРЕНО ОТКЛЮЧЕНИЕМ: она стоила 0.066 из
  // 0.217 мс, то есть 30% ядра при 17% трафика; через разделяемую -- 4 STG.128 на нить.
  // Пары регистров (0,1) (2,3) (4,5) (6,7) -- СОСЕДНИЕ СТОЛБЦЫ одной строки (карта wmma_acc_col),
  // поэтому в разделяемую кладём __half2: 16 STS вместо 32.
  __half* sO = sQ;                                   // sQ МЕРТВА: последнее чтение -- в G1
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int rg = 0; rg < 8; rg += 2) {
        const int r = i * 16 + rb + 2 * ((rg >> 1) & 1);
        const int c = j * 16 + cb + 4 * ((rg >> 2) & 1);
        *reinterpret_cast<__half2*>(&sO[(long)(ro + r) * LDQ + co + c]) =
            __floats2half2_rn(mv.acc[i][j][rg] * scale, mv.acc[i][j][rg + 1] * scale);
      }
  __syncthreads();
  for (int i = tid; i < BT * (DV / 8); i += nthr) {
    const int r = i / (DV / 8), c = (i % (DV / 8)) * 8;
    if (r < rows)
      *reinterpret_cast<uint4*>(&o[((long)(t0 + r) * H + ih) * DV + c]) =
          *reinterpret_cast<const uint4*>(&sO[r * LDQ + c]);
  }
}

// =================================================================================================
// ОДНОБЛОЧНАЯ РЕДАКЦИЯ (77 КБ): оставлена как замеренная опора сравнения, а не как рабочий путь
// =================================================================================================
// Та же математика и то же расписание, но h лежит в разделяемой ЦЕЛИКОМ, поэтому блок один на SM.
// Медленнее на всех формах (T=15680: 0.163 против 0.143), и разница -- ровно цена второго блока.
// Держать её стоит по двум причинам: она на четыре барьера короче (полезно, если DK вырастет и
// половинная подача перестанет делиться), и она нужна как база для будущих A/B.
template <int BT, int DK, int DV, int WM, int WN>
__global__ __launch_bounds__(WM * WN * 32, 1) void volta_gdn_chunk_o_wide(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ h, const float* __restrict__ g, __half* __restrict__ o,
    int T, int H, int Hg, float scale) {
  constexpr int PAD = 8;
  constexpr int LDQ = DK + PAD, LDV = DV + PAD, LDA = BT + PAD;
  constexpr int SM = BT / WM, SN = DV / WN, SNA = BT / WN;
  constexpr int MB = SM / 16, NB = SN / 16, NBA = SNA / 16;
  constexpr int nthr = WM * WN * 32;
  static_assert(SM % 16 == 0 && SN % 16 == 0 && SNA % 16 == 0, "плитки кратны шагу фрагмента");
  static_assert(DK == DV, "буфер выхода переиспользует sQ");

  extern __shared__ __half smem[];
  __half* sQ = smem;
  __half* sH = sQ + BT * LDQ;
  __half* sK = sH + DV * LDQ;
  __half* sA = sK + BT * LDQ;
  __half* sV = sK;                                   // то же место, что sK
  __shared__ float sG[BT];

  const int it = blockIdx.x, ih = blockIdx.y;
  const int NT = (T + BT - 1) / BT;
  if (it >= NT) return;
  const int t0 = it * BT, rows = min(BT, T - t0), rlast = rows - 1;
  const int ihg = ih / (H / Hg);
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int ro = wm * SM, co = wn * SN, ca = wn * SNA;
  const int rb = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);
  const int cb = (lane & 2) | ((lane & 4) << 1);

  constexpr int kQK = BT * (DK / 8) / nthr, kHR = DV * (DK / 8) / nthr, kVR = BT * (DV / 8) / nthr;
  static_assert(kQK * nthr == BT * (DK / 8) && kHR * nthr == DV * (DK / 8) &&
                    kVR * nthr == BT * (DV / 8), "плитки делятся на нити нацело");
  uint4 qr[kQK], kr[kQK], hr[kHR], vr[kVR];
#pragma unroll
  for (int t = 0; t < kQK; ++t) {
    const int i = tid + t * nthr, r = min(i / (DK / 8), rlast), c = (i % (DK / 8)) * 8;
    qr[t] = *reinterpret_cast<const uint4*>(&q[((long)(t0 + r) * Hg + ihg) * DK + c]);
    kr[t] = *reinterpret_cast<const uint4*>(&k[((long)(t0 + r) * Hg + ihg) * DK + c]);
  }
#pragma unroll
  for (int t = 0; t < kVR; ++t) {
    const int i = tid + t * nthr, r = min(i / (DV / 8), rlast), c = (i % (DV / 8)) * 8;
    vr[t] = *reinterpret_cast<const uint4*>(&v[((long)(t0 + r) * H + ih) * DV + c]);
  }
#pragma unroll
  for (int t = 0; t < kHR; ++t) {
    const int i = tid + t * nthr, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
    hr[t] = *reinterpret_cast<const uint4*>(&h[(((long)it * H + ih) * DV + r) * DK + c]);
  }
  if (tid < BT) sG[tid] = g[(long)(t0 + min(tid, rlast)) * H + ih];
#pragma unroll
  for (int t = 0; t < kQK; ++t) {
    const int i = tid + t * nthr, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
    *reinterpret_cast<uint4*>(&sQ[r * LDQ + c]) = qr[t];
    *reinterpret_cast<uint4*>(&sK[r * LDQ + c]) = kr[t];
  }
#pragma unroll
  for (int t = 0; t < kHR; ++t) {
    const int i = tid + t * nthr, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
    *reinterpret_cast<uint4*>(&sH[r * LDQ + c]) = hr[t];
  }
  __syncthreads();

  WarpMma<MB, NBA, BLayout::kKMajor> ma;
  ma.clear();
  ma.template accumulate<DK>(sQ, LDQ, ro, sK, LDQ, ca, lane);
  {
    float gt[MB][2];
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int s = 0; s < 2; ++s) gt[i][s] = sG[ro + i * 16 + rb + 2 * s];
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NBA; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; rg += 2) {          // пара (rg,rg+1) -- СОСЕДНИЕ столбцы одной строки
          const int sub = (rg >> 1) & 1;
          const int tt = ro + i * 16 + rb + 2 * sub;
          const int s0 = ca + j * 16 + cb + 4 * ((rg >> 2) & 1);
          const float e0 = ma.acc[i][j][rg] * __expf(gt[i][sub] - sG[s0]);
          const float e1 = ma.acc[i][j][rg + 1] * __expf(gt[i][sub] - sG[s0 + 1]);
          const bool ok = tt < rows;
          *reinterpret_cast<__half2*>(&sA[(long)tt * LDA + s0]) = __floats2half2_rn(
              (ok && s0 <= tt && s0 < rows) ? e0 : 0.f,
              (ok && s0 + 1 <= tt && s0 + 1 < rows) ? e1 : 0.f);
        }
  }
  __syncthreads();
#pragma unroll
  for (int t = 0; t < kVR; ++t) {
    const int i = tid + t * nthr, r = i / (DV / 8), c = (i % (DV / 8)) * 8;
    *reinterpret_cast<uint4*>(&sV[r * LDV + c]) = vr[t];
  }
  WarpMma<MB, NB, BLayout::kKMajor> mo;
  mo.clear();
  mo.template accumulate<DK>(sQ, LDQ, ro, sH, LDQ, co, lane);
  WarpMma<MB, NB, BLayout::kNMajor> mv;
  {
    float eg[MB][2];
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int s = 0; s < 2; ++s) eg[i][s] = __expf(sG[ro + i * 16 + rb + 2 * s]);
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; ++rg) mv.acc[i][j][rg] = mo.acc[i][j][rg] * eg[i][(rg >> 1) & 1];
  }
  __syncthreads();
  mv.template accumulate<BT>(sA, LDA, ro, sV, LDV, co, lane);
  __half* sO = sQ;
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int rg = 0; rg < 8; rg += 2) {
        const int r = i * 16 + rb + 2 * ((rg >> 1) & 1);
        const int c = j * 16 + cb + 4 * ((rg >> 2) & 1);
        *reinterpret_cast<__half2*>(&sO[(long)(ro + r) * LDQ + co + c]) =
            __floats2half2_rn(mv.acc[i][j][rg] * scale, mv.acc[i][j][rg + 1] * scale);
      }
  __syncthreads();
  for (int i = tid; i < BT * (DV / 8); i += nthr) {
    const int r = i / (DV / 8), c = (i % (DV / 8)) * 8;
    if (r < rows)
      *reinterpret_cast<uint4*>(&o[((long)(t0 + r) * H + ih) * DV + c]) =
          *reinterpret_cast<const uint4*>(&sO[r * LDQ + c]);
  }
}

// РАЗМЕР РАЗДЕЛЯЕМОЙ ПАМЯТИ ДЛЯ ЗАПУСКА (Б). Считать по этим функциям, а не вручную: у рабочего
// ядра окно объединено, и наивная сумма буферов даёт 77 КБ вместо 44.
template <int BT, int DK, int DV>
constexpr size_t gdn_smem_bytes() {
  constexpr int PAD = 8;
  return (size_t)(BT * (DK + PAD) + BT * (BT + PAD) + DV * (DK / 2 + PAD)) * sizeof(__half);
}
template <int BT, int DK, int DV>
constexpr size_t gdn_smem_bytes_wide() {
  constexpr int PAD = 8;
  return (size_t)(BT * (DK + PAD) + DV * (DK + PAD) + BT * (DK + PAD) + BT * (BT + PAD)) * sizeof(__half);
}

}  // namespace vs2
}  // namespace gdn
}  // namespace fa2_sm70
