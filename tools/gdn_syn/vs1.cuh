// СГЕНЕРИРОВАНО tools/gdn_syn/mkvar.py -- НЕ ПРАВИТЬ РУКАМИ
// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// ВЫХОД ЧАНКОВОГО GATED-DELTA-NET НА ТЕНЗОРНЫХ ЯДРАХ VOLTA.  Замена chunk_fwd_o из fla/Triton.
//
// ЗАЧЕМ. Замерено (§4qq): Triton при цели sm_70 НЕ ЭМИТИТ тензорных инструкций вообще -- в PTX
// mma.sync 0, wmma 0, hmma 0 и 1024 fma.rn.f32, то есть tl.dot компилируется в цикл FP32-FMA на
// обычных ядрах. Потолок 3.5 TFLOP/s против 83 у cuBLAS и 44 у наших рукописных ядер. Этот шаг
// измеренно идёт на 2.85 TFLOP/s -- ровно на том потолке -- и составляет 40.8% скана.
//
// ЧТО СЧИТАЕТСЯ (ровно как в chunk_fwd_kernel_o, чанк BT=64):
//     o[t][v] = scale * ( e^{g_t} * SUM_k q[t][k] h[v][k]
//                       + SUM_{s<=t} e^{g_t - g_s} * (SUM_k q[t][k] k[s][k]) * v[s][v] )
// Три умножения матриц на пару (чанк, голова):
//     G1: b_o[64,128] = q[64,128] x h^T          (h лежит [V][K], k непрерывен -> операнд n-мажорный)
//     G2: b_A[64, 64] = q[64,128] x k^T          (k лежит [t][k], k непрерывен -> тоже n-мажорный)
//     G3: b_o += tril(b_A)[64,64] x v[64,128]    (v лежит [t][v], v непрерывен -> k-мажорный)
// То есть форма ОДИН В ОДИН наша: G2+G3 -- это QK^T и PV нашего внимания, а G1 -- лишний член
// состояния. Поэтому берётся готовый варповый оператор, а не пишется новый.
//
// РАСКЛАДКА. Один блок = одна пара (чанк, голова); 245 чанков x 24 головы = 5880 блоков на карту,
// машина заполнена с запасом. Восемь варпов: сетка 2x4 по выходу [64,128] -> плитка варпа 32x32,
// накопитель 2x2x8 = 32 регистра; b_A [64,64] на той же сетке -> плитка 32x16, ещё 16 регистров.
//
// РАЗДЕЛЯЕМАЯ ПАМЯТЬ И ПЕРЕИСПОЛЬЗОВАНИЕ. Живут одновременно только sQ, sK, sH (подача) и sA.
// Перекладка V и плитка выхода кладутся ПОВЕРХ буферов подачи, мёртвых к тому моменту, -- поэтому
// ни V, ни выход не стоят НИ ОДНОГО лишнего байта. Барьеры на границах обязательны.
//
// =================================================================================================
// ЗАНЯТОСТЬ: ДВА БЛОКА НА SM. Оба порога надо взять ОДНОВРЕМЕННО -- <= 48 КБ разделяемой И <= 128
// регистров на нить; снятие одного не даёт НИЧЕГО, и это здесь замерено с обеих сторон:
//     45 КБ, 148 рег -> 1 блок (упор в регистры) ... 0.1799 мс
//     77 КБ, 103 рег -> 1 блок (упор в память) .... 0.1669 мс
//     45 КБ, 126 рег -> 2 БЛОКА, 16 варпов ........ 0.1546 мс   <- только когда взяты ОБА
// Порог по памяти берёт подача КУСКАМИ по оси сокращения (77 -> 45 КБ, само по себе БЕСПЛАТНО:
// байты те же, каждый элемент читается один раз). Порог по регистрам берут два хода: G1 и G2
// считаются ОДНИМ проходом с общим операндом A, а глубина развёртки ограничена (UN=8). Полная
// развёртка даёт 148 регистров и теряет второй блок -- то есть развёртка и занятость здесь прямо
// конкурируют, и на этом ядре занятость выигрывает (в отгруженном forward -- наоборот, §4z).
// ТРИ блока (29 КБ, 24 варпа) достижимы только принуждением до 80 регистров, и это РАЗЛИВ: x0.43.
//
// =================================================================================================
// ЧТО ЯДРО ОГРАНИЧИВАЕТ -- ЗАМЕРЕНО ПО ЧАСТЯМ (tools/volta_gdn_o_probe.cu, T=15680, V100 @1530):
//     подача q,k .... 0.0411 мс -> 782 ГБ/с        G1 ......... 0.0342 мс
//     подача h ...... 0.0337 мс -> 952 ГБ/с        G2 ......... 0.0109 мс
//     подача v ...... 0.0168 мс -> 953 ГБ/с        эпилог A ... 0.0198 мс
//     ВСЯ ПОДАЧА .... 0.0930 мс -> 863 ГБ/с        G3 ......... 0.0180 мс
//     ЗАПИСЬ ВЫХОДА . 0.0858 мс -> 187 ГБ/с  <-- 16 МБ полезных стоят как 80 МБ чтений
// Подача УЖЕ на достижимой полосе (841 ГБ/с по docs/VOLTA_SM70.md §2) -- там выигрывать нечего.
// А ЗАПИСЬ ВЫХОДА шла в 4.5 раза хуже полосы, и это была САМАЯ КРУПНАЯ статья после подачи:
// карта аккумулятора Volta даёт варпу 8 РАЗНЫХ СТРОК по 4 НЕСМЕЖНЫХ столбца, то есть каждая из 32
// команд STG.E.U16 на нить трогала 8 секторов ради 64 полезных байт (умножение трафика 4-8x).
// Лечится ПЕРЕКЛАДКОЙ ЧЕРЕЗ РАЗДЕЛЯЕМУЮ: плитка выхода собирается в sQ (она к тому моменту мертва,
// доплаты памятью НОЛЬ) и выносится наружу командами STG.128 по строкам -- полное слияние.
// Это тот же класс дефекта, что "FRAGMENT-ORDER emit" в backward: не объём, а ФОРМА обращения.
//
// РОФЛАЙН ЭТОГО ЯДРА -- ПАМЯТЬ, А НЕ ТЕНЗОРЫ. На T=15680 трафик 96.3 МБ при 4.11 GFLOP, то есть
// 42.7 FLOP/байт при нужных 114 -- предел 96.3 МБ / 841 ГБ/с = 0.115 мс = 36 TFLOP/s. Считать эту
// работу против 93.6 TFLOP/s бессмысленно: до тензорного потолка тут дела нет.
//
// ЛОВУШКА ЗАМЕРА, СТОИВШАЯ ЧАСА: если входы синтетические и мелкие, произведения выходят
// ПОДДЁННОРМАЛЬНЫМИ в fp16, и запись выхода замедляется ВДВОЕ (0.253 -> 0.497 мс на том же коде).
// Синтетика для замера обязана давать НОРМАЛЬНЫЙ диапазон fp16.






namespace fa2_sm70 {
namespace gdn {
namespace vs1 {

// РАСКЛАДКИ ОПЕРАНДА B В НАШЕМ API -- ПРОТИВОПОЛОЖНЫ ИНТУИЦИИ ИМЕНИ, и я на этом уже ошибся:
//   kKMajor: B это [N][K], непрерывен K  -> инструкция .row.col, это случай Q*K^T
//   kNMajor: B это [K][N], непрерывен N  -> инструкция .row.row, это случай P*V
// Отсюда: G1 (h лежит [V][K]) и G2 (k лежит [t][K]) -- ОБА kKMajor; G3 (v лежит [t][V]) -- kNMajor.

// Ключи вариантов. Каждый включается отдельно, чтобы его вклад мерился A/B в одном процессе.
enum : int {
  kOptCoalesce = 1,   // эпилог через разделяемую -> слитая запись STG.128 (см. шапку)
  kOptChunkK   = 2,   // подача кусками по K/2: разделяемая 45 КБ вместо 77 -> порог ДВУХ блоков
  kOptTwoBlk   = 4,   // __launch_bounds__(.., 2): требовать <= 128 регистров
  kOptRowExp   = 8,   // экспоненты гейтов считать ПО СТРОКАМ/СТОЛБЦАМ, а не поэлементно
  kOptChunkK4  = 16,  // подача кусками по K/4: 29 КБ -> порог ТРЁХ блоков (нужно <= 85 регистров)
  kOptThreeBlk = 32,  // __launch_bounds__(.., 3): требовать <= 85 регистров
};
template <int DK, int OPT>
__host__ __device__ constexpr int gdn_kc() {
  return (OPT & kOptChunkK4) ? DK / 4 : ((OPT & kOptChunkK) ? DK / 2 : DK);
}

// Размер разделяемой памяти -- ОДНО МЕСТО, где он считается: и хост, и ядро берут его отсюда,
// иначе рассинхронизация раскладки даёт молчаливую порчу.
template <int BT, int DK, int DV, int OPT>
__host__ __device__ constexpr int gdn_smem_halves() {
  // sQ[BT][KC+8] + sK[BT][KC+8] + sH[DV][KC+8] + sA[BT][BT+8]; плюс низ должен вмещать плитку
  // выхода sO[BT][DV+8] и перекладку sV[BT][DV+8] -- обе живут ПОСЛЕ подачи, в её же памяти.
  constexpr int KC = gdn_kc<DK, OPT>();
  constexpr int feed = BT * (KC + 8) + BT * (KC + 8) + DV * (KC + 8) + BT * (BT + 8);
  constexpr int tail = BT * (DV + 8) + BT * (BT + 8);
  return feed > tail ? feed : tail;
}
template <int BT, int DK, int DV, int OPT = kOptCoalesce | kOptChunkK | kOptRowExp>
__host__ __device__ constexpr size_t gdn_smem_bytes() {
  return (size_t)gdn_smem_halves<BT, DK, DV, OPT>() * sizeof(__half);
}

// -------------------------------------------------------------------------------------------------
// НАКОПЛЕНИЕ ПРЯМО В ЗАДАННЫЙ МАССИВ, А НЕ В ОБЪЕКТ. Даёт три вещи, которых нет у WarpMma::accumulate:
//  1. G3 кладётся В ТОТ ЖЕ накопитель, что G1 -- отдельный объект стоил бы ещё MB*NB*8 регистров
//     (32 при сетке 2x4) плюс цикл поэлементного слияния. Законно потому, что КАРТА АККУМУЛЯТОРА У
//     ВСЕХ ЧЕТЫРЁХ ВАРИАНТОВ ИНСТРУКЦИИ ОДНА И ТА ЖЕ (замерено, шапка volta_warp_mma.h): меняется
//     только подача операндов, а она здесь задана явно.
//  2. ДВА ОПЕРАНДА B НА ОДИН ПРОХОД (G1 и G2 делят операнд A = q): фрагменты A читаются ОДИН раз
//     вместо двух. Загрузок на шаг k было 5 (2 A + 2 B1 + 1 B2) на 6 умножений, стало 4 на 6.
//  3. ГЛУБИНА РАЗВЁРТКИ УПРАВЛЯЕМАЯ (UN): при полной развёртке ptxas держит слишком много
//     короткоживущих временных, и порог 128 регистров недостижим (volta_prims §4z).
// -------------------------------------------------------------------------------------------------
template <int K, int MB, int NB1, int NB2, int UN, BLayout BL>
__device__ __forceinline__ void accum_into(float (&acc1)[MB][NB1][8], float (*acc2)[NB2][8],
                                           const __half* sA, int lda, int row0,
                                           const __half* sB1, int ldb1, int col1,
                                           const __half* sB2, int ldb2, int col2, int lane) {
  const __half* pa = sA + (long)(row0 + wmma_a_row(lane)) * lda;
  const __half* pb1 = (BL == BLayout::kKMajor)
                          ? sB1 + (long)(col1 + wmma_b_col(lane)) * ldb1
                          : sB1 + (long)wmma_b_krow(lane) * ldb1 + col1 + wmma_b_nbase(lane);
  const __half* pb2 = sB2 + (long)(col2 + wmma_b_col(lane)) * ldb2;   // второй операнд -- всегда k-мажорный
  uint32_t fa[2][MB][2], fb1[2][NB1][2], fb2[2][NB2][2];
  auto ldA = [&](uint32_t (&f)[MB][2], int k) {
#pragma unroll
    for (int i = 0; i < MB; ++i) {
      const uint2 t = *reinterpret_cast<const uint2*>(pa + (long)(i * 16) * lda + k);
      f[i][0] = t.x; f[i][1] = t.y;
    }
  };
  auto ldB1 = [&](uint32_t (&f)[NB1][2], int k) {
#pragma unroll
    for (int j = 0; j < NB1; ++j) {
      const __half* p = (BL == BLayout::kKMajor) ? (pb1 + (long)(j * 16) * ldb1 + k)
                                                 : (pb1 + (long)k * ldb1 + j * 16);
      const uint2 t = *reinterpret_cast<const uint2*>(p);
      f[j][0] = t.x; f[j][1] = t.y;
    }
  };
  auto ldB2 = [&](uint32_t (&f)[NB2][2], int k) {
#pragma unroll
    for (int j = 0; j < NB2; ++j) {
      const uint2 t = *reinterpret_cast<const uint2*>(pb2 + (long)(j * 16) * ldb2 + k);
      f[j][0] = t.x; f[j][1] = t.y;
    }
  };
  ldA(fa[0], 0); ldB1(fb1[0], 0);
  if (acc2) ldB2(fb2[0], 0);
#pragma unroll UN
  for (int k = 0; k < K; k += 4) {
    const int c = (k >> 2) & 1;
    if (k + 4 < K) {
      ldA(fa[c ^ 1], k + 4); ldB1(fb1[c ^ 1], k + 4);
      if (acc2) ldB2(fb2[c ^ 1], k + 4);
    }
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB1; ++j) hmma_884_b<BL>(acc1[i][j], fa[c][i], fb1[c][j]);
    if (acc2)
#pragma unroll
      for (int i = 0; i < MB; ++i)
#pragma unroll
        for (int j = 0; j < NB2; ++j)
          hmma_884_b<BLayout::kKMajor>(acc2[i][j], fa[c][i], fb2[c][j]);
  }
}

// BT -- размер чанка (64), DK/DV -- размерности головы, WM x WN -- сетка варпов по выходу,
// UN -- глубина развёртки цикла по оси сокращения (0 = полная).
template <int BT, int DK, int DV, int WM, int WN,
          int OPT = kOptCoalesce | kOptChunkK | kOptRowExp, int UN = 8>
__global__ __launch_bounds__(WM * WN * 32,
                             (OPT & kOptThreeBlk) ? 3 : ((OPT & kOptTwoBlk) ? 2 : 1))
void volta_gdn_chunk_o(
    const __half* __restrict__ q,      // [T, Hg, DK]
    const __half* __restrict__ k,      // [T, Hg, DK]
    const __half* __restrict__ v,      // [T, H,  DV]
    const __half* __restrict__ h,      // [NT, H, DV, DK]   состояние на начало чанка
    const float*  __restrict__ g,      // [T, H]            накопленный логарифм гейта
    __half* __restrict__ o,            // [T, H, DV]
    int T, int H, int Hg, float scale) {
  constexpr int PAD = 8;                       // 8 полуслов = 16 Б: держит LDS.128 выровненной
  constexpr int KC  = gdn_kc<DK, OPT>();       // ширина куска подачи по оси сокращения
  constexpr bool kChunk = KC < DK;
  constexpr int LDQ = KC + PAD;                // шаг строки sQ/sK/sH
  constexpr int LDV = DV + PAD, LDA = BT + PAD, LDO = DV + PAD;
  constexpr int SM = BT / WM, SN = DV / WN;    // плитка варпа по выходу
  constexpr int SNA = BT / WN;                 // плитка варпа по b_A
  constexpr int MB = SM / 16, NB = SN / 16, NBA = SNA / 16;
  static_assert(SM % 16 == 0 && SN % 16 == 0 && SNA % 16 == 0, "плитки кратны шагу фрагмента");
  static_assert(LDQ * 2 % 16 == 0 && LDV * 2 % 16 == 0 && LDO * 2 % 16 == 0,
                "шаг строки кратен 16 Б, иначе широкие обращения незаконны (volta_prims §4b)");

  // ПОРЯДОК БУФЕРОВ ЗНАЧИМ: sQ и sK лежат РЯДОМ, потому что после последнего куска подачи оба
  // мертвы и вместе дают место под плитку выхода sO (BT*LDO полуслов) без единого лишнего байта.
  extern __shared__ __half smem[];
  __half* sQ = smem;                    // [BT][LDQ]
  __half* sK = sQ + BT * LDQ;           // [BT][LDQ]   -- при полной подаче перекрывается sV
  __half* sH = sK + BT * LDQ;           // [DV][LDQ]   -- h как [v][k]; при kOptChunkK -- место sV
  __half* sA = sH + DV * LDQ;           // [BT][LDA]   -- tril(A) в fp16 для G3
  // sV и sO ложатся на буферы подачи, мёртвые к этому моменту. sV -- начиная с sH, если он
  // достаточно велик, иначе с самого начала; sO -- всегда с начала.
  constexpr int kOffH = 2 * BT * LDQ;                    // смещение sH в полусловах
  constexpr bool kVatH = BT * LDV <= DV * LDQ;
  __half* sV = kVatH ? sH : sQ;
  __half* sO = sQ;
  // Если sO накрывает sV, между последним чтением V и записью выхода нужен барьер.
  constexpr bool kNeedBarOV = !kVatH || (BT * LDO > kOffH);
  constexpr int kOffA = 2 * BT * LDQ + DV * LDQ;         // sA начинается здесь и переживает всё
  static_assert(BT * LDO <= kOffA, "плитка выхода не помещается в мёртвые буферы подачи");
  static_assert(BT * LDV <= (kVatH ? DV * LDQ : kOffA), "перекладка V не помещается");
  __shared__ float sG[BT];

  const int it = blockIdx.x, ih = blockIdx.y;
  const int NT = (T + BT - 1) / BT;
  if (it >= NT) return;
  const int t0 = it * BT, rows = min(BT, T - t0);
  const int ihg = ih / (H / Hg);                       // голова ключа для GQA-подобного отображения

  const int tid = threadIdx.x, nthr = WM * WN * 32;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int ro = wm * SM, co = wn * SN, ca = wn * SNA;

  for (int i = tid; i < BT; i += nthr) sG[i] = (t0 + i) < T ? g[(long)(t0 + i) * H + ih] : 0.f;

  WarpMma<MB, NB,  BLayout::kKMajor> mo;
  WarpMma<MB, NBA, BLayout::kKMajor> ma;
  mo.clear(); ma.clear();

  // --- подача и G1/G2. Каждая строка читается ЦЕЛИКОМ широкими обращениями (§1c: MIO считает
  // инструкции). При kOptChunkK ось сокращения проходится КУСКАМИ: разделяемой нужно вдвое меньше,
  // ценой лишней пары барьеров на кусок -- сами байты те же, каждый элемент читается ОДИН раз.
#pragma unroll 1
  for (int k0 = 0; k0 < DK; k0 += KC) {
    if (k0) __syncthreads();                           // все дочитали предыдущий кусок
    for (int i = tid; i < BT * (KC / 8); i += nthr) {
      const int r = i / (KC / 8), c = (i % (KC / 8)) * 8;
      const bool ok = (t0 + r) < T;
      *reinterpret_cast<uint4*>(&sQ[r * LDQ + c]) =
          ok ? *reinterpret_cast<const uint4*>(&q[((long)(t0 + r) * Hg + ihg) * DK + k0 + c])
             : uint4{0, 0, 0, 0};
      *reinterpret_cast<uint4*>(&sK[r * LDQ + c]) =
          ok ? *reinterpret_cast<const uint4*>(&k[((long)(t0 + r) * Hg + ihg) * DK + k0 + c])
             : uint4{0, 0, 0, 0};
    }
    for (int i = tid; i < DV * (KC / 8); i += nthr) {
      const int r = i / (KC / 8), c = (i % (KC / 8)) * 8;
      *reinterpret_cast<uint4*>(&sH[r * LDQ + c]) =
          *reinterpret_cast<const uint4*>(&h[(((long)it * H + ih) * DV + r) * DK + k0 + c]);
    }
    __syncthreads();
    // G1 и G2: у обоих B имеет вид [N][K] с непрерывным K, то есть kKMajor. Считаются ОДНИМ
    // проходом -- операнд A (это q) у них общий, и читать его дважды незачем.
    accum_into<KC, MB, NB, NBA, UN ? UN : KC / 4, BLayout::kKMajor>(
        mo.acc, ma.acc, sQ, LDQ, ro, sH, LDQ, co, sK, LDQ, ca, lane);
  }

  // --- гейты и причинная маска ВНУТРИ чанка.
  // b_o *= e^{g_t};  b_A[t][s] *= e^{g_t - g_s} и обнуляется при s > t.
  // Обе поправки поэлементны по накопителю -- то есть свободный слот FP32 под HMMA (§4p).
  if (OPT & kOptRowExp) {
    // Экспонента -- SFU на ЧЕТВЕРТЬ темпа (§4e), и поэлементно её звали 48 раз на нить. Но карта
    // аккумулятора даёт на нить всего ЧЕТЫРЕ разных строки и ЧЕТЫРЕ разных столбца на плитку:
    // строка = 16i + ((l&1)|((l&16)>>2)|(l&8)) + 2*((rg>>1)&1), столбец = 16j + ... + (rg&1) +
    // 4*((rg>>2)&1). Значит e^{g} считается ПО СТРОКАМ и ПО СТОЛБЦАМ, а произведение
    // e^{g_t-g_s} = e^{g_t} * e^{-g_s} разлагается -- 48 вызовов SFU становятся 4+4.
    // СТРОКА зависит от rg ОДНИМ битом ((rg>>1)&1 -> +2), СТОЛБЕЦ -- ДВУМЯ ((rg&1) -> +1 и
    // ((rg>>2)&1) -> +4). Первый заход учитывал у столбца только старший бит, и ошибка выросла
    // 3.0e-4 -> 4.8e-3: карту аккумулятора надо брать целиком, а не "по смыслу".
    float er[MB][2], ec[NBA][4];
    const int rb = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);
    const int cb = (lane & 2) | ((lane & 4) << 1);
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int tt = ro + i * 16 + rb + 2 * u;
        er[i][u] = __expf(sG[tt]);
      }
#pragma unroll
    for (int j = 0; j < NBA; ++j)
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int ss = ca + j * 16 + cb + (u & 1) + 4 * (u >> 1);
        ec[j][u] = __expf(-sG[ss]);
      }
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; ++rg) mo.acc[i][j][rg] *= er[i][(rg >> 1) & 1];
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NBA; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; ++rg) {
          const int tt = ro + i * 16 + rb + 2 * ((rg >> 1) & 1);
          const int ss = ca + j * 16 + cb + (rg & 1) + 4 * ((rg >> 2) & 1);
          const int u = (rg & 1) + 2 * ((rg >> 2) & 1);
          ma.acc[i][j][rg] = (ss <= tt && tt < rows && ss < rows)
                                 ? ma.acc[i][j][rg] * er[i][(rg >> 1) & 1] * ec[j][u]
                                 : 0.f;
        }
  } else {
    mo.visit(lane, [&](float& x, int r, int c) { x *= __expf(sG[ro + r]); });
    ma.visit(lane, [&](float& x, int r, int c) {
      const int tt = ro + r, ss = ca + c;
      x = (ss <= tt && tt < rows && ss < rows) ? x * __expf(sG[tt] - sG[ss]) : 0.f;
    });
  }

  // --- tril(A) в разделяемую (в fp16, как операнд третьего умножения). Регистры rg и rg+1 при
  // чётном rg -- ОДНА строка и СОСЕДНИЕ столбцы, поэтому пара кладётся одной STS.32: вдвое меньше
  // обращений и вдвое меньше конверсий (F2F -- самая дорогая операция, §4e).
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NBA; ++j)
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const int rg = 2 * p;
        const int r = ro + i * 16 + wmma_acc_row(lane, rg);
        const int c = ca + j * 16 + wmma_acc_col(lane, rg);
        *reinterpret_cast<__half2*>(&sA[(long)r * LDA + c]) =
            __floats2half2_rn(ma.acc[i][j][rg], ma.acc[i][j][rg + 1]);
      }
  __syncthreads();                       // sK/sH больше не нужны -> их место занимает sV

  for (int i = tid; i < BT * (DV / 8); i += nthr) {
    const int r = i / (DV / 8), c = (i % (DV / 8)) * 8;
    const bool ok = (t0 + r) < T;
    *reinterpret_cast<uint4*>(&sV[r * LDV + c]) =
        ok ? *reinterpret_cast<const uint4*>(&v[((long)(t0 + r) * H + ih) * DV + c])
           : uint4{0, 0, 0, 0};
  }
  __syncthreads();

  // --- G3: += tril(A) x V. V лежит [t][v] с непрерывным v, то есть [K][N] -> kNMajor.
  // Накапливается ПРЯМО в mo (см. accum_into): ни второго накопителя, ни цикла слияния.
  accum_into<BT, MB, NB, NBA, UN ? UN : BT / 4, BLayout::kNMajor>(
      mo.acc, (float(*)[NBA][8])nullptr, sA, LDA, ro, sV, LDV, co, sK, LDQ, 0, lane);

  // --- запись
  if (OPT & kOptCoalesce) {
    // Плитка собирается в разделяемой в СТРОЧНОМ порядке, а наружу идёт STG.128 по строкам.
    // Прямая запись из накопителя даёт на команду 8 разных строк по 4 несмежных столбца -- 8
    // секторов ради 64 полезных байт; замерено 187 ГБ/с против 863 у подачи (см. шапку).
    if (kNeedBarOV) __syncthreads();     // при узком куске sO ложится ПОВЕРХ ещё читаемой sV
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const int rg = 2 * p;
          const int r = ro + i * 16 + wmma_acc_row(lane, rg);
          const int c = co + j * 16 + wmma_acc_col(lane, rg);
          *reinterpret_cast<__half2*>(&sO[(long)r * LDO + c]) =
              __floats2half2_rn(mo.acc[i][j][rg] * scale, mo.acc[i][j][rg + 1] * scale);
        }
    __syncthreads();
    for (int i = tid; i < BT * (DV / 8); i += nthr) {
      const int r = i / (DV / 8), c = (i % (DV / 8)) * 8;
      if (t0 + r < T)
        *reinterpret_cast<uint4*>(&o[((long)(t0 + r) * H + ih) * DV + c]) =
            *reinterpret_cast<const uint4*>(&sO[r * LDO + c]);
    }
  } else {
    mo.visit(lane, [&](float& x, int r, int c) {
      const int tt = ro + r;
      if (tt < rows) o[((long)(t0 + tt) * H + ih) * DV + co + c] = __float2half(x * scale);
    });
  }
}

}  // namespace vs1
}  // namespace gdn
}  // namespace fa2_sm70
