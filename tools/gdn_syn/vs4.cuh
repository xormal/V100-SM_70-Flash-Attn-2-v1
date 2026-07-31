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
// РАЗДЕЛЯЕМАЯ ПАМЯТЬ И ПЕРЕИСПОЛЬЗОВАНИЕ. sQ, sK, sH нужны в G1/G2; sV -- только в G3, когда sK уже
// не нужен, поэтому sV кладётся ПОВЕРХ sK (барьер между ними обязателен). Итог 77.0 КБ из 96.
//
// -------------------------------------------------------------------------------------------------
// ГДЕ ВРЕМЯ (разбор по фазам, T=15680, отсечением кусков, tools/gdn_ablate.cuh; V100 @1530):
//     полное ядро .................................. 0.216 мс
//     только подача глобальная->разделяемая ........ 0.097 мс   45%
//     без записи выхода ............................ 0.137 мс   -> ЗАПИСЬ ВЫХОДА 0.079 мс = 37%
//     БЕЗ ЕДИНОЙ ТЕНЗОРНОЙ ИНСТРУКЦИИ .............. 0.173 мс   80% времени НЕ тензорное
// То есть ядро упирается НЕ в тензорный конвейер (18.9 из 93.6 TFLOP/s -- это 20% пика, и потолок
// тут вовсе не 93.6): при 42.6 флопа на байт трафика крыша по памяти даёт ~35 TFLOP/s, и мы на 54%
// от неё. Рычаг искать в ТРАНСПОРТЕ, а не в умножениях.
//
// ЗАПИСЬ ВЫХОДА ЧЕРЕЗ РАЗДЕЛЯЕМУЮ ПАМЯТЬ (главная правка, x1.18).  Раскладка накопителя даёт лейну
// лишь ДВА соседних столбца (регистры 2u и 2u+1 отличаются младшим битом столбца), а строки внутри
// одной инструкции расходятся на восемь разных строк выхода. Значит прямая запись наружу трогает
// 32-байтовый сектор строки ВОСЕМЬЮ разными обращениями, каждое из которых пишет его частично:
// на 512 нужных секторов приходится 4096 транзакций. Складываем плитку в sQ (он мёртв с конца G2 --
// лишнего места не нужно, 64x136 полуслов это ровно его размер) и выносим блоком по 16 Б на нить:
// каждый сектор пишется РОВНО ОДИН РАЗ.  Замерено: 0.216 -> 0.184 мс.
//   Промежуточный шаг (парная запись half2 прямо наружу, без разделяемой) даёт 0.195 -- то есть
//   половину выигрыша берёт удвоение ширины обращения, а вторую половину -- именно слитность.
//
// ПРОВЕРЕНО И ОТВЕРГНУТО (числа -- в отчёте по задаче, стенд tools/gdn_bench.cu):
//   * A ИЗ РЕГИСТРОВ вместо круга через разделяемую (перекладка накопителя G2 в операнд G3
//     двумя ступенями бабочки, 24 SHFL): работает и точна, но МЕДЛЕННЕЕ -- x0.77. Причина не в
//     цене обмена: регистровая передача требует, чтобы варп держал ВСЕ 64 столбца A для своих
//     строк (межварповых обменов на Volta нет), то есть плитку 16 строк вместо 32x32, а это само
//     по себе x0.87 по закону геометрии плитки. При РАВНОЙ геометрии обмен и круг сходятся в ноль.
//   * УДЕШЕВЛЕНИЕ КРУГА (STS.32 вместо двух STS.16 и LDS.128 вместо двух LDS.64 -- вдвое меньше
//     обращений): РОВНО НОЛЬ (0.2209 против 0.2202). Круг не на критическом пути, и это же
//     объясняет, почему его снятие ничего не даёт.
//   * sV ОТДЕЛЬНО от sK, чтобы убрать барьер между G2 и G3 (94 КБ разделяемой): x0.94, хуже.






namespace fa2_sm70 {
namespace gdn {
namespace vs4 {

// РАСКЛАДКИ ОПЕРАНДА B В НАШЕМ API -- ПРОТИВОПОЛОЖНЫ ИНТУИЦИИ ИМЕНИ, и я на этом уже ошибся:
//   kKMajor: B это [N][K], непрерывен K  -> инструкция .row.col, это случай Q*K^T
//   kNMajor: B это [K][N], непрерывен N  -> инструкция .row.row, это случай P*V
// Отсюда: G1 (h лежит [V][K]) и G2 (k лежит [t][K]) -- ОБА kKMajor; G3 (v лежит [t][V]) -- kNMajor.

// Третье умножение копится ПРЯМО В НАКОПИТЕЛЬ ПЕРВОГО (передан по ссылке): отдельный WarpMma под
// G3 и слияние 32 сложениями не нужны -- это 36 регистров (сам по себе приём даёт 206 -> 170) и 32
// инструкции даром. По времени приём НЕЙТРАЛЕН (x0.99-1.00), взят ради регистров; итоговое ядро
// стоит 202 регистра -- перекладка выхода часть сэкономленного забирает обратно.
template <int MB, int NB, int K>
__device__ __forceinline__ void gdn_g3(float (&acc)[MB][NB][8],
                                       const __half* sA, int lda, int row0,
                                       const __half* sB, int ldb, int col0, int lane) {
  const __half* pa = sA + (long)(row0 + wmma_a_row(lane)) * lda;
  const __half* pb = sB + (long)wmma_b_krow(lane) * ldb + col0 + wmma_b_nbase(lane);
  uint32_t fa[2][MB][2], fb[2][NB][2];
#pragma unroll
  for (int i = 0; i < MB; ++i) {
    const uint2 v = *reinterpret_cast<const uint2*>(pa + (long)(i * 16) * lda);
    fa[0][i][0] = v.x; fa[0][i][1] = v.y;
  }
#pragma unroll
  for (int j = 0; j < NB; ++j) {
    const uint2 v = *reinterpret_cast<const uint2*>(pb + j * 16);
    fb[0][j][0] = v.x; fb[0][j][1] = v.y;
  }
#pragma unroll
  for (int k = 0; k < K; k += 4) {
    const int c = (k >> 2) & 1;
    if (k + 4 < K) {
      const __half* qa = pa + k + 4;
#pragma unroll
      for (int i = 0; i < MB; ++i) {
        const uint2 v = *reinterpret_cast<const uint2*>(qa + (long)(i * 16) * lda);
        fa[c ^ 1][i][0] = v.x; fa[c ^ 1][i][1] = v.y;
      }
      const __half* p = pb + (long)(k + 4) * ldb;
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        const uint2 v = *reinterpret_cast<const uint2*>(p + j * 16);
        fb[c ^ 1][j][0] = v.x; fb[c ^ 1][j][1] = v.y;
      }
    }
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j) hmma_884_b<BLayout::kNMajor>(acc[i][j], fa[c][i], fb[c][j]);
  }
}

// BT -- размер чанка (64), DK/DV -- размерности головы, WM x WN -- сетка варпов по выходу.
template <int BT, int DK, int DV, int WM, int WN>
__global__ __launch_bounds__(WM * WN * 32, 1) void volta_gdn_chunk_o(
    const __half* __restrict__ q,      // [T, Hg, DK]
    const __half* __restrict__ k,      // [T, Hg, DK]
    const __half* __restrict__ v,      // [T, H,  DV]
    const __half* __restrict__ h,      // [NT, H, DV, DK]   состояние на начало чанка
    const float*  __restrict__ g,      // [T, H]            накопленный логарифм гейта
    __half* __restrict__ o,            // [T, H, DV]
    int T, int H, int Hg, float scale) {
  constexpr int PAD = 8;                       // 8 полуслов = 16 Б: держит LDS.128 выровненной
  constexpr int LDQ = DK + PAD, LDV = DV + PAD, LDA = BT + PAD;
  constexpr int LDO = DV + PAD;                // строка перекладки выхода: 272 Б, кратна 16 (§4a)
  constexpr int SM = BT / WM, SN = DV / WN;    // плитка варпа по выходу
  constexpr int SNA = BT / WN;                 // плитка варпа по b_A
  static_assert(SM % 16 == 0 && SN % 16 == 0 && SNA % 16 == 0, "плитки кратны шагу фрагмента");
  static_assert(BT * LDO <= BT * LDQ, "перекладка выхода должна помещаться в sQ");

  extern __shared__ __half smem[];
  __half* sQ = smem;                    // [BT][LDQ]
  __half* sH = sQ + BT * LDQ;           // [DV][LDQ]   -- h как [v][k]
  __half* sK = sH + DV * LDQ;           // [BT][LDQ]   -- позже перекрывается sV
  __half* sA = sK + BT * LDQ;           // [BT][LDA]   -- tril(A) в fp16 для G3
  __half* sV = sK;                      // ТО ЖЕ МЕСТО, что sK: нужен только после G2 (см. барьер)
  __half* sO = sQ;                      // ТО ЖЕ МЕСТО, что sQ: он мёртв с конца G2
  __shared__ float sG[BT];

  const int it = blockIdx.x, ih = blockIdx.y;
  const int NT = (T + BT - 1) / BT;
  if (it >= NT) return;
  const int t0 = it * BT, rows = min(BT, T - t0);
  const int ihg = ih / (H / Hg);                       // голова ключа для GQA-подобного отображения

  const int tid = threadIdx.x, nthr = WM * WN * 32;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;

  // --- подача. Каждая строка читается ЦЕЛИКОМ широкими обращениями (§1c: MIO считает инструкции).
  for (int i = tid; i < BT * (DK / 8); i += nthr) {
    const int r = i / (DK / 8), c = (i % (DK / 8)) * 8;
    const bool ok = (t0 + r) < T;
    *reinterpret_cast<uint4*>(&sQ[r * LDQ + c]) =
        ok ? *reinterpret_cast<const uint4*>(&q[((long)(t0 + r) * Hg + ihg) * DK + c]) : uint4{0, 0, 0, 0};
    *reinterpret_cast<uint4*>(&sK[r * LDQ + c]) =
        ok ? *reinterpret_cast<const uint4*>(&k[((long)(t0 + r) * Hg + ihg) * DK + c]) : uint4{0, 0, 0, 0};
  }
  for (int i = tid; i < DV * (DK / 8); i += nthr) {
    const int r = i / (DK / 8), c = (i % (DK / 8)) * 8;
    *reinterpret_cast<uint4*>(&sH[r * LDQ + c]) =
        *reinterpret_cast<const uint4*>(&h[(((long)it * H + ih) * DV + r) * DK + c]);
  }
  for (int i = tid; i < BT; i += nthr) sG[i] = (t0 + i) < T ? g[(long)(t0 + i) * H + ih] : 0.f;
  __syncthreads();

  // --- G1 и G2: у обоих B имеет вид [N][K] с непрерывным K, то есть kKMajor.
  WarpMma<SM / 16, SN / 16, BLayout::kKMajor> mo;
  WarpMma<SM / 16, SNA / 16, BLayout::kKMajor> ma;
  mo.clear(); ma.clear();
  mo.template accumulate<DK>(sQ, LDQ, wm * SM, sH, LDQ, wn * SN, lane);
  ma.template accumulate<DK>(sQ, LDQ, wm * SM, sK, LDQ, wn * SNA, lane);

  // --- гейты и причинная маска ВНУТРИ чанка.
  // b_o *= e^{g_t};  b_A[t][s] *= e^{g_t - g_s} и обнуляется при s > t.
  // Обе поправки поэлементны по накопителю -- то есть свободный слот FP32 под HMMA (§4p).
  const int ro = wm * SM, co = wn * SN, ca = wn * SNA;
  mo.visit(lane, [&](float& x, int r, int c) { x *= __expf(sG[ro + r]); });
  ma.visit(lane, [&](float& x, int r, int c) {
    const int tt = ro + r, ss = ca + c;
    x = (ss <= tt && tt < rows && ss < rows) ? x * __expf(sG[tt] - sG[ss]) : 0.f;
  });

  // --- tril(A) в разделяемую (в fp16, как операнд третьего умножения).
  // Парная запись half2 здесь ЗАМЕРЕННО НЕ ДАЁТ НИЧЕГО (0.2209 против 0.2202): круг не на
  // критическом пути. Оставлено как есть, чтобы не тратить регистры на нулевую правку.
  ma.visit(lane, [&](float& x, int r, int c) { sA[(ro + r) * LDA + ca + c] = __float2half(x); });
  __syncthreads();                       // sK больше не нужен -> его место занимает sV

  for (int i = tid; i < BT * (DV / 8); i += nthr) {
    const int r = i / (DV / 8), c = (i % (DV / 8)) * 8;
    const bool ok = (t0 + r) < T;
    *reinterpret_cast<uint4*>(&sV[r * LDV + c]) =
        ok ? *reinterpret_cast<const uint4*>(&v[((long)(t0 + r) * H + ih) * DV + c]) : uint4{0, 0, 0, 0};
  }
  __syncthreads();

  // --- G3: b_o += tril(A) x V. V лежит [t][v] с непрерывным v, то есть [K][N] -> kNMajor.
  // Копится прямо в mo -- см. gdn_g3.
  gdn_g3<SM / 16, SN / 16, BT>(mo.acc, sA, LDA, ro, sV, LDV, co, lane);
#pragma unroll
  for (int i = 0; i < SM / 16; ++i)
#pragma unroll
    for (int j = 0; j < SN / 16; ++j)
#pragma unroll
      for (int r = 0; r < 8; ++r) mo.acc[i][j][r] *= scale;

  // --- ЗАПИСЬ ЧЕРЕЗ РАЗДЕЛЯЕМУЮ (см. шапку). Барьер перед перекладкой НЕ НУЖЕН: sQ последний раз
  // читается в G2, а между G2 и этим местом уже стоят два __syncthreads.
#pragma unroll
  for (int i = 0; i < SM / 16; ++i)
#pragma unroll
    for (int j = 0; j < SN / 16; ++j)
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int r0 = 2 * u;              // регистры r0 и r0+1 -> соседние столбцы -> одна STS.32
        const int rr = ro + i * 16 + wmma_acc_row(lane, r0);
        const int cc = co + j * 16 + wmma_acc_col(lane, r0);
        *reinterpret_cast<__half2*>(&sO[(long)rr * LDO + cc]) =
            __floats2half2_rn(mo.acc[i][j][r0], mo.acc[i][j][r0 + 1]);
      }
  __syncthreads();
  for (int i = tid; i < BT * (DV / 8); i += nthr) {
    const int r = i / (DV / 8), c = (i % (DV / 8)) * 8;
    if (t0 + r < T)
      *reinterpret_cast<uint4*>(&o[((long)(t0 + r) * H + ih) * DV + c]) =
          *reinterpret_cast<const uint4*>(&sO[r * LDO + c]);
  }
}

}  // namespace vs4
}  // namespace gdn
}  // namespace fa2_sm70
