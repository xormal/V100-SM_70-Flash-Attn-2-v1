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
// не нужен, поэтому sV кладётся ПОВЕРХ sK (барьер между ними обязателен). Итог 78.8 КБ из 96.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "volta_warp_mma.h"

namespace fa2_sm70 {
namespace gdn {

// РАСКЛАДКИ ОПЕРАНДА B В НАШЕМ API -- ПРОТИВОПОЛОЖНЫ ИНТУИЦИИ ИМЕНИ, и я на этом уже ошибся:
//   kKMajor: B это [N][K], непрерывен K  -> инструкция .row.col, это случай Q*K^T
//   kNMajor: B это [K][N], непрерывен N  -> инструкция .row.row, это случай P*V
// Отсюда: G1 (h лежит [V][K]) и G2 (k лежит [t][K]) -- ОБА kKMajor; G3 (v лежит [t][V]) -- kNMajor.

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
  constexpr int SM = BT / WM, SN = DV / WN;    // плитка варпа по выходу
  constexpr int SNA = BT / WN;                 // плитка варпа по b_A
  static_assert(SM % 16 == 0 && SN % 16 == 0 && SNA % 16 == 0, "плитки кратны шагу фрагмента");

  extern __shared__ __half smem[];
  __half* sQ = smem;                    // [BT][LDQ]
  __half* sH = sQ + BT * LDQ;           // [DV][LDQ]   -- h как [v][k]
  __half* sK = sH + DV * LDQ;           // [BT][LDQ]   -- позже перекрывается sV
  __half* sA = sK + BT * LDQ;           // [BT][LDA]   -- tril(A) в fp16 для G3
  __half* sV = sK;                      // ТО ЖЕ МЕСТО, что sK: нужен только после G2 (см. барьер)
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

  // --- tril(A) в разделяемую (в fp16, как операнд третьего умножения)
  ma.visit(lane, [&](float& x, int r, int c) { sA[(ro + r) * LDA + ca + c] = __float2half(x); });
  __syncthreads();                       // sK больше не нужен -> его место занимает sV

  for (int i = tid; i < BT * (DV / 8); i += nthr) {
    const int r = i / (DV / 8), c = (i % (DV / 8)) * 8;
    const bool ok = (t0 + r) < T;
    *reinterpret_cast<uint4*>(&sV[r * LDV + c]) =
        ok ? *reinterpret_cast<const uint4*>(&v[((long)(t0 + r) * H + ih) * DV + c]) : uint4{0, 0, 0, 0};
  }
  __syncthreads();

  // --- G3: += tril(A) x V. V лежит [t][v] с непрерывным v, то есть [K][N] -> kNMajor.
  WarpMma<SM / 16, SN / 16, BLayout::kNMajor> mv;
  mv.clear();
  mv.template accumulate<BT>(sA, LDA, wm * SM, sV, LDV, wn * SN, lane);
  // Слить два накопителя поэлементно нельзя через visit (разные объекты обходятся независимо),
  // поэтому складываем напрямую по одинаковой раскладке: у обоих MB, NB и лейн-отображение совпадают.
#pragma unroll
  for (int i = 0; i < SM / 16; ++i)
#pragma unroll
    for (int j = 0; j < SN / 16; ++j)
#pragma unroll
      for (int r = 0; r < 8; ++r) mo.acc[i][j][r] = (mo.acc[i][j][r] + mv.acc[i][j][r]) * scale;

  // --- запись
  mo.visit(lane, [&](float& x, int r, int c) {
    const int tt = ro + r;
    if (tt < rows) o[((long)(t0 + tt) * H + ih) * DV + co + c] = __float2half(x);
  });
}

}  // namespace gdn
}  // namespace fa2_sm70
