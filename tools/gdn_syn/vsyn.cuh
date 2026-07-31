// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// КАНДИДАТ СИНТЕЗА. Рамка -- ядро стратегии "КОНВЕЙЕР" (лучшее одиночное, 0.142 мс), в неё
// вставлены ключами приёмы двух других стратегий, чтобы каждый мерился A/B в ОДНОМ процессе:
//   ROWEXP -- строчно-столбцовое разложение e^{g_t-g_s} = e^{g_t} * e^{-g_s}   (стратегия ЗАНЯТОСТЬ)
//   WM x WN / WMA x WNA -- РАЗДЕЛЬНЫЕ сетки варпов для выхода и для A          (стратегия ГЕОМЕТРИЯ)
// Всё остальное -- дословно рамка: окно sK/половина sH/sV, расписание LDG->регистры->STS,
// засев накопителя G3 гейтованным G1, вынос выхода через разделяемую.

namespace fa2_sm70 {
namespace gdn {
namespace vsyn {

template <int BT, int DK, int DV>
__host__ __device__ constexpr size_t gdn_smem_bytes() {
  constexpr int PAD = 8;
  constexpr int win_k = BT * (DK + PAD), win_h = DV * (DK / 2 + PAD), win_v = BT * (DV + PAD);
  constexpr int win = win_k > win_h ? (win_k > win_v ? win_k : win_v) : (win_h > win_v ? win_h : win_v);
  return (size_t)(BT * (DK + PAD) + BT * (BT + PAD) + win) * sizeof(__half);
}

// DIAG -- ДИАГНОСТИКА ЦЕНЫ ЗАПИСИ, а не режим работы (ядро при DIAG!=0 даёт неверный выход):
//   1 -- запись ПРЕДИКАТНО отменена условием на РАНТАЙМНЫЙ аргумент (scale > 1e30f). Компилятор
//        свернуть его не может, поэтому G3, эпилог и подача V ОСТАЮТСЯ ЖИВЫМИ. Это и есть честная
//        абляция записи: обычное "закомментировать STG" убивает заодно всю цепочку зависимостей
//        (мёртвый код), и цена записи выходит завышенной.
//   2 -- пишется каждая ЧЕТВЁРТАЯ строка (трафик записи 16.1 -> 4.0 МБ) при полностью сохранной
//        сплошности сегмента: строка = 256 Б целиком, просто три четверти строк пропущены.
template <int BT, int DK, int DV, int WM = 2, int WN = 4, int ROWEXP = 0, int WMA = 2, int WNA = 4,
          int DIAG = 0>
__global__ __launch_bounds__(WM * WN * 32, 2) void volta_gdn_chunk_o(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ h, const float* __restrict__ g, __half* __restrict__ o,
    int T, int H, int Hg, float scale) {
  constexpr int PAD = 8;
  constexpr int LDQ = DK + PAD, LDV = DV + PAD, LDA = BT + PAD;
  constexpr int DKH = DK / 2, LDH = DKH + PAD;
  constexpr int SM = BT / WM, SN = DV / WN;
  constexpr int SMA = BT / WMA, SNA = BT / WNA;
  constexpr int MB = SM / 16, NB = SN / 16, MBA = SMA / 16, NBA = SNA / 16;
  constexpr int nthr = WM * WN * 32;
  static_assert(WM * WN == WMA * WNA, "обе сетки живут на одном числе варпов");
  static_assert(SM % 16 == 0 && SN % 16 == 0 && SMA % 16 == 0 && SNA % 16 == 0, "плитки кратны 16");
  static_assert(DK == DV, "буфер выхода переиспользует sQ");

  extern __shared__ __half smem[];
  __half* sQ = smem;
  __half* sA = sQ + BT * LDQ;
  __half* sU = sA + BT * LDA;                   // ОКНО: sK -> половина sH -> sV
  __half* sK = sU; __half* sHh = sU; __half* sV = sU;
  __shared__ float sG[BT];

  const int it = blockIdx.x, ih = blockIdx.y;
  const int NT = (T + BT - 1) / BT;
  if (it >= NT) return;
  const int t0 = it * BT, rows = min(BT, T - t0), rlast = rows - 1;
  const int ihg = ih / (H / Hg);
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int wma = warp / WNA, wna = warp % WNA;
  const int ro = wm * SM, co = wn * SN, roa = wma * SMA, ca = wna * SNA;
  const int rb = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);
  const int cb = (lane & 2) | ((lane & 4) << 1);

  constexpr int kQK = BT * (DK / 8) / nthr, kHH = DV * (DKH / 8) / nthr, kVR = BT * (DV / 8) / nthr;
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

  ldg_h(h1, 1);
  WarpMma<MBA, NBA, BLayout::kKMajor> ma;
  ma.clear();
  ma.template accumulate<DK>(sQ, LDQ, roa, sK, LDQ, ca, lane);

  if (ROWEXP) {
    // e^{g_t-g_s} = e^{g_t} * e^{-g_s}: карта аккумулятора даёт нити всего MBA*2 РАЗНЫХ строк и
    // NBA*4 РАЗНЫХ столбцов, поэтому MBA*NBA*8 вызовов SFU становятся MBA*2 + NBA*4.
    // СТОЛБЕЦ несёт ДВА бита полосы: (rg&1) -> +1 и ((rg>>2)&1) -> +4 (стратегия ЗАНЯТОСТЬ, п.5.3).
    float er[MBA][2], ec[NBA][4];
#pragma unroll
    for (int i = 0; i < MBA; ++i)
#pragma unroll
      for (int s = 0; s < 2; ++s) er[i][s] = __expf(sG[roa + i * 16 + rb + 2 * s]);
#pragma unroll
    for (int j = 0; j < NBA; ++j)
#pragma unroll
      for (int u = 0; u < 4; ++u) ec[j][u] = __expf(-sG[ca + j * 16 + cb + (u & 1) + 4 * (u >> 1)]);
#pragma unroll
    for (int i = 0; i < MBA; ++i)
#pragma unroll
      for (int j = 0; j < NBA; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; rg += 2) {
          const int sub = (rg >> 1) & 1, b = (rg >> 2) & 1;
          const int tt = roa + i * 16 + rb + 2 * sub;
          const int s0 = ca + j * 16 + cb + 4 * b;
          const float e0 = ma.acc[i][j][rg] * er[i][sub] * ec[j][2 * b];
          const float e1 = ma.acc[i][j][rg + 1] * er[i][sub] * ec[j][1 + 2 * b];
          const bool ok = tt < rows;
          *reinterpret_cast<__half2*>(&sA[(long)tt * LDA + s0]) = __floats2half2_rn(
              (ok && s0 <= tt && s0 < rows) ? e0 : 0.f,
              (ok && s0 + 1 <= tt && s0 + 1 < rows) ? e1 : 0.f);
        }
  } else {
    float gt[MBA][2];
#pragma unroll
    for (int i = 0; i < MBA; ++i)
#pragma unroll
      for (int s = 0; s < 2; ++s) gt[i][s] = sG[roa + i * 16 + rb + 2 * s];
#pragma unroll
    for (int i = 0; i < MBA; ++i)
#pragma unroll
      for (int j = 0; j < NBA; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; rg += 2) {
          const int sub = (rg >> 1) & 1;
          const int tt = roa + i * 16 + rb + 2 * sub;
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

  sts_h(h0);
  __syncthreads();
  WarpMma<MB, NB, BLayout::kKMajor> mo;
  mo.clear();
  mo.template accumulate<DKH>(sQ, LDQ, ro, sHh, LDH, co, lane);
  __syncthreads();
  sts_h(h1);
#pragma unroll
  for (int t = 0; t < kVR; ++t) {
    const int i = tid + t * nthr, r = min(i / (DV / 8), rlast), c = (i % (DV / 8)) * 8;
    vr[t] = *reinterpret_cast<const uint4*>(&v[((long)(t0 + r) * H + ih) * DV + c]);
  }
  __syncthreads();
  mo.template accumulate<DKH>(sQ + DKH, LDQ, ro, sHh, LDH, co, lane);
  __syncthreads();
#pragma unroll
  for (int t = 0; t < kVR; ++t) {
    const int i = tid + t * nthr, r = i / (DV / 8), c = (i % (DV / 8)) * 8;
    *reinterpret_cast<uint4*>(&sV[r * LDV + c]) = vr[t];
  }

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
    bool st = r < rows;
    if (DIAG == 1) st = st && (scale > 1e30f);      // рантаймно-ложно: цепочка жива, записи нет
    if (DIAG == 2) st = st && ((r & 3) == 0);       // четверть строк, сегмент по-прежнему целый
    if (st)
      *reinterpret_cast<uint4*>(&o[((long)(t0 + r) * H + ih) * DV + c]) =
          *reinterpret_cast<const uint4*>(&sO[r * LDQ + c]);
  }
}

}  // namespace vsyn
}  // namespace gdn
}  // namespace fa2_sm70
