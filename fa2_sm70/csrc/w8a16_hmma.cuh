// W8A16 на ТЕНЗОРНЫХ ядрах Volta. GPTQ int8, вес потребляется упакованным.
//
// ПОЧЕМУ ЭТО ВООБЩЕ ВОЗМОЖНО. IMMA у sm_70 нет (ptxas отвергает m8n8k16.s8, nvdisasm не знает
// опкода). Но она не нужна: fp16 представляет целые до 2048 БЕЗ ошибки, а аккумулятор HMMA -- fp32,
// точный до 2^24. Значит байт веса, поданный как fp16, даёт ПОБИТОВО точное целочисленное
// произведение. Замерено против torch._int_mm: 96.5 TOPS против 52.2 у DP4A (1.85x).
//
// РАСПАКОВКА БЕЗ cvt. fp16 0x6400 == 1024.0, мантисса 10 бит, поэтому 0x6400 | q даёт РОВНО 1024+q
// для беззнакового q в 0..255 -- а GPTQ-байт как раз беззнаковый. Один OR, ни одной конверсии.
//
// ЗАМЕРЕНО, И ЭТО НЕ РЫЧАГ ДЛЯ МАЛОГО M. Ядро КОРРЕКТНО (relL2 2.9e-04 на всех M, совпадает с
// SIMT-путём) и обгоняет SIMT при M>=32 (1.07x при 32, 1.91x при 64), но плотному cublas проигрывает
// везде: 0.14..0.28x. Причина СТРУКТУРНАЯ, а не в тензорных ядрах:
//   плотный путь при M=16 читает 178 МБ за 240 мкс = 741 ГБ/с, то есть 88% достижимой полосы -- он
//   уже memory-bound. Наш упакованный читает 89 МБ, его пол 106 мкс (потенциал x2.3), а мы на 976 --
//   в 9 раз от собственного пола. При M=16 плитка 16x64 даёт 10.7 MAC на байт при точке равновесия
//   55: задача memory-bound, а форма ядра compute-овая (5 барьеров на группу, 136 групп, 4 варпа).
// ВЫВОД: при малом M рычаг -- ПОЛОСА, а не тензорные ядра. HMMA окупается на compute-bound int8
// (большие M, GEMM-масштаб), где и замерены 96.5 TOPS против 52.2 у DP4A. Чтобы взять x2.3, нужна
// не эта форма, а ядро, оптимальное по полосе: широкая плитка по N, минимум барьеров, векторная
// загрузка веса и распаковка в тени загрузки.
//
// СМЕЩЕНИЕ 1024 СВОРАЧИВАЕТСЯ БЕСПЛАТНО. Внутри группы (G ключей, общие s и z):
//     sum_k x_k*w_k = s*(sum_k x_k*q_k - z*sum_k x_k) = s*(H - (1024+z)*Sx),
// где H -- то, что вернул HMMA на закодированных байтах, а Sx = sum x по группе У НАС УЖЕ
// СЧИТАЕТСЯ (она нужна для свёртки нулевой точки). То есть поправка не стоит ничего.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "gemm/volta_warp_mma.h"

namespace fa2_sm70 {
namespace w8a16 {

using fa2_sm70::WarpMma;
using fa2_sm70::BLayout;

// BN -- столбцов выхода на блок; WN -- варпов по N; G -- размер группы GPTQ (обычно 128).
template <int BN, int WN, int G>
__global__ __launch_bounds__(WN * 32) void w8a16_hmma(
    const __half* __restrict__ x, const int* __restrict__ qweight, const int* __restrict__ qzeros,
    const __half* __restrict__ scales, __half* __restrict__ y, int M, int N, int K) {
  constexpr int MT = 16;                       // одна плитка HMMA по M (m8n8k4 -> кратно 16)
  constexpr int NBw = (BN / WN) / 16;          // блоков 16 столбцов на варп
  // [РАЗНЫЕ ДОПОЛНЕНИЯ ДЛЯ sA И sB -- ОДНО НА ОБА БЫЛО КОМПРОМИССОМ, А НЕ ВЫБОРОМ]
  // Конфликтность = |R| / |{r*S mod 32}| (docs/SM70_KERNEL_PLAYBOOK.md, «диагональ по строкам»):
  //   sA: LDA = G+8 = 136 полуслов = 68 слов, 68 mod 32 = 4  -> 2.00.  PAD=4 (LDA=132 = 66 слов) -> 1.00.
  //       Это ровно закон «DK+4», уже выведенный и применённый в volta_gdn_{h,wu,kkt}.
  //   sB: LDB = BN+8 = 72 полуслова = 36 слов -> 1.60.  PAD=16 (LDB=80 = 40 слов) -> 1.00
  //       (закон n-мажорного: (LD/2) mod 32 == 8).
  // Законность: LDS.64 требует лишь LD кратно 4 полусловам -- 132/4=33, 80/4=20, оба целые.
  // Обе точки лежат в зоне, где обе модели конфликтности (по всем 32 полосам и внутри 16-полосной
  // фазы) СОГЛАСНЫ, поэтому неопределённость фазовой модели здесь не работает.
  // Разделяемая: sA 4352 -> 4224 Б (МЕНЬШЕ), sB 18432 -> 20480 Б; итого +1920 Б.
  // Ответ не меняется ни на бит: это только размещение.
  constexpr int PADA = 4, PADB = 16;
  constexpr int LDA = G + PADA, LDB = BN + PADB;
  static_assert(BN % (WN * 16) == 0, "BN кратно WN*16");
  static_assert(G % 4 == 0, "группа кратна 4 (шаг k у HMMA)");

  __shared__ __half sA[MT * LDA];               // x: [MT][G]   k-мажорный
  __shared__ __half sB[G * LDB];                // вес: [G][BN] n-мажорный (распакованный, 1024+q)
  __shared__ float sSx[MT];                     // sum x по группе, на строку
  __shared__ float sS[BN], sZ[BN];               // масштаб и нуль группы -- ОДИН раз, не на элемент

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n0 = blockIdx.x * BN;
  const int m0 = blockIdx.y * MT;
  const int mrows = min(MT, M - m0);

  WarpMma<1, NBw, BLayout::kNMajor> o; o.clear();   // O += x * Wenc, Wenc это [k][n] -> n-мажорный
  float acc[NBw][8];
#pragma unroll
  for (int j = 0; j < NBw; ++j)
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[j][r] = 0.f;

  for (int g0 = 0; g0 < K; g0 += G) {
    const int g = g0 / G;
    __syncthreads();
    // --- x группы в smem + сумма по строке ------------------------------------------------------
    for (int i = tid; i < MT * G; i += WN * 32) {
      const int r = i / G, c = i - r * G;
      const bool live = (r < mrows) && (g0 + c < K);
      sA[r * LDA + c] = live ? x[(long)(m0 + r) * K + g0 + c] : __ushort_as_half(0);
    }
    __syncthreads();   // ГОНКА: sSx суммирует строку sA, которую пишут ДРУГИЕ нити. Без барьера
                       // нить 0 читает то, что нить 100 ещё не записала. Проявлялось как верный
                       // ответ при M<=8 и relL2 3.2e-02 при M=16 -- то есть маскировалось раскладкой.
    for (int r = tid; r < MT; r += WN * 32) {
      float sx = 0.f;
      if (r < mrows) for (int c = 0; c < G; ++c) sx += __half2float(sA[r * LDA + c]);
      sSx[r] = sx;
    }
    // --- вес: ОДИН OR на байт, ни одной конверсии ------------------------------------------------
    for (int i = tid; i < (G / 4) * BN; i += WN * 32) {
      const int kq = i / BN, n = i - kq * BN;             // kq -- индекс int32 внутри группы
      const int nn = n0 + n;
      unsigned w = (nn < N) ? (unsigned)qweight[(long)((g0 >> 2) + kq) * N + nn] : 0u;
      __half* p = &sB[(kq * 4) * LDB + n];
#pragma unroll
      for (int b = 0; b < 4; ++b)
        p[b * LDB] = __ushort_as_half(0x6400u | ((w >> (b * 8)) & 0xFFu));   // = 1024 + q, ТОЧНО
    }
    // масштаб и нуль группы: по одному чтению на СТОЛБЕЦ, а не на элемент аккумулятора. В первой
    // версии они читались внутри эпилога, то есть K/G * NBw * 8 раз на нить -- это и был лимит.
    for (int c = tid; c < BN; c += WN * 32) {
      const int nn = n0 + c;
      sS[c] = (nn < N) ? __half2float(scales[(long)g * N + nn]) : 0.f;
      sZ[c] = (nn < N) ? (float)((qzeros[(long)g * (N >> 2) + (nn >> 2)] >> ((nn & 3) * 8)) & 0xFF) : 0.f;
    }
    __syncthreads();
    o.clear();
    o.template accumulate<G>(sA, LDA, 0, sB, LDB, warp * (BN / WN), lane);
    // --- эпилог группы: s*(H - (1024+z)*Sx) ------------------------------------------------------
#pragma unroll
    for (int j = 0; j < NBw; ++j)
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int row = ((lane & 1) | ((lane & 16) >> 2) | (lane & 8)) + 2 * ((r >> 1) & 1);
        const int col = ((lane & 2) | ((lane & 4) << 1)) + (r & 1) + 4 * ((r >> 2) & 1) + j * 16
                        + warp * (BN / WN);
        const int nn = n0 + col;
        if (nn < N && row < mrows)
          acc[j][r] += sS[col] * (o.acc[0][j][r] - (1024.f + sZ[col]) * sSx[row]);
      }
  }
  // --- запись -----------------------------------------------------------------------------------
#pragma unroll
  for (int j = 0; j < NBw; ++j)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int row = ((lane & 1) | ((lane & 16) >> 2) | (lane & 8)) + 2 * ((r >> 1) & 1);
      const int col = ((lane & 2) | ((lane & 4) << 1)) + (r & 1) + 4 * ((r >> 2) & 1) + j * 16
                      + warp * (BN / WN);
      const int nn = n0 + col;
      if (nn < N && row < mrows) y[(long)(m0 + row) * N + nn] = __float2half(acc[j][r]);
    }
}

}  // namespace w8a16
}  // namespace fa2_sm70
