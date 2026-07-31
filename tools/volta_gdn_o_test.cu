// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Проверка и замер ядра выхода gated-delta-net (volta_gdn_o.cuh) против ПРЯМОГО эталона на CPU.
// Эталон считается по определению, а не другим ядром: иначе общая ошибка в понимании формулы
// пройдёт обе стороны сверки незамеченной.
//
// Сборка: nvcc -arch=sm_70 -O3 -I fa2_src/fmha_kernel/gemm -o gdno tools/volta_gdn_o_test.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>
#include "../fa2_sm70/csrc/volta_gdn_o.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ОТКАЗ %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static float h2f(__half h) { return __half2float(h); }

int main(int argc, char** argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 512;
  constexpr int BT = 64, DK = 128, DV = 128, WM = 2, WN = 4;
  const int H = 4, Hg = 4;                       // для проверки берём H == Hg
  const int NT = (T + BT - 1) / BT;
  const float scale = 1.0f / sqrtf((float)DK);

  std::vector<__half> q(T * Hg * DK), k(T * Hg * DK), v(T * H * DV), hh((size_t)NT * H * DV * DK);
  std::vector<float> g(T * H);
  srand(7);
  auto rnd = [] { return (float)rand() / RAND_MAX * 0.4f - 0.2f; };
  for (auto& x : q) x = __float2half(rnd());
  for (auto& x : k) x = __float2half(rnd());
  for (auto& x : v) x = __float2half(rnd());
  for (auto& x : hh) x = __float2half(rnd());
  // g -- НАКОПЛЕННЫЙ логарифм гейта внутри чанка: неубывающе отрицательный, как после cumsum
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) g[t * H + h] = -0.01f * (float)(t % BT) - 0.001f * h;

  __half *dq, *dk, *dv, *dh, *doo; float* dg;
  CK(cudaMalloc(&dq, q.size() * 2));  CK(cudaMalloc(&dk, k.size() * 2));
  CK(cudaMalloc(&dv, v.size() * 2));  CK(cudaMalloc(&dh, hh.size() * 2));
  CK(cudaMalloc(&doo, (size_t)T * H * DV * 2)); CK(cudaMalloc(&dg, g.size() * 4));
  CK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dh, hh.data(), hh.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));

  // Размер разделяемой брать ТОЛЬКО из gdn_smem_bytes: в ядре три буфера объединены в одно окно,
  // и наивная сумма даёт 77 КБ вместо 44 -- то есть молча теряет ВТОРОЙ БЛОК на SM (и ~35% скорости).
  const size_t sh = fa2_sm70::gdn::gdn_smem_bytes<BT, DK, DV>();
  auto kern = fa2_sm70::gdn::volta_gdn_chunk_o<BT, DK, DV, WM, WN>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  dim3 grid(NT, H);
  kern<<<grid, WM * WN * 32, sh>>>(dq, dk, dv, dh, dg, doo, T, H, Hg, scale);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<__half> out((size_t)T * H * DV);
  CK(cudaMemcpy(out.data(), doo, out.size() * 2, cudaMemcpyDeviceToHost));

  // --- ЭТАЛОН ПО ОПРЕДЕЛЕНИЮ (fp64), только для первого чанка каждой головы: этого достаточно,
  // чтобы поймать и раскладку операндов, и маску, и гейты.
  double emax = 0, rmax = 0;
  for (int h = 0; h < H; ++h) {
    const int rows = (T < BT ? T : BT);
    for (int t = 0; t < rows; ++t)
      for (int vv = 0; vv < DV; ++vv) {
        double acc = 0;
        for (int kk = 0; kk < DK; ++kk)                       // e^{g_t} * q.h^T
          acc += (double)h2f(q[(size_t)t * Hg * DK + h * DK + kk]) *
                 (double)h2f(hh[((size_t)0 * H + h) * DV * DK + (size_t)vv * DK + kk]);
        acc *= exp((double)g[t * H + h]);
        for (int s = 0; s <= t; ++s) {                        // причинная внутричанковая часть
          double a = 0;
          for (int kk = 0; kk < DK; ++kk)
            a += (double)h2f(q[(size_t)t * Hg * DK + h * DK + kk]) *
                 (double)h2f(k[(size_t)s * Hg * DK + h * DK + kk]);
          acc += a * exp((double)g[t * H + h] - (double)g[s * H + h]) *
                 (double)h2f(v[(size_t)s * H * DV + h * DV + vv]);
        }
        acc *= scale;
        const double got = h2f(out[(size_t)t * H * DV + h * DV + vv]);
        const double e = fabs(got - acc);
        if (e > emax) emax = e;
        if (fabs(acc) > rmax) rmax = fabs(acc);
      }
  }
  printf("T=%d  относительная ошибка %.2e  (макс.|эталон| %.4f)\n", T, emax / (rmax + 1e-30), rmax);

  // --- замер. ДВЕ ЛОВУШКИ, каждая уже стоившая часа и дающая ошибку В ДВА РАЗА:
  //  1. карта без режима постоянства между процессами уходит в низкое состояние по частоте, и
  //     прогрев "по числу итераций" её не поднимает -- греем ПО ЧАСАМ, не меньше 300 мс;
  //  2. карты делят соседние прогоны, поэтому среднее негодно -- берём МИНИМУМ по сериям.
  // Для сравнения ВАРИАНТОВ между собой этого всё равно мало: нужен один процесс с чередованием,
  // см. tools/gdn_syn/bench.cu (там же исходное ядро как контроль годности выборки).
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  auto launch = [&] { kern<<<grid, WM * WN * 32, sh>>>(dq, dk, dv, dh, dg, doo, T, H, Hg, scale); };
  for (clock_t t0c = clock(); (double)(clock() - t0c) / CLOCKS_PER_SEC < 0.3;) {
    for (int i = 0; i < 20; ++i) launch();
    CK(cudaDeviceSynchronize());
  }
  float ms = 1e30f;
  for (int s = 0; s < 10; ++s) {
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(a));
    for (int i = 0; i < 20; ++i) launch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float t; CK(cudaEventElapsedTime(&t, a, b)); t /= 20.f;
    if (t < ms) ms = t;
  }
  const double fl = (double)NT * H * (2.0 * BT * DK * DV + 2.0 * BT * BT * DK + 2.0 * BT * BT * DV);
  printf("время %.3f мс   %.1f TFLOP/s   (разделяемой %.1f КБ)\n", ms, fl / (ms * 1e-3) / 1e12, sh / 1024.0);
  return 0;
}
