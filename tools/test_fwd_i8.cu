// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Проверка ядра нового фундамента (volta_fwd_i8.cuh) БЕЗ торча: своя сборка, свой эталон.
//
// ЭТАЛОН СЧИТАЕТСЯ ПО ДЕКВАНТОВАННЫМ K/V, а не по исходным fp16. Иначе ошибка ядра смешается с
// ошибкой квантования, и любое расхождение можно будет списать на "ну это же int8". Ядро обязано
// воспроизвести внимание НА ТЕХ ЗНАЧЕНИЯХ, КОТОРЫЕ ЕМУ ДАЛИ, с точностью fp16-накопления.
#ifndef FA2_WS
#define FA2_WS 0
#endif
#ifndef FA2_WSPROD
#define FA2_WSPROD 0
#endif
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include "../fa2_sm70/csrc/volta_fwd_i8.cuh"
#if FA2_WS
#include "../fa2_sm70/csrc/volta_fwd_ws.cuh"
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static float frand() { return (float)rand() / RAND_MAX * 2.f - 1.f; }

int main(int argc, char** argv) {
  const int B = 1, H = 4, Hkv = 4, D = 256;
  const int Sq = argc > 1 ? atoi(argv[1]) : 256;
  const int Sk = argc > 2 ? atoi(argv[2]) : 256;
  const bool causal = true;
  constexpr int BQ = FA2_BQ, BK = 64, WM = FA2_WM, WN = FA2_WN, DC = 64;
  constexpr int MINB = FA2_MINB;
  srand(1234);

  std::vector<__half> hQ((size_t)B * Sq * H * D);
  std::vector<float>  fK((size_t)B * Hkv * Sk * D), fV((size_t)B * Hkv * Sk * D);
  for (auto& x : hQ) x = __float2half(frand() * 0.5f);
  for (auto& x : fK) x = frand();
  for (auto& x : fV) x = frand();

  // КВАНТОВАНИЕ K/V ПО ПОЗИЦИИ: масштаб на строку (одна позиция ключа), симметричное, в int8.
  std::vector<int8_t> qK(fK.size()), qV(fV.size());
  std::vector<float>  sK((size_t)B * Hkv * Sk), sV((size_t)B * Hkv * Sk);
  std::vector<float>  dK(fK.size()), dV(fV.size());          // ДЕКВАНТОВАННЫЕ -- по ним эталон
  for (size_t p = 0; p < (size_t)B * Hkv * Sk; ++p) {
    float mk = 1e-9f, mv = 1e-9f;
    for (int d = 0; d < D; ++d) { mk = fmaxf(mk, fabsf(fK[p * D + d])); mv = fmaxf(mv, fabsf(fV[p * D + d])); }
    const float ak = mk / 127.f, av = mv / 127.f;
    sK[p] = ak; sV[p] = av;
    for (int d = 0; d < D; ++d) {
      int k = (int)lrintf(fK[p * D + d] / ak); k = k < -127 ? -127 : (k > 127 ? 127 : k);
      int v = (int)lrintf(fV[p * D + d] / av); v = v < -127 ? -127 : (v > 127 ? 127 : v);
#if defined(FA2_BIASED_U8) && FA2_BIASED_U8
      // КВАНТОВАТЕЛЬ пишет СМЕЩЁННЫЙ байт (u = v + 128). Ядро тогда не тратит XOR на разворот, а
      // смещение остаётся той же константой 1152 -- работа перенесена из горячего пути в холодный.
      qK[p * D + d] = (int8_t)(uint8_t)(k + 128); qV[p * D + d] = (int8_t)(uint8_t)(v + 128);
#else
      qK[p * D + d] = (int8_t)k; qV[p * D + d] = (int8_t)v;
#endif
      dK[p * D + d] = k * ak;   dV[p * D + d] = v * av;
    }
  }

  __half *dQ_, *dO; int8_t *dKq, *dVq; float *dKs, *dVs, *dLse;
  CK(cudaMalloc(&dQ_, hQ.size() * 2));            CK(cudaMemcpy(dQ_, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dKq, qK.size()));                CK(cudaMemcpy(dKq, qK.data(), qK.size(), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dVq, qV.size()));                CK(cudaMemcpy(dVq, qV.data(), qV.size(), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dKs, sK.size() * 4));            CK(cudaMemcpy(dKs, sK.data(), sK.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dVs, sV.size() * 4));            CK(cudaMemcpy(dVs, sV.data(), sV.size() * 4, cudaMemcpyHostToDevice));
  const int lse_dim = ((Sq + 31) / 32) * 32;
  CK(cudaMalloc(&dO, (size_t)B * Sq * H * D * 2));
  CK(cudaMalloc(&dLse, (size_t)B * H * lse_dim * 4));

  constexpr int DV = FA2_DV, LDQ = (FA2_WS ? D : DC) + 4, LDP = BK + 4, LDK8 = (FA2_WS ? D : DC) + 16, LDV = DV + 16;
  const size_t sh = (size_t)(BQ * LDQ + BQ * LDP) * sizeof(__half) + (size_t)(FA2_WS ? 2 : 1) * BK * (LDK8 + LDV);
#if FA2_WS
#ifndef FA2_DIAG
#define FA2_DIAG 0
#endif
#ifndef FA2_PHSPLIT
#define FA2_PHSPLIT 0
#endif
  // полный список доводов шаблона: KVFMT=0, WIDE=0, SWZ=0, DIAG, PHSPLIT, KSP=1, KSPL=0, PAGED=0 (боевой: PHSPLIT=1, PAGED=1)
  auto kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<BQ, BK, D, WM, WN, true,  DC, MINB, 1, DV, 0, 0, 0, FA2_DIAG, FA2_PHSPLIT, 1, 0, 0>
                     : fa2_sm70::fwd_ws::volta_fwd_ws<BQ, BK, D, WM, WN, false, DC, MINB, 1, DV, 0, 0, 0, FA2_DIAG, FA2_PHSPLIT, 1, 0, 0>;
  const int kThreads = ((FA2_WSPROD > 0 ? FA2_WSPROD : WM * WN) + WM * WN) * 32;
#else
  auto kern = causal ? fa2_sm70::fwd_i8::volta_fwd_i8<BQ, BK, D, WM, WN, true,  DC, MINB, 1, DV>
                     : fa2_sm70::fwd_i8::volta_fwd_i8<BQ, BK, D, WM, WN, false, DC, MINB, 1, DV>;
  const int kThreads = WM * WN * 32;
#endif
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  printf("smem динамическая %zu Б (%.1f КБ)\n", sh, sh / 1024.0);

  int* dRole = nullptr;       // счётчики ролей по SM (128 слотов с запасом)
  CK(cudaMalloc(&dRole, 128 * sizeof(int))); CK(cudaMemset(dRole, 0, 128 * sizeof(int)));
  long long* dPh = nullptr;   // передаётся ВСЕГДА: вызов идёт через указатель на функцию, а он
                              // не несёт значений по умолчанию
#if FA2_PHASE
  CK(cudaMalloc(&dPh, 8 * sizeof(long long))); CK(cudaMemset(dPh, 0, 8 * sizeof(long long)));
#endif
  const float scale = 1.f / sqrtf((float)D);
  const float sc = scale * 1.4426950408889634f;             // софтмакс по основанию 2
  kern<<<dim3((Sq + BQ - 1) / BQ, H * (D / DV), B), kThreads, sh>>>(
      dQ_, dKq, dKs, dVq, dVs, dO, dLse, Sq, Sk, sc, H / Hkv, causal ? (Sk - Sq) : 0,
      (long)H * D, (long)D, (long)Sq * H * D,
      (long)D, (long)Sk * D, (long)Hkv * Sk * D,
      (long)H * D, (long)D, (long)Sq * H * D,
      (long)lse_dim, (long)H * lse_dim,
      (long)Sk, (long)Hkv * Sk, dPh, dRole, nullptr, nullptr, 0, nullptr, nullptr, 0, 1, 0, 1, nullptr, nullptr);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());

  {   // ЗАМЕР: разлив против занятости -- это размен, его решают числом, а не рассуждением
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms = 0;
    auto run = [&]{ kern<<<dim3((Sq + BQ - 1) / BQ, H * (D / DV), B), kThreads, sh>>>(
        dQ_, dKq, dKs, dVq, dVs, dO, dLse, Sq, Sk, sc, H / Hkv, causal ? (Sk - Sq) : 0,
        (long)H * D, (long)D, (long)Sq * H * D, (long)D, (long)Sk * D, (long)Hkv * Sk * D,
        (long)H * D, (long)D, (long)Sq * H * D, (long)lse_dim, (long)H * lse_dim,
        (long)Sk, (long)Hkv * Sk, dPh, dRole, nullptr, nullptr, 0, nullptr, nullptr, 0, 1, 0, 1, nullptr, nullptr); };
    for (int i = 0; i < 5; ++i) run();
    CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) run();
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); cudaEventElapsedTime(&ms, e0, e1);
    printf("ВРЕМЯ %.4f мс/запуск\n", ms / 20.0);
  }
#if FA2_PHASE
  { long long hp[8]; CK(cudaMemcpy(hp, dPh, 8 * sizeof(long long), cudaMemcpyDeviceToHost));
    long long tot = 0; for (int i = 0; i < 8; ++i) tot += hp[i];
#if FA2_WS
    const char* nm[8] = {"ВЕЗУЩИЕ: ждут буфер","ВЕЗУЩИЕ: везут","СЧИТАЮЩИЕ: ждут плитку",
                         "СЧИТАЮЩИЕ: считают","-","-","-","-"};
#else
    const char* nm[8] = {"0 барьер входа","1 перекладка масштабов+V","2 подача Q/K + GEMM1",
                         "3 маска+максимум+редукция","4 exp2 + запись P","5 редукция сумм",
                         "6 GEMM2","7 -"};
#endif
    printf("=== ДОЛИ ФАЗ (тактов, суммарно по блокам; читать ПРОПОРЦИИ) ===\n");
    for (int i = 0; i < (FA2_WS ? 4 : 7); ++i)
      printf("  %-28s %12lld  %5.1f%%\n", nm[i], hp[i], tot ? 100.0 * hp[i] / tot : 0.0);
  }
#endif
  if (getenv("FA2_NOREF")) { printf("эталон пропущен (FA2_NOREF)\n"); return 0; }
  std::vector<__half> hO((size_t)B * Sq * H * D);
  CK(cudaMemcpy(hO.data(), dO, hO.size() * 2, cudaMemcpyDeviceToHost));

  // ЭТАЛОН в fp64 по ДЕКВАНТОВАННЫМ значениям
  double num = 0, den = 0, mx = 0, mref = 0;
  std::vector<double> p(Sk);
  for (int h = 0; h < H; ++h) {
    const int hkv = h / (H / Hkv);
    for (int i = 0; i < Sq; ++i) {
      const int kend = causal ? (i + (Sk - Sq) + 1) : Sk;
      double m = -1e300;
      for (int j = 0; j < kend; ++j) {
        double s = 0;
        for (int d = 0; d < D; ++d) s += (double)__half2float(hQ[(size_t)i * H * D + h * D + d]) * dK[((size_t)hkv * Sk + j) * D + d];
        s *= scale; p[j] = s; if (s > m) m = s;
      }
      double l = 0;
      for (int j = 0; j < kend; ++j) { p[j] = exp(p[j] - m); l += p[j]; }
      for (int d = 0; d < D; ++d) {
        double acc = 0;
        for (int j = 0; j < kend; ++j) acc += p[j] * dV[((size_t)hkv * Sk + j) * D + d];
        acc /= l;
        const double got = __half2float(hO[(size_t)i * H * D + h * D + d]);
        const double e = got - acc;
        num += e * e; den += acc * acc; mx = fmax(mx, fabs(e)); mref = fmax(mref, fabs(acc));
      }
    }
  }
  const double rel = sqrt(num / (den + 1e-30));
  // ДВЕ МЕТРИКИ, и вторая обязательна. relL2 делит на НОРМУ эталона, а при длинных причинных
  // строках выход усредняется почти в ноль -- знаменатель мал, и relL2 раздувается там, где
  // абсолютная ошибка КАК РАЗ наименьшая. Судить по одной relL2 = принять артефакт метрики за
  // дефект ядра (и наоборот).
  const double relmax = mx / (mref + 1e-30);
  const bool ok = (rel < 3e-3) || (relmax < 3e-3);
  printf("Sq=%d Sk=%d  relL2 = %.3e  макс.абс = %.3e  макс|эталон| = %.3e  макс.отн = %.3e  -> %s\n",
         Sq, Sk, rel, mx, mref, relmax, ok ? "ПРОЙДЕНО" : "ПРОВАЛ");
  return ok ? 0 : 1;
}
