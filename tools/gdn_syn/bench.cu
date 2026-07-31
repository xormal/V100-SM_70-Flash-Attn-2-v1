// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// СТЕНД СИНТЕЗА: все варианты ядра выхода gated-delta-net в ОДНОМ процессе, замер ЧЕРЕДУЮЩИЙСЯ.
//
// ЗАЧЕМ ИМЕННО ТАК. Карты 0/1 делят соседние прогоны, частоты не фиксируются (нет прав на -lgc) и
// гуляют 1290..1530 МГц. Абсолютные числа из РАЗНЫХ процессов сравнивать нельзя -- на этом уже
// потеряны часы у трёх стратегий из четырёх. Здесь все варианты живут в одном процессе, гоняются
// по кругу, и берётся МИНИМУМ по раундам: медленный участок задевает всех одинаково.
//
// Эталон -- fp64 по определению на CPU, ДОСЛОВНО тот же, что в tools/volta_gdn_o_test.cu
// (первый чанк каждой головы). Дополнительно сверяется ПОСЛЕДНИЙ чанк -- он может быть неполным,
// а хвост чанка правится в нескольких вариантах по-разному (зажим строки против предиката).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "../../fa2_src/fmha_kernel/gemm/volta_warp_mma.h"

#include "vorig.cuh"
#include "vs1.cuh"
#include "vs2.cuh"
#include "vs3.cuh"
#include "vs4.cuh"
#include "vsyn.cuh"
// ОТГРУЖАЕМЫЙ ФАЙЛ РЕПОЗИТОРИЯ -- ЦЕЛИКОМ, чтобы мерилось ровно то, что отгружено, а не копия.
#include "../../fa2_sm70/csrc/volta_gdn_o.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ОТКАЗ %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static float h2f(__half h) { return __half2float(h); }

static constexpr int BT = 64, DK = 128, DV = 128;

struct Ctx {
  const __half *q, *k, *v, *h; const float* g; __half* o;
  int T, H, Hg; float scale; dim3 grid;
};

struct Var {
  const char* name;
  size_t smem;
  int thr;
  bool diag;                 // диагностический вариант: выход НЕВЕРЕН, сверка пропускается
  std::function<void(const Ctx&)> launch;
  double best;   // мс
  double err;
};

template <class K>
static void prep(K kern, size_t smem) {
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
}

int main(int argc, char** argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 15680;
  const int ROUNDS = argc > 2 ? atoi(argv[2]) : 9;
  const int ITERS = argc > 3 ? atoi(argv[3]) : 20;
  const int H = 4, Hg = 4;
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

  Ctx cx{dq, dk, dv, dh, dg, doo, T, H, Hg, scale, dim3(NT, H)};

  // ---------------- РЕГИСТРАЦИЯ ВАРИАНТОВ ----------------
  std::vector<Var> vs;
  {  // 0: исходное ядро из репозитория (опора всех отношений)
    constexpr int PAD = 8, LDQ = DK + PAD, LDA = BT + PAD;
    const size_t sh = (size_t)(BT * LDQ + DV * LDQ + BT * LDQ + BT * LDA) * 2;
    auto kern = fa2_sm70::gdn::vorig::volta_gdn_chunk_o<BT, DK, DV, 2, 4>;
    prep(kern, sh);
    vs.push_back({"orig", sh, 256, false, [kern](const Ctx& c) {
      kern<<<c.grid, 256, (size_t)(BT * 136 + DV * 136 + BT * 136 + BT * 72) * 2>>>(
          c.q, c.k, c.v, c.h, c.g, c.o, c.T, c.H, c.Hg, c.scale); }, 1e9, 0});
  }
  {  // 1: ЗАНЯТОСТЬ (подача кусками K/2 + слитая запись + строчные экспоненты, UN=8)
    constexpr int OPT = fa2_sm70::gdn::vs1::kOptCoalesce | fa2_sm70::gdn::vs1::kOptChunkK |
                        fa2_sm70::gdn::vs1::kOptRowExp;
    const size_t sh = fa2_sm70::gdn::vs1::gdn_smem_bytes<BT, DK, DV, OPT>();
    auto kern = fa2_sm70::gdn::vs1::volta_gdn_chunk_o<BT, DK, DV, 2, 4, OPT, 8>;
    prep(kern, sh);
    vs.push_back({"s1_occ", sh, 256, false, [kern, sh](const Ctx& c) {
      kern<<<c.grid, 256, sh>>>(c.q, c.k, c.v, c.h, c.g, c.o, c.T, c.H, c.Hg, c.scale); }, 1e9, 0});
  }
  {  // 2: КОНВЕЙЕР (окно sK/половина sH/sV + расписание LDG-STS + слитая запись)
    const size_t sh = fa2_sm70::gdn::vs2::gdn_smem_bytes<BT, DK, DV>();
    auto kern = fa2_sm70::gdn::vs2::volta_gdn_chunk_o<BT, DK, DV, 2, 4>;
    prep(kern, sh);
    vs.push_back({"s2_pipe", sh, 256, false, [kern, sh](const Ctx& c) {
      kern<<<c.grid, 256, sh>>>(c.q, c.k, c.v, c.h, c.g, c.o, c.T, c.H, c.Hg, c.scale); }, 1e9, 0});
  }
  {  // 3: ГЕОМЕТРИЯ (1x8 по выходу, 2x4 по A, h прямо из глобальной)
    const size_t sh = fa2_sm70::gdn::vs3::gdn_o_smem_bytes<BT, DK, DV, 8, 8, false, true, false>();
    auto kern = fa2_sm70::gdn::vs3::volta_gdn_chunk_o<BT, DK, DV>;
    prep(kern, sh);
    vs.push_back({"s3_geom", sh, 256, false, [kern, sh](const Ctx& c) {
      kern<<<c.grid, 256, sh>>>(c.q, c.k, c.v, c.h, c.g, c.o, c.T, c.H, c.Hg, c.scale); }, 1e9, 0});
  }
  {  // 4: БЕЗ КРУГА (только вынос выхода через разделяемую, один блок)
    constexpr int PAD = 8, LDQ = DK + PAD, LDA = BT + PAD;
    const size_t sh = (size_t)(BT * LDQ + DV * LDQ + BT * LDQ + BT * LDA) * 2;
    auto kern = fa2_sm70::gdn::vs4::volta_gdn_chunk_o<BT, DK, DV, 2, 4>;
    prep(kern, sh);
    vs.push_back({"s4_wr", sh, 256, false, [kern, sh](const Ctx& c) {
      kern<<<c.grid, 256, sh>>>(c.q, c.k, c.v, c.h, c.g, c.o, c.T, c.H, c.Hg, c.scale); }, 1e9, 0});
  }
  {  // ОТГРУЖЕНО: файл репозитория как есть
    const size_t sh = fa2_sm70::gdn::gdn_smem_bytes<BT, DK, DV>();
    auto kern = fa2_sm70::gdn::volta_gdn_chunk_o<BT, DK, DV, 2, 4>;
    prep(kern, sh);
    vs.push_back({"SHIPPED", sh, 256, false, [kern, sh](const Ctx& c) {
      kern<<<c.grid, 256, sh>>>(c.q, c.k, c.v, c.h, c.g, c.o, c.T, c.H, c.Hg, c.scale); }, 1e9, 0});
  }
  // ---- кандидаты синтеза (vsyn.cuh) ----
#define SYN(tag, ROWEXP, WMv, WNv, WMAv, WNAv, DG)                                                 \
  {                                                                                                \
    const size_t sh = fa2_sm70::gdn::vsyn::gdn_smem_bytes<BT, DK, DV>();                           \
    auto kern =                                                                                    \
        fa2_sm70::gdn::vsyn::volta_gdn_chunk_o<BT, DK, DV, WMv, WNv, ROWEXP, WMAv, WNAv, DG>;      \
    prep(kern, sh);                                                                                \
    vs.push_back({tag, sh, WMv * WNv * 32, DG != 0, [kern, sh](const Ctx& c) {                      \
      kern<<<c.grid, WMv * WNv * 32, sh>>>(c.q, c.k, c.v, c.h, c.g, c.o, c.T, c.H, c.Hg, c.scale); \
    }, 1e9, 0});                                                                                   \
  }
  SYN("syn_base", 0, 2, 4, 2, 4, 0)     // = s2, переписанный параметрически (контроль переноса)
  SYN("syn_rowexp", 1, 2, 4, 2, 4, 0)   // + строчно-столбцовые экспоненты (приём стратегии 1)
  SYN("syn_18", 1, 1, 8, 2, 4, 0)       // + сетка выхода 1x8 (приём стратегии 3)
  SYN("syn_18a42", 1, 1, 8, 4, 2, 0)    // + сетка A 4x2
  SYN("syn_a42", 1, 2, 4, 4, 2, 0)      // сетка выхода 2x4, сетка A 4x2
  SYN("DIAG_nowr", 0, 2, 4, 2, 4, 1)    // абляция записи БЕЗ мёртвого кода (выход неверный)
  SYN("DIAG_wr14", 0, 2, 4, 2, 4, 2)    // четверть строк выхода (выход неверный)
#undef SYN

  // ---------------- ПРОВЕРКА ----------------
  // Эталон fp64 по определению. Чанки: первый (chunk 0) и ПОСЛЕДНИЙ (NT-1, может быть неполным).
  const int chunks[2] = {0, NT - 1};
  std::vector<__half> out((size_t)T * H * DV);
  for (auto& var : vs) {
    CK(cudaMemset(doo, 0, (size_t)T * H * DV * 2));
    var.launch(cx);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("%-12s ЗАПУСК ОТКАЗАЛ: %s\n", var.name, cudaGetErrorString(e)); exit(1); }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(), doo, out.size() * 2, cudaMemcpyDeviceToHost));
    if (var.diag) { var.err = -1.0; continue; }
    double emax = 0, rmax = 0;
    for (int ci = 0; ci < 2; ++ci) {
      const int it = chunks[ci];
      if (ci == 1 && chunks[1] == chunks[0]) break;
      const int t0 = it * BT, rows = (T - t0 < BT ? T - t0 : BT);
      for (int h = 0; h < H; ++h)
        for (int t = 0; t < rows; ++t)
          for (int vv = 0; vv < DV; ++vv) {
            double acc = 0;
            for (int kk = 0; kk < DK; ++kk)
              acc += (double)h2f(q[(size_t)(t0 + t) * Hg * DK + h * DK + kk]) *
                     (double)h2f(hh[((size_t)it * H + h) * DV * DK + (size_t)vv * DK + kk]);
            acc *= exp((double)g[(t0 + t) * H + h]);
            for (int s = 0; s <= t; ++s) {
              double a = 0;
              for (int kk = 0; kk < DK; ++kk)
                a += (double)h2f(q[(size_t)(t0 + t) * Hg * DK + h * DK + kk]) *
                     (double)h2f(k[(size_t)(t0 + s) * Hg * DK + h * DK + kk]);
              acc += a * exp((double)g[(t0 + t) * H + h] - (double)g[(t0 + s) * H + h]) *
                     (double)h2f(v[(size_t)(t0 + s) * H * DV + h * DV + vv]);
            }
            acc *= scale;
            const double got = h2f(out[(size_t)(t0 + t) * H * DV + h * DV + vv]);
            const double er = fabs(got - acc);
            if (er > emax) emax = er;
            if (fabs(acc) > rmax) rmax = fabs(acc);
          }
    }
    var.err = emax / (rmax + 1e-30);
  }

  // ---------------- ЗАМЕР ЧЕРЕДУЮЩИЙСЯ ----------------
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int w = 0; w < 3; ++w) for (auto& var : vs) { for (int i = 0; i < ITERS; ++i) var.launch(cx); }
  CK(cudaDeviceSynchronize());
  for (int r = 0; r < ROUNDS; ++r)
    for (auto& var : vs) {
      CK(cudaDeviceSynchronize()); CK(cudaEventRecord(a));
      for (int i = 0; i < ITERS; ++i) var.launch(cx);
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= (float)ITERS;
      if (ms < var.best) var.best = ms;
    }

  const double fl = (double)NT * H * (2.0 * BT * DK * DV + 2.0 * BT * BT * DK + 2.0 * BT * BT * DV);
  const double byt = (double)T * (Hg * DK + Hg * DK + H * DV + H * DV) * 2.0 +
                     (double)NT * H * DV * DK * 2.0;
  printf("T=%d  блоков=%d  трафик=%.2f МБ  работа=%.3f ГФЛОП  раундов=%d\n",
         T, NT * H, byt / 1e6, fl / 1e9, ROUNDS);
  printf("%-12s %8s %8s %8s %8s %8s %10s\n", "вариант", "мс", "TFLOP/s", "ГБ/с", "КБ", "к orig", "отн.ошибка");
  const double base = vs[0].best;
  for (auto& var : vs)
    printf("%-12s %8.4f %8.1f %8.0f %8.1f %8.3f %10.2e\n", var.name, var.best,
           fl / (var.best * 1e-3) / 1e12, byt / (var.best * 1e-3) / 1e9, var.smem / 1024.0,
           base / var.best, var.err);
  return 0;
}
