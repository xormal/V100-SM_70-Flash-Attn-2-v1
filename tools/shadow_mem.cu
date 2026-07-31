// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// СТЕНД ГЛУБИНЫ ТЕНИ, хозяин №1: ПАМЯТНЫЙ (W8A16 GEMV, M=1 -- архетип gptq-умножений декода,
// 51% токена, простаивает СЧЁТ ~98%).
//
// ВОПРОС, НА КОТОРЫЙ ОТВЕЧАЕТ СТЕНД: сколько ВЫДАЧ (не FLOP!) можно вставить в тело памятного
// хозяина, прежде чем его время тронется. До этой точки наездник бесплатен, после -- платит линейно.
//
// ПОЧЕМУ В ВЫДАЧАХ, А НЕ В ДОЛЯХ ПИКА (EPILOG §1a.0): на Volta выдача одна -- 4 планировщика по
// 1 инструкции за такт. "Счёт простаивает на 98%" означает простой ТЕНЗОРНОГО/ALU блока, но влезет
// ли туда работа -- решает ПОРТ ВЫДАЧИ. Поэтому наездник подаётся ПОРЦИЯМИ ИНСТРУКЦИЙ, и колено
// ищется по числу инструкций на загруженный байт.
//
// ЧЕМ ОГРАНИЧЕНО КОЛЕНО -- без ncu (RmProfilingAdminOnly=1, root недоступен) различаем ПО ВИДУ
// НАЕЗДНИКА, это и есть фальсификатор:
//   выдача (общий порт)   -> все виды инструкций упираются на ОДНОМ числе инструкций;
//   конкретный конвейер   -> колено пропорционально его пропускной (FMA 1/2 выдачи, XU 1/4);
//   зависимость по данным -> колено сдвигается при росте ILP (1 цепочка против 4) при том же числе;
//   путь разделяемой (LSU)-> LDS упирается много раньше FFMA;
//   ёмкость (регистры)    -> колено совпадает со скачком регистров (ptxas -v) или разливом.
//
// Сборка:
//   nvcc -arch=sm_70 -O3 -std=c++17 --use_fast_math -Xptxas -v -o shadow_mem tools/shadow_mem.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ОТКАЗ %s: %s (%d)\n", #x, cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// Боевая форма: down_proj Huihui-Qwen3.6-27B-INT8, K=17408, N=5120, группа 128, M=1 (декод).
#define K_DIM   17408
#define N_DIM   5120
#define G_DIM   128
#define THREADS 256
#define XT      1024          // k-плитка активации в разделяемой (4 КБ при MT=1)

// Виды наездника. Каждый -- ОДНА инструкция на порцию (проверяется по SASS).
#define KIND_NONE 0
#define KIND_FFMA 1           // конвейер FMA (fp32), 1/2 выдачи на планировщик
#define KIND_PRMT 2           // конвейер ALU (перестановка байт), полная выдача
#define KIND_MUFU 3           // конвейер XU (ex2), 1/4 выдачи
#define KIND_LDS  4           // путь разделяемой (LSU)
#define KIND_LDG  5           // КОНТРОЛЬ: байтовый наездник в ПАМЯТНОГО хозяина -- обязан платить сразу

// R -- инструкций наездника на одно загруженное слово веса (4 байта на поток).
// ILP -- независимых цепочек (различает зависимость от выдачи).
template <int R, int KIND, int ILP>
__global__ __launch_bounds__(THREADS) void w8a16_shadow(
    const __half* __restrict__ x, const int* __restrict__ qweight,
    const int* __restrict__ qzeros, const __half* __restrict__ scales,
    float* __restrict__ ypart, const unsigned* __restrict__ scratch,
    int M, int N, int K, int G, int groups_per_split, float eps, unsigned smask) {
  __shared__ float sx[XT];
  const int n = blockIdx.x * THREADS + threadIdx.x;
  const bool live = (n < N);
  const int nw = n >> 2, nsh = (n & 3) * 8;

  const int kbeg = blockIdx.y * groups_per_split * G;
  const int kend = min(K, kbeg + groups_per_split * G);
  if (kbeg >= kend) return;

  float acc = 0.f;
  // Состояние наездника: ILP независимых цепочек.
  float  rf[ILP];  unsigned ru[ILP];
  #pragma unroll
  for (int i = 0; i < ILP; ++i) { rf[i] = (float)(threadIdx.x + i + 1); ru[i] = threadIdx.x + i; }

  for (int k0 = kbeg; k0 < kend; k0 += XT) {
    const int len = min(XT, kend - k0);
    __syncthreads();
    for (int t = threadIdx.x; t < len; t += THREADS) sx[t] = __half2float(x[k0 + t]);
    __syncthreads();
    if (!live) continue;
    for (int gk = 0; gk < len; gk += G) {
      const int g = (k0 + gk) / G;
      const float s = __half2float(scales[(long)g * N + n]);
      const float z = (float)((qzeros[(long)g * (N >> 2) + nw] >> nsh) & 0xFF);
      const int glen = min(G, len - gk);
      float qa = 0.f, xa = 0.f;
      #pragma unroll 4
      for (int k = 0; k < glen; k += 4) {
        const unsigned w = (unsigned)qweight[(long)((k0 + gk + k) >> 2) * N + n];
        const float q0 = (float)( w        & 0xFF), q1 = (float)((w >>  8) & 0xFF);
        const float q2 = (float)((w >> 16) & 0xFF), q3 = (float)((w >> 24) & 0xFF);
        const float4 a = *reinterpret_cast<const float4*>(&sx[gk + k]);
        qa = fmaf(a.x, q0, qa); qa = fmaf(a.y, q1, qa);
        qa = fmaf(a.z, q2, qa); qa = fmaf(a.w, q3, qa);
        xa += a.x + a.y + a.z + a.w;
        // ---------------- НАЕЗДНИК: R инструкций на 4 байта веса ----------------
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          if constexpr (KIND == KIND_FFMA) rf[r % ILP] = fmaf(rf[r % ILP], 1.0000001f, a.x);
          if constexpr (KIND == KIND_PRMT) ru[r % ILP] = __byte_perm(ru[r % ILP], w, 0x4321);
          if constexpr (KIND == KIND_MUFU) rf[r % ILP] = exp2f(rf[r % ILP]);
          if constexpr (KIND == KIND_LDS)  rf[r % ILP] += sx[(threadIdx.x * 17 + r * 5 + k) & (XT - 1)];
          // байтовый наездник СОГЛАСОВАННЫЙ (соседние полосы -- соседние слова), иначе меряли бы
          // не байты, а рассеяние: 32 сектора на варп вместо 4.
          if constexpr (KIND == KIND_LDG)  ru[r % ILP] += scratch[((unsigned)(n + ((k0 + gk + k) >> 2) * N_DIM + r * 4099)) & smask];
        }
        // ------------------------------------------------------------------------
      }
      acc = fmaf(s, qa - z * xa, acc);
    }
  }
  if (!live) return;
  // Сток наездника: eps=0 в бою, но компилятор этого не знает -- работа не выкидывается.
  float sink = 0.f;
  #pragma unroll
  for (int i = 0; i < ILP; ++i) sink += rf[i] + (float)ru[i];
  atomicAdd(ypart + n, acc + eps * sink);
}

// -------------------------------------------------------------------------------------------------
struct Cfg { const char* name; int R; int kind; int ilp; void (*launch)(dim3, dim3, const void**);
             const void* fn; };

static const __half* d_x; static const int* d_qw; static const int* d_qz; static const __half* d_sc;
static float* d_y; static unsigned* d_scratch; static int g_gps; static unsigned g_smask;

template <int R, int KIND, int ILP>
static const void* fnptr() { return (const void*)w8a16_shadow<R, KIND, ILP>; }

template <int R, int KIND, int ILP>
static void launcher(dim3 grid, dim3 blk, const void**) {
  w8a16_shadow<R, KIND, ILP><<<grid, blk>>>(d_x, d_qw, d_qz, d_sc, d_y, d_scratch,
                                            1, N_DIM, K_DIM, G_DIM, g_gps, 0.f, g_smask);
}

#define CFG(R, KIND, ILP) { #KIND "/R=" #R "/ilp" #ILP, R, KIND, ILP, launcher<R, KIND, ILP>, \
                            (const void*)w8a16_shadow<R, KIND, ILP> }

int main(int argc, char** argv) {
  const int reps = (argc > 1) ? atoi(argv[1]) : 11;
  // argv[2] -- ПРИНУДИТЕЛЬНОЕ число k-срезов. Тот же кирпич даёт ВТОРОЙ режим хозяина:
  //   0  -> боевая формула (~7 блоков/SM, упор в ПОЛОСУ, 650-670 ГБ/с);
  //   1  -> 20 блоков на 80 SM, упор в ЗАДЕРЖКУ при простое ОБОИХ ресурсов (48 ГБ/с, см. шапку
  //         w8a16_gemv.cu) -- это архетип ДЕКОДНОГО внимания и недозаполненного скана.
  const int ksplit_force = (argc > 2) ? atoi(argv[2]) : 0;
  CK(cudaSetDevice(0));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("# карта %s, SM %d, такт %d МГц\n", p.name, p.multiProcessorCount, p.clockRate / 1000);

  const long qw_words = (long)(K_DIM / 4) * N_DIM;
  const int ngroups = K_DIM / G_DIM;
  std::vector<int> h_qw(qw_words); for (long i = 0; i < qw_words; ++i) h_qw[i] = (int)(i * 2654435761u);
  std::vector<int> h_qz((long)ngroups * (N_DIM / 4), 0x80808080);
  std::vector<__half> h_sc((long)ngroups * N_DIM, __float2half(0.01f));
  std::vector<__half> h_x(K_DIM); for (int i = 0; i < K_DIM; ++i) h_x[i] = __float2half(0.01f * ((i % 7) - 3));

  void *pqw, *pqz, *psc, *px, *py, *pscr;
  CK(cudaMalloc(&pqw, qw_words * 4));            CK(cudaMemcpy(pqw, h_qw.data(), qw_words * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&pqz, h_qz.size() * 4));         CK(cudaMemcpy(pqz, h_qz.data(), h_qz.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&psc, h_sc.size() * 2));         CK(cudaMemcpy(psc, h_sc.data(), h_sc.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&px,  K_DIM * 2));               CK(cudaMemcpy(px,  h_x.data(),  K_DIM * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&py,  N_DIM * 4));
  const size_t scr_words = 64u << 20;            // 256 МБ -- гарантированно мимо L2 (6 МБ)
  CK(cudaMalloc(&pscr, scr_words * 4));          CK(cudaMemset(pscr, 1, scr_words * 4));
  d_qw = (const int*)pqw; d_qz = (const int*)pqz; d_sc = (const __half*)psc;
  d_x = (const __half*)px; d_y = (float*)py; d_scratch = (unsigned*)pscr;
  g_smask = (unsigned)(scr_words - 1);

  // Сетка -- боевая (та же формула, что в w8a16_gemv): ~8 блоков на SM.
  const int nblk = N_DIM / THREADS;
  int ksplit = ksplit_force ? ksplit_force
                            : std::max(1, std::min(ngroups, (80 * 8 + nblk - 1) / nblk));
  g_gps = (ngroups + ksplit - 1) / ksplit;
  ksplit = (ngroups + g_gps - 1) / g_gps;
  const dim3 grid(nblk, ksplit), blk(THREADS);
  const double bytes = (double)qw_words * 4;
  printf("# сетка %dx%d = %d блоков (%.1f на SM), вес %.1f МБ, слов на поток %ld\n",
         nblk, ksplit, nblk * ksplit, nblk * ksplit / 80.0, bytes / 1e6, (long)(K_DIM / 4));

  std::vector<Cfg> cfg = {
    CFG(0, KIND_NONE, 1),
    CFG(1, KIND_FFMA, 1), CFG(2, KIND_FFMA, 1), CFG(4, KIND_FFMA, 1), CFG(6, KIND_FFMA, 1),
    CFG(8, KIND_FFMA, 1), CFG(12, KIND_FFMA, 1), CFG(16, KIND_FFMA, 1), CFG(24, KIND_FFMA, 1),
    CFG(32, KIND_FFMA, 1), CFG(48, KIND_FFMA, 1), CFG(64, KIND_FFMA, 1), CFG(96, KIND_FFMA, 1),
    CFG(4, KIND_FFMA, 4), CFG(8, KIND_FFMA, 4), CFG(16, KIND_FFMA, 4), CFG(24, KIND_FFMA, 4),
    CFG(32, KIND_FFMA, 4), CFG(48, KIND_FFMA, 4), CFG(64, KIND_FFMA, 4),
    CFG(4, KIND_PRMT, 1), CFG(8, KIND_PRMT, 1), CFG(16, KIND_PRMT, 1), CFG(24, KIND_PRMT, 1),
    CFG(32, KIND_PRMT, 1), CFG(48, KIND_PRMT, 1), CFG(64, KIND_PRMT, 1), CFG(96, KIND_PRMT, 1),
    CFG(4, KIND_PRMT, 4), CFG(16, KIND_PRMT, 4), CFG(32, KIND_PRMT, 4), CFG(64, KIND_PRMT, 4),
    CFG(1, KIND_MUFU, 1), CFG(2, KIND_MUFU, 1), CFG(4, KIND_MUFU, 1), CFG(8, KIND_MUFU, 1),
    CFG(16, KIND_MUFU, 1), CFG(24, KIND_MUFU, 1), CFG(32, KIND_MUFU, 1),
    CFG(1, KIND_LDS, 1), CFG(2, KIND_LDS, 1), CFG(4, KIND_LDS, 1), CFG(8, KIND_LDS, 1),
    CFG(12, KIND_LDS, 1), CFG(16, KIND_LDS, 1), CFG(24, KIND_LDS, 1), CFG(32, KIND_LDS, 1),
    CFG(4, KIND_LDS, 4), CFG(16, KIND_LDS, 4),
    CFG(1, KIND_LDG, 1), CFG(2, KIND_LDG, 1), CFG(4, KIND_LDG, 1), CFG(8, KIND_LDG, 1),
  };
  const int NC = (int)cfg.size();
  std::vector<std::vector<float>> t(NC);

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int c = 0; c < NC; ++c) { CK(cudaMemset(py, 0, N_DIM * 4)); cfg[c].launch(grid, blk, nullptr); }
  CK(cudaDeviceSynchronize());
  // ЧЕРЕДОВАНИЕ конфигураций в ОДНОМ процессе (EPILOG §8.6): дрейф частот бьёт по всем одинаково.
  for (int it = 0; it < reps; ++it) {
    for (int c = 0; c < NC; ++c) {
      CK(cudaMemset(py, 0, N_DIM * 4));
      CK(cudaEventRecord(e0));
      cfg[c].launch(grid, blk, nullptr);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t[c].push_back(ms);
    }
  }
  printf("# конфигурация; мкс(МИНИМУМ); мкс(медиана); разброс%%; ГБ/с(мин); отн.базы(мин); рег; блоков/SM\n");
  double base = 0;
  for (int c = 0; c < NC; ++c) {
    std::sort(t[c].begin(), t[c].end());
    const double us = t[c][0] * 1e3, med = t[c][reps / 2] * 1e3;
    if (c == 0) base = us;
    cudaFuncAttributes at; cudaFuncGetAttributes(&at, cfg[c].fn);
    int occ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, cfg[c].fn, THREADS, 0);
    printf("%-24s %9.2f %9.2f %7.1f %8.1f %8.4f %5d %5d\n", cfg[c].name, us, med,
           100.0 * (t[c][reps - 1] - t[c][0]) / t[c][0], bytes / (us * 1e3), us / base,
           at.numRegs, occ);
  }
  return 0;
}
