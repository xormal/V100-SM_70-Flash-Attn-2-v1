// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Проверка и замер ядра beta*K*K^T (volta_gdn_kkt.cuh) против ПРЯМОГО эталона на CPU в fp64.
// Эталон считается ПО ОПРЕДЕЛЕНИЮ, а не вызовом другого ядра: иначе общая ошибка в понимании формулы
// прошла бы обе стороны сверки незамеченной.
//
// Вход повторяет боевую статистику, иначе тест ничего не доказывает:
//   k    -- нормальный шум, приведённый к ЕДИНИЧНОЙ L2-норме по каналу и округлённый до fp16
//           (в бою перед вызовом идёт l2norm_fwd, поэтому |k_i . k_j| <= 1);
//   beta -- sigmoid(нормальный шум) в fp16, то есть (0,1);
//   g    -- накопленная ВНУТРИ чанка сумма ОТРИЦАТЕЛЬНЫХ шагов -0.5..-0.01: при неверном знаке гейта
//           это даёт не inf, а тихую потерю точности, и такой вход её ловит.
//
// Сборка: nvcc -arch=sm_70 -O3 -std=c++17 -I fa2_src/fmha_kernel/gemm \
//              -o gdnkkt tools/volta_gdn_kkt_test.cu -Wno-deprecated-gpu-targets
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../fa2_sm70/csrc/volta_gdn_kkt.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ОТКАЗ %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static constexpr int BT = 64, DK = 128;

// --- воспроизводимый шум -------------------------------------------------------------------------
static unsigned long long g_rs = 88172645463325252ull;
static double urnd() {                       // (0,1)
  g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17;
  return ((g_rs >> 11) + 1.0) / 9007199254740994.0;
}
static double nrnd() { return sqrt(-2.0 * log(urnd())) * cos(6.283185307179586 * urnd()); }

struct Inputs {
  int T, H, Hg, GQ;
  std::vector<int> cu;                      // [N+1]
  std::vector<int> ci;                      // [NT][2]
  std::vector<__half> k, beta;
  std::vector<float> g;
};

// cu -- границы последовательностей; строит k/beta/g и таблицу чанков ровно как prepare_chunk_indices.
static Inputs make_inputs(const std::vector<int>& cu, int H, int Hg) {
  Inputs in;
  in.cu = cu; in.H = H; in.Hg = Hg; in.GQ = H / Hg;
  in.T = cu.back();
  for (size_t n = 0; n + 1 < cu.size(); ++n) {
    const int len = cu[n + 1] - cu[n], nt = (len + BT - 1) / BT;
    for (int t = 0; t < nt; ++t) { in.ci.push_back((int)n); in.ci.push_back(t); }
  }
  in.k.resize((size_t)in.T * Hg * DK);
  in.beta.resize((size_t)in.T * H);
  in.g.resize((size_t)in.T * H);
  for (long t = 0; t < in.T; ++t)
    for (int hk = 0; hk < Hg; ++hk) {                       // строка ЕДИНИЧНОЙ нормы, как после l2norm
      double v[DK], s = 0;
      for (int c = 0; c < DK; ++c) { v[c] = nrnd(); s += v[c] * v[c]; }
      const double inv = 1.0 / sqrt(s);
      for (int c = 0; c < DK; ++c) in.k[((size_t)t * Hg + hk) * DK + c] = __float2half((float)(v[c] * inv));
    }
  for (long t = 0; t < in.T; ++t)
    for (int h = 0; h < H; ++h)
      in.beta[(size_t)t * H + h] = __float2half((float)(1.0 / (1.0 + exp(-nrnd()))));
  // g -- ВНУТРИЧАНКОВЫЙ накопленный лог-гейт: сумма отрицательных шагов, сбрасывается на границе чанка
  // И на границе последовательности (chunk_local_cumsum считает по чанкам внутри последовательности).
  for (size_t n = 0; n + 1 < cu.size(); ++n) {
    const int bos = cu[n], len = cu[n + 1] - bos;
    for (int h = 0; h < H; ++h) {
      double acc = 0;
      for (int i = 0; i < len; ++i) {
        if (i % BT == 0) acc = 0;
        acc += -(0.01 + 0.49 * urnd());
        in.g[(size_t)(bos + i) * H + h] = (float)acc;
      }
    }
  }
  return in;
}

// --- ЭТАЛОН ПО ОПРЕДЕЛЕНИЮ, fp64 на CPU. Считается для чанков [0, nchk) таблицы chunk_indices.
static void reference(const Inputs& in, int nchk, std::vector<double>& out) {
  const int H = in.H, Hg = in.Hg, GQ = in.GQ;
  out.assign((size_t)nchk * H * BT * BT, 0.0);
  for (int c = 0; c < nchk; ++c) {
    const int i_n = in.ci[2 * c], i_t = in.ci[2 * c + 1];
    const int bos = in.cu[i_n], Tn = in.cu[i_n + 1] - bos, t0 = i_t * BT;
    const int rows = std::min(BT, Tn - t0);
    for (int h = 0; h < H; ++h) {
      const int hk = h / GQ;
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < i; ++j) {           // строго нижний треугольник, диагональ -- НОЛЬ
          double dot = 0;
          for (int cc = 0; cc < DK; ++cc)
            dot += (double)__half2float(in.k[((size_t)(bos + t0 + i) * Hg + hk) * DK + cc]) *
                   (double)__half2float(in.k[((size_t)(bos + t0 + j) * Hg + hk) * DK + cc]);
          const double bi = (double)__half2float(in.beta[(size_t)(bos + t0 + i) * H + h]);
          const double gi = (double)in.g[(size_t)(bos + t0 + i) * H + h];
          const double gj = (double)in.g[(size_t)(bos + t0 + j) * H + h];
          out[(((size_t)c * H + h) * BT + i) * BT + j] = bi * exp(gi - gj) * dot;
        }
    }
  }
}

struct Dev { __half *k, *beta; float *g, *A; int *cu, *ci; };

// --- ПОЛ ПО ТРАФИКУ: то же чтение k и та же запись A, но БЕЗ единого умножения. Нужен, чтобы не
// гадать, упирается ядро в память или во что-то ещё: разница между этим и полным ядром -- ровно цена
// счёта и разделяемой памяти. Загрузка обязана быть ПОТРЕБЛЕНА, иначе компилятор удалит её целиком
// (SM70_LATENCY_HIDING §10.2: чтение, результат которого никуда не идёт, исчезает вместе с LDG).
template <int BT_, int DK_, int GQ>
__global__ __launch_bounds__(256, 1) void kkt_traffic_floor(const __half* __restrict__ k,
                                                            float* __restrict__ A,
                                                            int T, int H, int Hg) {
  const int it = blockIdx.x, ihg = blockIdx.y, tid = threadIdx.x;
  const int t0 = it * BT_, rows = min(BT_, T - t0);
  if (rows <= 0) return;
  float s = 0.f;
  for (int i = tid; i < BT_ * (DK_ / 8); i += 256) {
    const int r = i / (DK_ / 8), c = (i % (DK_ / 8)) * 8;
    if (r < rows) {
      const uint4 v = *reinterpret_cast<const uint4*>(&k[((long)(t0 + r) * Hg + ihg) * DK_ + c]);
      s += __half2float(*reinterpret_cast<const __half*>(&v.x)) +
           __half2float(*reinterpret_cast<const __half*>(&v.w));
    }
  }
  const float4 o = make_float4(s, s, s, s);
  constexpr int RQ = GQ * (BT_ / 4);
  for (int i = tid; i < BT_ * RQ; i += 256) {
    const int r = i / RQ, kk = i % RQ;
    if (r >= rows) continue;
    const int hq = kk / (BT_ / 4), c0 = (kk % (BT_ / 4)) * 4;
    *reinterpret_cast<float4*>(&A[((long)(t0 + r) * H + ihg * GQ + hq) * BT_ + c0]) = o;
  }
}

// Сверка: возвращает максимальную АБСОЛЮТНУЮ ошибку (масштаб выхода |A| <= beta <= 1, поэтому
// абсолютная и относительная здесь совпадают по смыслу) и максимум эталона.
static void compare(const Inputs& in, const std::vector<double>& ref, int nchk,
                    const std::vector<float>& got, double& emax, double& rmax, long& nbad) {
  const int H = in.H;
  emax = 0; rmax = 0; nbad = 0;
  for (int c = 0; c < nchk; ++c) {
    const int i_n = in.ci[2 * c], i_t = in.ci[2 * c + 1];
    const int bos = in.cu[i_n], Tn = in.cu[i_n + 1] - bos, t0 = i_t * BT;
    const int rows = std::min(BT, Tn - t0);
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < BT; ++j) {
          const double r = ref[(((size_t)c * H + h) * BT + i) * BT + j];
          const double v = got[((size_t)(bos + t0 + i) * H + h) * BT + j];
          if (!(v == v) || fabs(v) > 1e30) { ++nbad; continue; }   // NaN/inf/несделанная запись
          const double e = fabs(v - r);
          if (e > emax) emax = e;
          if (fabs(r) > rmax) rmax = fabs(r);
        }
  }
}

template <int GQ, int WM, int WN, bool SKIP>
static double launch(const Inputs& in, const Dev& d, int iters, bool varlen) {
  auto kern = fa2_sm70::gdn::volta_gdn_kkt<BT, DK, GQ, WM, WN, SKIP>;
  const size_t sh = fa2_sm70::gdn::volta_gdn_kkt_smem<BT, DK>();
  const int NT = (int)in.ci.size() / 2;
  dim3 grid(NT, in.Hg);
  const int* cu = varlen ? d.cu : nullptr;
  const int* ci = varlen ? d.ci : nullptr;
  for (int i = 0; i < 5; ++i) kern<<<grid, WM * WN * 32, sh>>>(d.k, d.beta, d.g, d.A, cu, ci, in.T, in.H, in.Hg);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  if (iters <= 0) return 0.0;
  // усечённое среднее: отбрасываем по одному крайнему замеру с каждой стороны
  std::vector<double> ms;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int r = 0; r < iters; ++r) {
    CK(cudaEventRecord(a));
    kern<<<grid, WM * WN * 32, sh>>>(d.k, d.beta, d.g, d.A, cu, ci, in.T, in.H, in.Hg);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float t; CK(cudaEventElapsedTime(&t, a, b)); ms.push_back(t);
  }
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  std::sort(ms.begin(), ms.end());
  double s = 0; int n = 0;
  for (size_t i = ms.size() / 8; i < ms.size() - ms.size() / 8; ++i) { s += ms[i]; ++n; }
  return s / n;
}

// Инварианты хвоста, проверяемые ОТДЕЛЬНО от значений (ловят то, чего сверка не видит):
//   (а) строки за пределом последовательности не тронуты (в A лежит сигнальное значение);
//   (б) столбцы j >= rows внутри записанных строк -- явный ноль;
//   (в) диагональ всюду ноль.
static int check_tail(const Inputs& in, const std::vector<float>& got, float sentinel) {
  const int H = in.H;
  int bad = 0;
  std::vector<char> written((size_t)in.T, 0);
  for (size_t c = 0; c * 2 < in.ci.size(); ++c) {
    const int i_n = in.ci[2 * c], i_t = in.ci[2 * c + 1];
    const int bos = in.cu[i_n], Tn = in.cu[i_n + 1] - bos, t0 = i_t * BT;
    const int rows = std::min(BT, Tn - t0);
    for (int i = 0; i < rows; ++i) written[bos + t0 + i] = 1;
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < rows; ++i) {
        const float* row = &got[((size_t)(bos + t0 + i) * H + h) * BT];
        if (row[i] != 0.f) { if (++bad < 5) printf("  (в) диагональ != 0 при t=%d h=%d\n", bos + t0 + i, h); }
        for (int j = rows; j < BT; ++j)
          if (row[j] != 0.f) { if (++bad < 5) printf("  (б) столбец %d >= rows=%d не ноль\n", j, rows); }
      }
  }
  for (int t = 0; t < in.T; ++t)
    if (!written[t])
      for (int h = 0; h < H; ++h)
        for (int j = 0; j < BT; ++j)
          if (got[((size_t)t * H + h) * BT + j] != sentinel) {
            if (++bad < 5) printf("  (а) строка t=%d затёрта чужим блоком\n", t);
            h = H; break;
          }
  return bad;
}


// Полная проверка на ЗАДАННОЙ раскладке голов и списке последовательностей: сверка с fp64-эталоном по
// ВСЕМ чанкам плюс три инварианта хвоста. Отдельная функция нужна потому, что GQ -- параметр шаблона,
// и путь эпилога у GQ=1/2/3 РАЗНЫЙ (меняется длина сплошного куска строки).
template <int GQ>
static int verify(const std::vector<int>& cu, int H, int Hg, const char* name) {
  Inputs in = make_inputs(cu, H, Hg);
  const int NT = (int)in.ci.size() / 2;
  Dev d{};
  CK(cudaMalloc(&d.k, in.k.size() * 2));    CK(cudaMalloc(&d.beta, in.beta.size() * 2));
  CK(cudaMalloc(&d.g, in.g.size() * 4));    CK(cudaMalloc(&d.A, (size_t)in.T * H * BT * 4));
  CK(cudaMalloc(&d.cu, in.cu.size() * 4));  CK(cudaMalloc(&d.ci, in.ci.size() * 4));
  CK(cudaMemcpy(d.k, in.k.data(), in.k.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.beta, in.beta.data(), in.beta.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.g, in.g.data(), in.g.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.cu, in.cu.data(), in.cu.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.ci, in.ci.data(), in.ci.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(d.A, 0x7f, (size_t)in.T * H * BT * 4));
  float sentinel; unsigned sb = 0x7f7f7f7fu; memcpy(&sentinel, &sb, 4);
  launch<GQ, 2, 4, false>(in, d, 0, true);
  std::vector<float> got((size_t)in.T * H * BT);
  CK(cudaMemcpy(got.data(), d.A, got.size() * 4, cudaMemcpyDeviceToHost));
  std::vector<double> ref; reference(in, NT, ref);
  double emax, rmax; long nbad; compare(in, ref, NT, got, emax, rmax, nbad);
  const int bad = check_tail(in, got, sentinel);
  printf("%-34s H=%2d Hg=%2d GQ=%d NT=%2d | ошибка %.2e | NaN/пропусков %ld | инварианты (а)(б)(в): %s\n",
         name, H, Hg, GQ, NT, emax / (rmax + 1e-30), nbad, bad == 0 ? "ВСЕ ВЫПОЛНЕНЫ" : "НАРУШЕНЫ");
  CK(cudaFree(d.k)); CK(cudaFree(d.beta)); CK(cudaFree(d.g));
  CK(cudaFree(d.A)); CK(cudaFree(d.cu)); CK(cudaFree(d.ci));
  return bad + (int)nbad;
}

int main(int argc, char** argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 15680;
  const int H = argc > 2 ? atoi(argv[2]) : 24;
  const int Hg = argc > 3 ? atoi(argv[3]) : 8;
  const int NCHK = argc > 4 ? atoi(argv[4]) : 4;         // сколько чанков сверять эталоном
  if (H % Hg != 0 || H / Hg != 3) { printf("этот стенд собран под GQ = H/Hg = 3\n"); return 1; }

  int dev = 0; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
  printf("=== %s, SM %d, такт %d МГц ===\n", prop.name, prop.multiProcessorCount, clk / 1000);

  // ---------------------------------------------------------------- боевая форма: одна последовательность
  Inputs in = make_inputs({0, T}, H, Hg);
  const int NT = (int)in.ci.size() / 2;
  printf("T=%d H=%d Hg=%d NT=%d  сетка %dx%d = %d блоков\n", T, H, Hg, NT, NT, Hg, NT * Hg);

  Dev d{};
  CK(cudaMalloc(&d.k, in.k.size() * 2));       CK(cudaMalloc(&d.beta, in.beta.size() * 2));
  CK(cudaMalloc(&d.g, in.g.size() * 4));       CK(cudaMalloc(&d.A, (size_t)T * H * BT * 4));
  CK(cudaMalloc(&d.cu, in.cu.size() * 4));     CK(cudaMalloc(&d.ci, in.ci.size() * 4));
  CK(cudaMemcpy(d.k, in.k.data(), in.k.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.beta, in.beta.data(), in.beta.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.g, in.g.data(), in.g.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.cu, in.cu.data(), in.cu.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.ci, in.ci.data(), in.ci.size() * 4, cudaMemcpyHostToDevice));

  std::vector<double> ref; reference(in, std::min(NCHK, NT), ref);
  std::vector<float> got((size_t)T * H * BT);

  // Работа: Triton считает ПЛОТНЫЙ квадрат на КАЖДУЮ голову значения (NT*H*BT*BT*DK MAC).
  // Мы считаем его один раз на голову КЛЮЧА -- вот две разные меры, и обе надо назвать честно.
  const double flop_triton = 2.0 * NT * H * BT * BT * DK;
  const double flop_ours = 2.0 * NT * Hg * BT * BT * DK;
  const double bytes = (double)T * H * BT * 4 + (double)T * Hg * DK * 2 + (double)T * H * 4 * 3;

  printf("\n сетка варпов | плитка | рег | разл | бл/SM | варп/SM | занят | время, мс | ГБ/с |"
         " наш TFLOP/s | в мерке Triton | ошибка\n");
  double best = 1e30; const char* bestn = "";

#define RUN(WM, WN, SKIP, NAME)                                                                     \
  do {                                                                                              \
    CK(cudaMemset(d.A, 0x7f, (size_t)T * H * BT * 4));                                              \
    const double ms = launch<3, WM, WN, SKIP>(in, d, 30, true);                                     \
    CK(cudaMemcpy(got.data(), d.A, got.size() * 4, cudaMemcpyDeviceToHost));                        \
    double emax, rmax; long nbad;                                                                   \
    compare(in, ref, std::min(NCHK, NT), got, emax, rmax, nbad);                                    \
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa,                                            \
        (const void*)fa2_sm70::gdn::volta_gdn_kkt<BT, DK, 3, WM, WN, SKIP>));                       \
    int bps = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps,                             \
        (const void*)fa2_sm70::gdn::volta_gdn_kkt<BT, DK, 3, WM, WN, SKIP>, (WM) * (WN) * 32,       \
        fa2_sm70::gdn::volta_gdn_kkt_smem<BT, DK>()));                                              \
    const int wsm = bps * (WM) * (WN);                                                              \
    printf(" %-12s | %2dx%-3d | %3d | %4d | %5d | %7d | %4.1f%% | %9.4f | %4.0f | %11.1f | %14.1f"  \
           " | %.2e%s\n", NAME, BT / (WM), BT / (WN), (int)fa.numRegs, (int)fa.localSizeBytes, bps, \
           wsm, 100.0 * wsm / 64.0, ms, bytes / (ms * 1e-3) / 1e9,                                  \
           flop_ours / (ms * 1e-3) / 1e12, flop_triton / (ms * 1e-3) / 1e12,                        \
           emax / (rmax + 1e-30), nbad ? "  ЕСТЬ NaN/inf/пропуск!" : "");                           \
    if (ms < best) { best = ms; bestn = NAME; }                                                     \
  } while (0)

  RUN(2, 4, false, "2x4 (8в)");
  RUN(4, 2, false, "4x2 (8в)");
  RUN(4, 4, false, "4x4 (16в)");
  RUN(2, 2, false, "2x2 (4в)");
  RUN(1, 4, false, "1x4 (4в)");
  printf("лучшая: %s  %.4f мс\n", bestn, best);

  // Пол по трафику: то же чтение и та же запись без счёта. Он и есть ответ на вопрос "куда упёрлись".
  {
    std::vector<double> t;
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    dim3 grid(NT, Hg);
    for (int i = 0; i < 5; ++i) kkt_traffic_floor<BT, DK, 3><<<grid, 256>>>(d.k, d.A, T, H, Hg);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    for (int r = 0; r < 30; ++r) {
      CK(cudaEventRecord(a));
      kkt_traffic_floor<BT, DK, 3><<<grid, 256>>>(d.k, d.A, T, H, Hg);
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float x; CK(cudaEventElapsedTime(&x, a, b)); t.push_back(x);
    }
    std::sort(t.begin(), t.end());
    double s = 0; int n = 0;
    for (size_t i = t.size() / 8; i < t.size() - t.size() / 8; ++i) { s += t[i]; ++n; }
    const double ms = s / n;
    printf("пол по трафику (то же чтение k + та же запись A, БЕЗ счёта): %.4f мс, %.0f ГБ/с"
           "  -> счёт и разделяемая стоят %.1f%%\n",
           ms, bytes / (ms * 1e-3) / 1e9, 100.0 * (best - ms) / ms);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  }

  // Урезанная запись: шесть СТРОГО ВЕРХНИХ блоков 16x16 не читает никто (единственный потребитель --
  // merge_16x16_to_64x64_inverse_kernel в solve_tril). МЕНЯЕТ КОНТРАКТ ВЫХОДА, поэтому только замер.
  printf("\n[справочно] запись без шести строго верхних блоков 16x16 (-37.5%% трафика записи):\n");
  {
    const double keep = (double)T * H * BT * 4 * 0.625 + (double)T * Hg * DK * 2 + (double)T * H * 4 * 3;
    const double b_save = bytes; (void)b_save;
    CK(cudaMemset(d.A, 0x7f, (size_t)T * H * BT * 4));
    const double ms = launch<3, 2, 4, true>(in, d, 30, true);
    printf(" %-12s | %2dx%-3d | %9.4f мс | %4.0f ГБ/с | %.1f TFLOP/s (в мерке Triton)"
           "  (верх не сверяется)\n",
           "2x4 SKIP", BT / 2, BT / 4, ms, keep / (ms * 1e-3) / 1e9,
           flop_triton / (ms * 1e-3) / 1e12);
  }

  // ------------------------------------------------------- ХВОСТ, varlen и другие раскладки голов.
  // Первая последовательность НЕ кратна 64: только здесь проверяется, что хвостовой блок не пишет в
  // строки СЛЕДУЮЩЕЙ последовательности. GQ=1 и GQ=2 -- другие пути эпилога, их надо сверять отдельно.
  printf("\n=== корректность на всех чанках: хвост, varlen, разные раскладки голов ===\n");
  int bad = 0;
  bad += verify<3>({0, 197, 517}, 24, 8, "varlen [0,197,517], хвост 5 и 5");
  bad += verify<3>({0, 640}, 24, 8, "одна посл., ровно 10 чанков");
  bad += verify<3>({0, 1000}, 24, 8, "одна посл., хвост 40 строк");
  bad += verify<2>({0, 197, 517}, 16, 8, "varlen, GQ=2");
  bad += verify<1>({0, 197, 517}, 8, 8, "varlen, GQ=1 (H == Hg)");
  bad += verify<1>({0, 63}, 8, 8, "одна посл. КОРОЧЕ чанка (63)");
  if (bad) { printf("ЕСТЬ ОТКАЗЫ\n"); return 1; }
  printf("все проверки пройдены\n");
  return 0;
}
