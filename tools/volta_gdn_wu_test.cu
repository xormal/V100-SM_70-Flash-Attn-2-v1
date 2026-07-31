// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Проверка и замер ядра пересчёта w/u для gated-delta-net (volta_gdn_wu.cuh).
//
// ЭТАЛОН СЧИТАЕТСЯ ПО ОПРЕДЕЛЕНИЮ В fp64 НА CPU, а не вызовом другого ядра: иначе общая ошибка в
// понимании формулы прошла бы обе стороны сверки незамеченной. Матрица A строится НЕЗАВИСИМО --
// из fp64 k, beta, g собирается Araw[i][j] = beta_i * <k_i,k_j> * e^{g_i-g_j} (i>j), и (I+Araw)
// обращается прямой подстановкой в fp64 (делений нет: диагональ единичная по построению, §4pp
// библиотеки примитивов). Ядру отдаётся fp16-округление этой матрицы, эталон берёт ТЕ ЖЕ fp16-числа
// как double -- так меряется ошибка ЯДРА, а не квантование входа.
//
// Сборка:
//   nvcc -arch=sm_70 -O3 -std=c++17 -Xptxas -v -I fa2_src/fmha_kernel/gemm \
//        -o gdnwu tools/volta_gdn_wu_test.cu -Wno-deprecated-gpu-targets
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../fa2_sm70/csrc/volta_gdn_wu.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ОТКАЗ %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static constexpr int BT = 64, DK = 128, DV = 128;

static double h2d(__half h) { return (double)__half2float(h); }

// ------------------------------------------------------------------------------------------------
// Генерация входа. Раскладка ровно как у Triton: k [T,Hg,DK], v [T,H,DV], beta/g [T,H], A [T,H,BT].
// Строки k нормированы (в модели перед этим шагом стоит l2norm), beta в (0,1) -- сигмоида,
// g -- ВНУТРИЧАНКОВЫЙ включающий кумсум неположительных логарифмов гейта.
// ------------------------------------------------------------------------------------------------
struct Tensors {
  int T, H, Hg, nseq;
  std::vector<__half> k, v, beta, A;
  std::vector<float> g;
  std::vector<int> cu;          // [nseq+1]
  std::vector<int> ci;          // [NT_total][2]
};

static void gen(Tensors& t, int H, int Hg, const std::vector<int>& lens, bool exactA, unsigned seed) {
  t.H = H; t.Hg = Hg; t.nseq = (int)lens.size();
  t.cu.assign(t.nseq + 1, 0);
  for (int n = 0; n < t.nseq; ++n) t.cu[n + 1] = t.cu[n] + lens[n];
  t.T = t.cu[t.nseq];
  t.ci.clear();
  for (int n = 0; n < t.nseq; ++n)
    for (int c = 0; c < (lens[n] + BT - 1) / BT; ++c) { t.ci.push_back(n); t.ci.push_back(c); }

  srand(seed);
  auto rnd = [] { return (double)rand() / RAND_MAX * 2.0 - 1.0; };
  t.k.resize((size_t)t.T * Hg * DK);
  t.v.resize((size_t)t.T * H * DV);
  t.beta.resize((size_t)t.T * H);
  t.g.resize((size_t)t.T * H);
  t.A.assign((size_t)t.T * H * BT, __float2half(0.f));

  for (int i = 0; i < t.T * Hg; ++i) {          // строка k единичной нормы
    double s = 0, x[DK];
    for (int e = 0; e < DK; ++e) { x[e] = rnd(); s += x[e] * x[e]; }
    s = 1.0 / sqrt(s);
    for (int e = 0; e < DK; ++e) t.k[(size_t)i * DK + e] = __float2half((float)(x[e] * s));
  }
  for (auto& x : t.v) x = __float2half((float)(rnd() * 0.5));
  for (int i = 0; i < t.T * H; ++i) {
    t.beta[i] = __float2half((float)(0.05 + 0.9 * ((double)rand() / RAND_MAX)));
    t.g[i] = 0.f;                                // заполняется ниже кумсумом внутри чанка
  }
  for (int n = 0; n < t.nseq; ++n)
    for (int i = 0; i < lens[n]; ++i)
      for (int h = 0; h < H; ++h)
        t.g[(size_t)(t.cu[n] + i) * H + h] = (float)(-0.03 * (i % BT + 1) - 0.002 * h);

  // --- A: либо настоящая (I+Araw)^{-1} в fp64, либо дешёвая унитреугольная (для замера скорости)
  for (int n = 0; n < t.nseq; ++n) {
    const int L = lens[n], NC = (L + BT - 1) / BT;
    for (int c = 0; c < NC; ++c) {
      const int r0 = t.cu[n] + c * BT, rows = std::min(BT, L - c * BT);
      for (int h = 0; h < H; ++h) {
        const int hk = h / (H / Hg);
        std::vector<double> M((size_t)BT * BT, 0.0), X((size_t)BT * BT, 0.0);
        for (int i = 0; i < rows; ++i) M[(size_t)i * BT + i] = 1.0;
        if (exactA) {
          for (int i = 1; i < rows; ++i)
            for (int j = 0; j < i; ++j) {
              double d = 0;
              for (int e = 0; e < DK; ++e)
                d += h2d(t.k[((size_t)(r0 + i) * Hg + hk) * DK + e]) *
                     h2d(t.k[((size_t)(r0 + j) * Hg + hk) * DK + e]);
              M[(size_t)i * BT + j] = h2d(t.beta[(size_t)(r0 + i) * H + h]) * d *
                                      exp((double)t.g[(size_t)(r0 + i) * H + h] -
                                          (double)t.g[(size_t)(r0 + j) * H + h]);
            }
        } else {
          for (int i = 1; i < rows; ++i)
            for (int j = 0; j < i; ++j) M[(size_t)i * BT + j] = rnd() * 0.1;
        }
        // Обращение единичной нижнетреугольной: НИ ОДНОГО деления (диагональ -- единицы).
        for (int i = 0; i < rows; ++i) {
          X[(size_t)i * BT + i] = 1.0;
          for (int j = 0; j < i; ++j) {
            double s = 0;
            for (int m = j; m < i; ++m) s += M[(size_t)i * BT + m] * X[(size_t)m * BT + j];
            X[(size_t)i * BT + j] = -s;
          }
        }
        for (int i = 0; i < rows; ++i)
          for (int j = 0; j <= i; ++j)
            t.A[((size_t)(r0 + i) * H + h) * BT + j] = __float2half((float)X[(size_t)i * BT + j]);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
struct Dev {
  __half *k, *v, *beta, *A, *w, *u;
  float* g;
  int *cu, *ci;
  int NT;
};

static void upload(const Tensors& t, Dev& d) {
  CK(cudaMalloc(&d.k, t.k.size() * 2));    CK(cudaMemcpy(d.k, t.k.data(), t.k.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.v, t.v.size() * 2));    CK(cudaMemcpy(d.v, t.v.data(), t.v.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.beta, t.beta.size() * 2));
  CK(cudaMemcpy(d.beta, t.beta.data(), t.beta.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.A, t.A.size() * 2));    CK(cudaMemcpy(d.A, t.A.data(), t.A.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.g, t.g.size() * 4));    CK(cudaMemcpy(d.g, t.g.data(), t.g.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.cu, t.cu.size() * 4));  CK(cudaMemcpy(d.cu, t.cu.data(), t.cu.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.ci, t.ci.size() * 4));  CK(cudaMemcpy(d.ci, t.ci.data(), t.ci.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d.w, (size_t)t.T * t.H * DK * 2));
  CK(cudaMalloc(&d.u, (size_t)t.T * t.H * DV * 2));
  d.NT = (int)t.ci.size() / 2;
}
static void freeDev(Dev& d) {
  cudaFree(d.k); cudaFree(d.v); cudaFree(d.beta); cudaFree(d.A);
  cudaFree(d.g); cudaFree(d.cu); cudaFree(d.ci); cudaFree(d.w); cudaFree(d.u);
}

// Одна точка перебора компоновки. varlen=false -> одна последовательность, сетка (NT, B*H/HPB).
template <int WM, int WN, int HPB, int MINB, bool TRI>
struct Variant {
  static constexpr int LDA = BT + 4, LDB = DK + 16;
  using KT = void (*)(const __half*, const __half*, const __half*, const float*, const __half*,
                      __half*, __half*, const int*, const int*, int, int, int);
  static KT kern() { return fa2_sm70::gdn::volta_gdn_chunk_wu<BT, DK, DV, WM, WN, HPB, MINB, TRI, int>; }
  static size_t smem() { return (size_t)(BT * LDA + 2 * BT * LDB) * sizeof(__half); }
  static void launch(const Dev& d, const Tensors& t, bool varlen) {
    static bool once = true;
    if (once) { CK(cudaFuncSetAttribute(kern(), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem())); once = false; }
    const int NT = varlen ? d.NT : (t.T + BT - 1) / BT;
    dim3 grid(NT, t.H / HPB);
    kern()<<<grid, WM * WN * 32, smem()>>>(d.k, d.v, d.beta, d.g, d.A, d.w, d.u,
                                           varlen ? d.cu : nullptr, varlen ? d.ci : nullptr,
                                           t.T, t.H, t.Hg);
  }
  static int blocksPerSm() {
    int b = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, kern(), WM * WN * 32, smem()));
    return b;
  }
  static int regs() {
    cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, kern())); return a.numRegs;
  }
};

// ------------------------------------------------------------------------------------------------
// Сверка ВСЕХ выходных элементов против fp64-эталона.
// ------------------------------------------------------------------------------------------------
static double check(const Tensors& t, const Dev& d, const char* name) {
  std::vector<__half> hw((size_t)t.T * t.H * DK), hu((size_t)t.T * t.H * DV);
  CK(cudaMemcpy(hw.data(), d.w, hw.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hu.data(), d.u, hu.size() * 2, cudaMemcpyDeviceToHost));
  double emax = 0, rmax = 0;
  for (int n = 0; n < t.nseq; ++n) {
    const int L = t.cu[n + 1] - t.cu[n];
    for (int c = 0; c < (L + BT - 1) / BT; ++c) {
      const int r0 = t.cu[n] + c * BT, rows = std::min(BT, L - c * BT);
      for (int h = 0; h < t.H; ++h) {
        const int hk = h / (t.H / t.Hg);
        for (int i = 0; i < rows; ++i) {
          for (int dd = 0; dd < DV; ++dd) {
            double acc = 0;
            for (int j = 0; j <= i; ++j)                       // A[i][j] = 0 при j > i по построению
              acc += h2d(t.A[((size_t)(r0 + i) * t.H + h) * BT + j]) *
                     h2d(t.beta[(size_t)(r0 + j) * t.H + h]) *
                     h2d(t.v[((size_t)(r0 + j) * t.H + h) * DV + dd]);
            const double got = h2d(hu[((size_t)(r0 + i) * t.H + h) * DV + dd]);
            emax = std::max(emax, fabs(got - acc));
            rmax = std::max(rmax, fabs(acc));
          }
          for (int e = 0; e < DK; ++e) {
            double acc = 0;
            for (int j = 0; j <= i; ++j)
              acc += h2d(t.A[((size_t)(r0 + i) * t.H + h) * BT + j]) *
                     h2d(t.beta[(size_t)(r0 + j) * t.H + h]) *
                     exp((double)t.g[(size_t)(r0 + j) * t.H + h]) *
                     h2d(t.k[((size_t)(r0 + j) * t.Hg + hk) * DK + e]);
            const double got = h2d(hw[((size_t)(r0 + i) * t.H + h) * DK + e]);
            emax = std::max(emax, fabs(got - acc));
            rmax = std::max(rmax, fabs(acc));
          }
        }
      }
    }
  }
  const double rel = emax / (rmax + 1e-30);
  printf("  %-34s относительная ошибка %.2e   (макс.|эталон| %.4f)\n", name, rel, rmax);
  return rel;
}

// ------------------------------------------------------------------------------------------------
// Замер: прогрев, затем усечённое среднее по отдельным запускам.
// ------------------------------------------------------------------------------------------------
template <typename F>
static double bench(F&& launch) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < 30; ++i) launch();
  CK(cudaDeviceSynchronize());
  std::vector<float> ms;
  for (int i = 0; i < 60; ++i) {
    CK(cudaEventRecord(a));
    launch();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float t; CK(cudaEventElapsedTime(&t, a, b));
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  double s = 0; int n = 0;
  for (size_t i = ms.size() / 5; i < ms.size() - ms.size() / 5; ++i) { s += ms[i]; ++n; }
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return s / n;
}

int main(int argc, char** argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 15680;
  const int H = argc > 2 ? atoi(argv[2]) : 24;
  const int Hg = argc > 3 ? atoi(argv[3]) : 8;

  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("карта: %s, SM %d, такт %d МГц\n", p.name, p.multiProcessorCount, p.clockRate / 1000);

  using V = Variant<4, 2, 3, 2, true>;          // боевая точка: 3 головы значения на голову ключа
  printf("компоновка по умолчанию: 8 варпов 4x2, плитка варпа 16x64, голов на блок 3, "
         "разделяемой %.1f КБ, регистров %d, блоков/SM %d\n",
         V::smem() / 1024.0, V::regs(), V::blocksPerSm());

  // ---------------- 1. КОРРЕКТНОСТЬ ----------------
  printf("\n--- сверка с fp64-эталоном (A построена независимо и обращена в fp64) ---\n");
  bool ok = true;
  {   // ровный чанк, боевое отображение голов 24/8
    Tensors t; gen(t, 24, 8, {256}, true, 7);
    Dev d; upload(t, d);
    CK(cudaMemset(d.w, 0xff, (size_t)t.T * t.H * DK * 2));
    CK(cudaMemset(d.u, 0xff, (size_t)t.T * t.H * DV * 2));
    V::launch(d, t, false); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ok &= check(t, d, "T=256 H=24 Hg=8 HPB=3, ровно") < 1e-3;
    freeDev(d);
  }
  {   // хвостовой чанк неполной длины
    Tensors t; gen(t, 6, 2, {200}, true, 11);
    Dev d; upload(t, d);
    Variant<4, 2, 3, 2, true>::launch(d, t, false); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ok &= check(t, d, "T=200 H=6 Hg=2 HPB=3, хвост 8") < 1e-3;
    freeDev(d);
  }
  {   // varlen: две последовательности разной длины, обе с хвостом
    Tensors t; gen(t, 12, 4, {100, 173}, true, 13);
    Dev d; upload(t, d);
    Variant<4, 2, 3, 2, true>::launch(d, t, true); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ok &= check(t, d, "varlen 100+173, H=12 Hg=4 HPB=3") < 1e-3;
    freeDev(d);
  }
  {   // HPB=1 (блок = одна голова) и H == Hg -- вырожденное отображение голов
    Tensors t; gen(t, 4, 4, {130}, true, 17);
    Dev d; upload(t, d);
    Variant<2, 4, 1, 2, true>::launch(d, t, false); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ok &= check(t, d, "HPB=1, H=Hg=4, T=130") < 1e-3;
    freeDev(d);
  }
  {   // та же форма БЕЗ укорочения свёртки: проверяет, что треугольность не подменяет ответ
    Tensors t; gen(t, 6, 2, {200}, true, 11);
    Dev d; upload(t, d);
    Variant<2, 4, 3, 2, false>::launch(d, t, false); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ok &= check(t, d, "то же, полная свёртка (TRI=0)") < 1e-3;
    freeDev(d);
  }

  // ---------------- 2. ЗАМЕР НА БОЕВОЙ ФОРМЕ ----------------
  printf("\n--- замер, T=%d H=%d Hg=%d (NT=%d; блоков %d при HPB=3, %d при HPB=1) ---\n",
         T, H, Hg, (T + BT - 1) / BT, ((T + BT - 1) / BT) * H / 3, ((T + BT - 1) / BT) * H);
  Tensors t; gen(t, H, Hg, {T}, false, 3);
  Dev d; upload(t, d);
  const double NT = (T + BT - 1) / BT;
  const double flop = NT * H * 4.0 * BT * BT * DK;                 // плотно, два умножения
  // Трафик при общей плитке ключа: k читается ОДИН раз на голову ключа, beta/g -- один сектор на
  // группу голов значения (соседние головы лежат рядом в [T][H]).
  const double bytes = NT * (H * (double)(BT * BT * 2 + BT * DV * 2 + BT * DK * 2 + BT * DV * 2) +
                             Hg * (double)(BT * DK * 2) + H * (double)(BT * 6));

  struct Row { const char* name; double ms; int regs, bsm; };
  std::vector<Row> rows;
#define POINT(WMx, WNx, HPBx, MINBx, TRIx, NAME)                                                \
  do {                                                                                          \
    using Q = Variant<WMx, WNx, HPBx, MINBx, TRIx>;                                             \
    const double ms = bench([&] { Q::launch(d, t, false); });                                   \
    rows.push_back({NAME, ms, Q::regs(), Q::blocksPerSm()});                                    \
  } while (0)
  POINT(4, 2, 3, 2, true,  "4x2 HPB=3 TRI  MINB=2");
  POINT(4, 2, 3, 2, false, "4x2 HPB=3      MINB=2");
  POINT(2, 4, 3, 2, true,  "2x4 HPB=3 TRI  MINB=2");
  POINT(4, 2, 1, 2, true,  "4x2 HPB=1 TRI  MINB=2");
  POINT(2, 4, 1, 2, true,  "2x4 HPB=1 TRI  MINB=2");
  POINT(4, 2, 3, 3, true,  "4x2 HPB=3 TRI  MINB=3");
  POINT(1, 8, 3, 2, true,  "1x8 HPB=3 TRI  MINB=2");
#undef POINT

  printf("  %-24s %8s %10s %10s %6s %5s\n", "компоновка", "мс", "TFLOP/s", "ГБ/с", "рег", "бл/SM");
  for (auto& r : rows)
    printf("  %-24s %8.3f %10.2f %10.1f %6d %5d\n", r.name, r.ms,
           flop / (r.ms * 1e-3) / 1e12, bytes / (r.ms * 1e-3) / 1e9, r.regs, r.bsm);

  double best = 1e30;
  for (auto& r : rows) best = std::min(best, r.ms);
  printf("\nлучшее: %.3f мс, %.2f TFLOP/s плотно (%.2f полезно с учётом треугольности), %.1f ГБ/с\n",
         best, flop / (best * 1e-3) / 1e12, flop * 0.5079 / (best * 1e-3) / 1e12,
         bytes / (best * 1e-3) / 1e9);
  printf("сверка: %s\n", ok ? "ВСЁ В ПРЕДЕЛАХ fp16" : "ОТКАЗ");
  freeDev(d);
  return ok ? 0 : 1;
}
