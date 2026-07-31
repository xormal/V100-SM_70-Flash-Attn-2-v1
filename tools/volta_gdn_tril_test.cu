// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Проверка и замер ядра обращения единичной нижнетреугольной 64x64 (volta_gdn_tril.cuh).
//
// ЭТАЛОН СЧИТАЕТСЯ ПО ОПРЕДЕЛЕНИЮ В fp64 НА CPU -- прямой подстановкой по столбцам, а не вызовом
// другого ядра: иначе общая ошибка в понимании формулы прошла бы обе стороны сверки незамеченной.
// Сверяется ВЕСЬ тайл 64x64, включая строго верхний треугольник, который обязан быть ТОЧНЫМ нулём
// (это часть контракта с потребителем recompute_w_u_fwd, читающим плитку целиком).
//
// ВХОД ГЕНЕРИРУЕТСЯ ПО ТОЙ ЖЕ МОДЕЛИ, ЧТО В БОЮ, а не случайными числами: A[i][j] =
// beta_i * (k_i . k_j) * exp(g_i - g_j) при i>j, где k l2-нормированы, beta = sigmoid(.) в (0,1),
// g -- внутричанковая кумсумма неположительных величин. Отсюда |A|<=1. Корреляция ключей rho идёт
// по шаблонам от 0 до 0.995: при rho->1 обусловленность растёт, и именно там fp16-операнд мог бы
// проиграть -- на случайном входе этого бы не увидели.
//
// ПОКРЫТИЕ: боевая форма (T=15680, H=24, ровно 245 полных чанков) И отдельный varlen-случай с
// четырьмя последовательностями некратной длины -- иначе хвостовая ветка и разбор chunk_indices
// не проверяются вовсе, а в бою они живые.
//
// Сборка:
//   nvcc -arch=sm_70 -O3 -std=c++17 -I fa2_src/fmha_kernel/gemm \
//        -o /tmp/tril tools/volta_gdn_tril_test.cu
// Запуск: ./tril [T] [H] [dump_prefix]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include "../fa2_sm70/csrc/volta_gdn_tril.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ОТКАЗ %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static constexpr int BT = 64;
static constexpr int DK = 128;      // размерность ключа, только для генерации входа
static constexpr int NTPL = 64;     // число различных шаблонов 64x64

// --- модель входа: та же алгебра, что в chunk_scaled_dot_kkt_fwd
static void make_template(int p, std::vector<float>& A) {
  std::mt19937 rng(1234u + 7919u * p);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::uniform_real_distribution<double> ud(0.0, 1.0);
  const double rho = (p < NTPL / 2) ? (0.05 * p / (NTPL / 2.0))
                                    : (0.995 * (p - NTPL / 2) / (NTPL / 2.0 - 1));
  std::vector<double> u(DK), k((size_t)BT * DK), beta(BT), g(BT);
  double nu = 0; for (int d = 0; d < DK; ++d) { u[d] = nd(rng); nu += u[d] * u[d]; }
  nu = sqrt(nu); for (int d = 0; d < DK; ++d) u[d] /= nu;
  for (int i = 0; i < BT; ++i) {
    double n2 = 0;
    for (int d = 0; d < DK; ++d) {
      const double v = sqrt(rho) * u[d] + sqrt(1.0 - rho) * nd(rng);
      k[(size_t)i * DK + d] = v; n2 += v * v;
    }
    n2 = sqrt(n2);
    for (int d = 0; d < DK; ++d) k[(size_t)i * DK + d] /= n2;
    beta[i] = 1.0 / (1.0 + exp(-nd(rng)));                                    // sigmoid
    const double dt = -exp(nd(rng) * 0.5) * log1p(exp(ud(rng) * 2.0 - 1.0));  // <= 0
    g[i] = (i ? g[i - 1] : 0.0) + dt;                                         // кумсумма
  }
  A.assign((size_t)BT * BT, 0.f);
  for (int i = 0; i < BT; ++i)
    for (int j = 0; j < i; ++j) {
      double dot = 0;
      for (int d = 0; d < DK; ++d) dot += k[(size_t)i * DK + d] * k[(size_t)j * DK + d];
      A[(size_t)i * BT + j] = (float)(beta[i] * dot * exp(g[i] - g[j]));
    }
}

// --- ЭТАЛОН: (I + A)^{-1} прямой подстановкой в fp64. Диагональ ровно 1, деления нет.
// У нижнетреугольной матрицы ведущий блок обратной равен обратной ведущего блока -- поэтому один
// эталон 64x64 годится и для хвостового чанка длины nr (берётся его левый верхний угол).
static void ref_inverse(const float* A, int n, std::vector<double>& X) {
  X.assign((size_t)n * n, 0.0);
  for (int j = 0; j < n; ++j) {
    X[(size_t)j * n + j] = 1.0;
    for (int i = j + 1; i < n; ++i) {
      double s = 0;
      for (int k = j; k < i; ++k) s += (double)A[(size_t)i * n + k] * X[(size_t)k * n + j];
      X[(size_t)i * n + j] = -s;
    }
  }
}

struct Chunk { int row0; int nr; };          // глобальная строка начала чанка и число валидных строк
struct Res { double ms; double errRel; double errAbs; double refMax; long badZero; long badWrite; };

static std::vector<std::vector<float>> g_tpl;
static std::vector<std::vector<double>> g_ref;

template <int DIAG>
static Res run(const float* dA, __half* dAi, const int* dCi, const int* dCu, int T, int H,
               const std::vector<Chunk>& ch, std::vector<__half>& out, int reps, bool quiet) {
  auto kern = fa2_sm70::gdn::volta_gdn_solve_tril<BT, DIAG>;
  const size_t sh = fa2_sm70::gdn::volta_gdn_solve_tril_smem<BT, DIAG>();
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  const int NT = (int)ch.size();
  const dim3 grid(NT, H);
  // 0xff = -NaN в fp16: любая непрописанная позиция станет видна и как badWrite, и как ошибка
  CK(cudaMemset(dAi, 0xff, (size_t)T * H * BT * sizeof(__half)));
  kern<<<grid, 128, sh>>>(dA, dAi, dCi, dCu, T, H);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), dAi, out.size() * sizeof(__half), cudaMemcpyDeviceToHost));

  Res r{0, 0, 0, 0, 0, 0};
  const int step = NT > 64 ? 17 : 1;                       // на мелких формах проверяем ВСЁ
  for (int c = 0; c < NT; ++c) {
    if (c % step && c != NT - 1) continue;
    for (int h = 0; h < H; ++h) {
      const std::vector<double>& R = g_ref[(c * 24 + h) % NTPL];
      for (int i = 0; i < ch[c].nr; ++i)
        for (int j = 0; j < BT; ++j) {
          const int nr = ch[c].nr;
          const double want = (j < nr) ? R[(size_t)i * BT + j] : (i == j ? 1.0 : 0.0);
          const double got = __half2float(out[((size_t)(ch[c].row0 + i) * H + h) * BT + j]);
          if (!(got == got)) { r.badWrite++; continue; }   // NaN = позицию не записали
          if (j > i && got != 0.0) r.badZero++;
          const double e = fabs(got - want);
          if (e > r.errAbs) r.errAbs = e;
          if (fabs(want) > r.refMax) r.refMax = fabs(want);
        }
    }
  }
  r.errRel = r.errAbs / (r.refMax + 1e-30);

  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < 10; ++i) kern<<<grid, 128, sh>>>(dA, dAi, dCi, dCu, T, H);
  CK(cudaDeviceSynchronize());
  std::vector<double> ts;
  for (int i = 0; i < reps; ++i) {
    CK(cudaEventRecord(a));
    kern<<<grid, 128, sh>>>(dA, dAi, dCi, dCu, T, H);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  const int lo = reps / 5, hi = reps - reps / 5;
  double s = 0; for (int i = lo; i < hi; ++i) s += ts[i];
  r.ms = s / (hi - lo);
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));

  if (!quiet) {
    cudaFuncAttributes at; CK(cudaFuncGetAttributes(&at, kern));
    printf("   регистров %d, локальной %zu Б, разделяемой %.1f КБ -> блоков на SM по разделяемой %d\n",
           at.numRegs, (size_t)at.localSizeBytes, sh / 1024.0, (int)(96 * 1024 / sh));
  }
  return r;
}

// Один прогон формы. seqs пустой -> одна последовательность длины T без cu_seqlens (боевой путь).
static void run_case(const char* name, std::vector<int> seqs, int H, bool varlen, const char* dump) {
  int T = 0; for (int L : seqs) T += L;
  std::vector<Chunk> ch;
  std::vector<int> hCi, hCu(1, 0);
  { int bos = 0;
    for (size_t n = 0; n < seqs.size(); ++n) {
      const int L = seqs[n], nt = (L + BT - 1) / BT;
      for (int t = 0; t < nt; ++t) {
        ch.push_back({bos + t * BT, std::min(BT, L - t * BT)});
        hCi.push_back((int)n); hCi.push_back(t);
      }
      bos += L; hCu.push_back(bos);
    } }
  const int NT = (int)ch.size();
  printf("\n=== %s: T=%d H=%d, последовательностей %zu, чанков %d, сетка %d блоков по 128 нитей ===\n",
         name, T, H, seqs.size(), NT, NT * H);

  std::vector<float> hA((size_t)T * H * BT, 0.f);
  for (int c = 0; c < NT; ++c)
    for (int h = 0; h < H; ++h) {
      const std::vector<float>& S = g_tpl[(c * 24 + h) % NTPL];
      for (int i = 0; i < ch[c].nr; ++i)
        for (int j = 0; j < ch[c].nr; ++j)
          hA[((size_t)(ch[c].row0 + i) * H + h) * BT + j] = S[(size_t)i * BT + j];
    }

  float* dA; __half* dAi; int *dCi = nullptr, *dCu = nullptr;
  CK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CK(cudaMalloc(&dAi, hA.size() * sizeof(__half)));
  CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  if (varlen) {
    CK(cudaMalloc(&dCi, hCi.size() * 4)); CK(cudaMalloc(&dCu, hCu.size() * 4));
    CK(cudaMemcpy(dCi, hCi.data(), hCi.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCu, hCu.data(), hCu.size() * 4, cudaMemcpyHostToDevice));
  }
  std::vector<__half> out(hA.size());

  const double macTriton = (double)NT * H * 79872.0;   // счёт Triton: 4*14*256 + 16 умножений 16^3
  const double bytes = (double)NT * H * (10240.0 + 8192.0);
  const char* nm[3] = {"диагональ на HMMA (возведение в квадрат)",
                       "диагональ строчной рекурсией fp32, вход из ГЛОБАЛЬНОЙ fp32",
                       "диагональ строчной рекурсией fp32, вход из РАЗДЕЛЯЕМОЙ fp16  <- по умолчанию"};
  Res R[3];
  R[0] = run<0>(dA, dAi, dCi, dCu, T, H, ch, out, 50, false);
  if (dump) { char f[512]; snprintf(f, sizeof f, "%s_mma.bin", dump);
              FILE* fp = fopen(f, "wb"); fwrite(out.data(), 2, out.size(), fp); fclose(fp); }
  R[1] = run<1>(dA, dAi, dCi, dCu, T, H, ch, out, 50, false);
  if (dump) { char f[512]; snprintf(f, sizeof f, "%s_fp32.bin", dump);
              FILE* fp = fopen(f, "wb"); fwrite(out.data(), 2, out.size(), fp); fclose(fp); }
  R[2] = run<2>(dA, dAi, dCi, dCu, T, H, ch, out, 50, false);
  if (dump) { char f[512]; snprintf(f, sizeof f, "%s_smem.bin", dump);
              FILE* fp = fopen(f, "wb"); fwrite(out.data(), 2, out.size(), fp); fclose(fp);
              snprintf(f, sizeof f, "%s_A.bin", dump);
              fp = fopen(f, "wb"); fwrite(hA.data(), 4, hA.size(), fp); fclose(fp); }

  for (int m = 0; m < 3; ++m)
    printf("%-62s %.4f мс | отн.ошибка %.2e | верх!=0: %ld | не записано: %ld | %.2f TFLOP/s | %.0f ГБ/с\n",
           nm[m], R[m].ms, R[m].errRel, R[m].badZero, R[m].badWrite,
           2 * macTriton / (R[m].ms * 1e-3) / 1e12, bytes / (R[m].ms * 1e-3) / 1e9);
  printf("пол по трафику этой формы: %.0f МБ / 819 ГБ/с = %.4f мс\n", bytes / 1e6, bytes / 819e9 * 1e3);

  CK(cudaFree(dA)); CK(cudaFree(dAi));
  if (varlen) { CK(cudaFree(dCi)); CK(cudaFree(dCu)); }
}

int main(int argc, char** argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 15680;
  const int H = argc > 2 ? atoi(argv[2]) : 24;
  const char* dump = argc > 3 ? argv[3] : nullptr;

  g_tpl.resize(NTPL); g_ref.resize(NTPL);
  double refMax = 0;
  for (int p = 0; p < NTPL; ++p) {
    make_template(p, g_tpl[p]);
    ref_inverse(g_tpl[p].data(), BT, g_ref[p]);
    for (double v : g_ref[p]) refMax = std::max(refMax, fabs(v));
  }
  printf("вход: %d шаблонов (rho 0..0.995), max|Ai| по fp64-эталону = %.3f\n", NTPL, refMax);

  run_case("БОЕВАЯ ФОРМА (одна последовательность, без cu_seqlens)", {T}, H, false, dump);
  run_case("ХВОСТЫ И VARLEN (длины не кратны 64, разбор chunk_indices)",
           {88, 200, 64, 377, 15}, 5, true, nullptr);
  return 0;
}
