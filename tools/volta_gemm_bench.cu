// Стенд полного конвейера рукописного мейнлупа Volta, и одновременно — честный шлюз перед тем, как
// встраивать его во что-либо. Меряет то, что нельзя померить внутри FMHA: сквозную пропускную
// способность связки глобальная -> разделяемая (двойная буферизация) -> регистры -> HMMA на форме,
// у которой есть общепризнанный эталон (cuBLAS на той же матрице).
//
// Сборка и запуск (V100, такты зафиксированы):
//   nvcc -O3 -std=c++17 -arch=sm_70 -I ../fa2_src/fmha_kernel/gemm -lcublas -o gemm volta_gemm_bench.cu
//   LD_LIBRARY_PATH=$CUDA_HOME/lib ./gemm 0
//
// ЗАЧЕМ ЭТО НУЖНО БЫЛО. Первая версия мейнлупа считала ЧЕТВЕРТЬ своей плитки и проходила сверку с
// torch — сверка сравнивала ровно те ячейки, которые вычислялись. Ошибку показал ЗАМЕР СКОРОСТИ, а не
// проверка значений. С тех пор порядок такой: покрытие проверяется отдельным утверждением
// (tests/test_volta_hmma.py), а скорость — здесь, ДО встраивания.
//
// ИЗМЕРЕНО (V100 @ 1530 МГц, пик 125.3 ТФЛОП/с; C = A*B^T, fp16 -> fp32):
//   потолок мейнлупа (данные в smem):      120.96 ТФЛОП/с = 96.8% пика  (плитка варпа 64x64)
//   полный конвейер против cuBLAS:
//       4096^3   73.6 против 96.6   x0.76      1024x1024x8192  49.1 против 73.6   x0.67
//       2048^2 x4096  62.8 против 76.5  x0.82  8192^2 x1024    56.9 против 82.0   x0.69
//       4096x1024x2048 57.4 против 55.3 x1.04  512^2 x16384    12.8 против 46.3   x0.28
//   Узкое место конвейера НЕ в математике: ncu даёт тензорный конвейер на 61% при DRAM всего
//   152 ГБ/с и попадании в L2 90.3%. Разбор простоев (тактов на выданную инструкцию, всего 4.66):
//   long_scoreboard 0.95, wait 0.92, mio_throttle 0.53, barrier 0.24. То есть задержка глобальных
//   загрузок не покрыта и очередь памяти забита — а не тензорные ядра голодают.
//   Малые формы (512^2) проседают из-за заполнения: 16 блоков на 80 SM, cuBLAS там делит по K.
//
// ЧТО ИЗ ЭТОГО ПОЧИНЕНО В ЗАГОЛОВКЕ (и подтверждено в опкодах):
//   * дополнение строк LD=BK+4 даёт бесконфликтные 8-байтовые доступы, но ЗАПРЕЩАЕТ 16-байтовые
//     (нечётные строки выровнены лишь на 8) -> XOR-перестановка чанков: вдвое меньше LDS и STS;
//   * "+f" вместо отдельных C и D: минус 8 копий регистров на каждую выдачу;
//   * обычный asm вместо asm volatile: иначе ptxas не имеет права переставлять загрузки под HMMA.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "volta_hmma.h"

using namespace fa2_sm70;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define CB(x) do { cublasStatus_t e=(x); if(e){printf("cuBLAS %s @%d: %d\n",#x,__LINE__,(int)e);exit(1);} } while(0)

// ---- одна 16-байтовая порция глобальной памяти на поток, разложенная по строкам плитки
template <int ROWS, int BK, int NTHR>
struct TileLoader {
  static constexpr int kChunks = ROWS * BK / 8;          // порций по 8 половинок
  static constexpr int kPer = kChunks / NTHR;            // порций на поток
  static constexpr int kPerRow = BK / 8;
  uint4 buf[kPer];
  int row[kPer], col[kPer];
  __device__ __forceinline__ void init(int tid) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) { const int idx = tid + i * NTHR; row[i] = idx / kPerRow; col[i] = (idx % kPerRow) * 8; }
  }
  __device__ __forceinline__ void ldg(const __half* g, int ld, int k0) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) buf[i] = *reinterpret_cast<const uint4*>(g + (long)row[i] * ld + k0 + col[i]);
  }
  // Разделяемая память дополнена до LD = BK+4 половинок (72 Б при BK=32): строки попадают на разные
  // банки, но 16-байтовое выравнивание теряется для нечётных строк -> храним двумя STS.64.
  // XOR-перестановка: одна STS.128 вместо двух STS.64, без дополнения строк.
  template <int BK2>
  __device__ __forceinline__ void sts_sw(__half* s) const {
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      *reinterpret_cast<uint4*>(s + SwizzleTile<BK2>::off(row[i], col[i] / 8)) = buf[i];
  }
  __device__ __forceinline__ void sts(__half* s, int LD) const {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      __half* p = s + (long)row[i] * LD + col[i];
      *reinterpret_cast<uint2*>(p) = make_uint2(buf[i].x, buf[i].y);
      *reinterpret_cast<uint2*>(p + 4) = make_uint2(buf[i].z, buf[i].w);
    }
  }
};

// Переупорядочение блоков под L2: подряд идущие блоки образуют компактный столбец GROUP x всё-N,
// поэтому одновременно исполняемые блоки делят одни и те же плитки B (и соседние — плитки A).
// Линейный порядок blockIdx даёт полосу шириной во всю матрицу: к моменту возврата к той же плитке
// B она давно вытеснена из L2, и каждый блок читает её из HBM заново.
template <int GROUP>
__device__ __forceinline__ void swizzle(int bid, int nm, int nn, int& bm, int& bn) {
  const int per = GROUP * nn;
  const int grp = bid / per, first = grp * GROUP;
  const int gs = min(GROUP, nm - first);
  bm = first + (bid % per) % gs;
  bn = ((bid % per) / gs);
}

template <int BM, int BN, int BK, int MB, int NB, int WM, int WN>
__global__ __launch_bounds__(WM* WN * 32) void k_gemm(const __half* __restrict__ A,
                                                      const __half* __restrict__ B,
                                                      float* __restrict__ C, int M, int N, int K) {
  constexpr int NTHR = WM * WN * 32, LD = BK + 4;
  extern __shared__ __half smem[];
  __half* sA = smem;                       // [2][BM][LD]
  __half* sB = smem + 2 * BM * LD;         // [2][BN][LD]

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int bmi, bni; swizzle<8>(blockIdx.x, M / BM, N / BN, bmi, bni);
  const int bm = bmi * BM, bn = bni * BN;
  TileLoader<BM, BK, NTHR> la; la.init(tid);
  TileLoader<BN, BK, NTHR> lb; lb.init(tid);
  const __half* gA = A + (long)(bm + 0) * K;
  const __half* gB = B + (long)(bn + 0) * K;

  la.ldg(gA, K, 0); lb.ldg(gB, K, 0);
  la.sts(sA, LD);   lb.sts(sB, LD);
  __syncthreads();
  if (K > BK) { la.ldg(gA, K, BK); lb.ldg(gB, K, BK); }

  const int wm = warp / WN, wn = warp % WN;
  float acc[MB][NB][8];
  hmma_clear<MB, NB>(acc);
  uint32_t fa[2][MB][2], fb[2][NB][2];
  int cur = 0;
  const __half* pA = sA + (wm * MB * 16 + hmma_a_row(lane)) * LD;
  const __half* pB = sB + (wn * NB * 16 + hmma_b_col(lane)) * LD;
  hmma_ld_a(fa[0], pA, LD, 0);
  hmma_ld_b(fb[0], pB, LD, 0);

  const int NT = K / BK;
  for (int kt = 0; kt < NT; ++kt) {
#pragma unroll
    for (int k = 0; k < BK; k += 4) {
      const int f = (k >> 2) & 1;
      if (k + 4 < BK) {
        hmma_ld_a(fa[f ^ 1], pA + cur * BM * LD, LD, k + 4);
        hmma_ld_b(fb[f ^ 1], pB + cur * BN * LD, LD, k + 4);
      }
      hmma_k4<MB, NB>(acc, fa[f], fb[f]);
    }
    if (kt + 1 < NT) {
      la.sts(sA + (cur ^ 1) * BM * LD, LD);
      lb.sts(sB + (cur ^ 1) * BN * LD, LD);
      __syncthreads();
      if (kt + 2 < NT) { la.ldg(gA, K, (kt + 2) * BK); lb.ldg(gB, K, (kt + 2) * BK); }
      cur ^= 1;
      hmma_ld_a(fa[0], pA + cur * BM * LD, LD, 0);
      hmma_ld_b(fb[0], pB + cur * BN * LD, LD, 0);
    }
  }

  const int r0 = bm + wm * MB * 16, c0 = bn + wn * NB * 16;
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = r0 + i * 16 + hmma_acc_row(lane, r), n = c0 + j * 16 + hmma_acc_col(lane, r);
        if (m < M && n < N) C[(long)m * N + n] = acc[i][j][r];
      }
}

template <int BM, int BN, int BK, int MB, int NB, int WM, int WN>
double run(const __half* A, const __half* B, float* C, int M, int N, int K, int iters, int* regs) {
  constexpr int LD = BK + 4;
  const size_t sh = (size_t)2 * (BM + BN) * LD * sizeof(__half);
  auto kern = k_gemm<BM, BN, BK, MB, NB, WM, WN>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  cudaFuncAttributes at; cudaFuncGetAttributes(&at, (const void*)kern); *regs = at.numRegs;
  dim3 g((M / BM) * (N / BN)), b(WM * WN * 32);
  kern<<<g, b, sh>>>(A, B, C, M, N, K); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 3; ++i) kern<<<g, b, sh>>>(A, B, C, M, N, K);
  CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) kern<<<g, b, sh>>>(A, B, C, M, N, K);
  cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / iters;
}


template <int BM, int BN, int BK, int MB, int NB, int WM, int WN>
__global__ __launch_bounds__(WM* WN * 32) void k_gemm_sw(const __half* __restrict__ A,
                                                         const __half* __restrict__ B,
                                                         float* __restrict__ C, int M, int N, int K) {
  constexpr int NTHR = WM * WN * 32;
  extern __shared__ __half smem[];
  __half* sA = smem;                       // [2][BM][BK]
  __half* sB = smem + 2 * BM * BK;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int bmi, bni; swizzle<8>(blockIdx.x, M / BM, N / BN, bmi, bni);
  const int bm = bmi * BM, bn = bni * BN;
  TileLoader<BM, BK, NTHR> la; la.init(tid);
  TileLoader<BN, BK, NTHR> lb; lb.init(tid);
  const __half* gA = A + (long)bm * K;
  const __half* gB = B + (long)bn * K;
  la.ldg(gA, K, 0); lb.ldg(gB, K, 0);
  la.template sts_sw<BK>(sA); lb.template sts_sw<BK>(sB);
  __syncthreads();
  if (K > BK) { la.ldg(gA, K, BK); lb.ldg(gB, K, BK); }
  const int wm = warp / WN, wn = warp % WN;
  const int rowA = wm * MB * 16 + hmma_a_row(lane), rowB = wn * NB * 16 + hmma_b_col(lane);
  float acc[MB][NB][8];
  hmma_clear<MB, NB>(acc);
  int cur = 0;
  const int NT = K / BK;
  for (int kt = 0; kt < NT; ++kt) {
    hmma_warp_tile_sw<MB, NB, BK>(acc, sA + cur * BM * BK, rowA, sB + cur * BN * BK, rowB);
    if (kt + 1 < NT) {
      la.template sts_sw<BK>(sA + (cur ^ 1) * BM * BK);
      lb.template sts_sw<BK>(sB + (cur ^ 1) * BN * BK);
      __syncthreads();
      if (kt + 2 < NT) { la.ldg(gA, K, (kt + 2) * BK); lb.ldg(gB, K, (kt + 2) * BK); }
      cur ^= 1;
    }
  }
  const int r0 = bm + wm * MB * 16, c0 = bn + wn * NB * 16;
#pragma unroll
  for (int i = 0; i < MB; ++i)
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = r0 + i * 16 + hmma_acc_row(lane, r), n = c0 + j * 16 + hmma_acc_col(lane, r);
        if (m < M && n < N) C[(long)m * N + n] = acc[i][j][r];
      }
}

template <int BM, int BN, int BK, int MB, int NB, int WM, int WN>
double run_sw(const __half* A, const __half* B, float* C, int M, int N, int K, int iters, int* regs) {
  const size_t sh = (size_t)2 * (BM + BN) * BK * sizeof(__half);
  auto kern = k_gemm_sw<BM, BN, BK, MB, NB, WM, WN>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  cudaFuncAttributes at; cudaFuncGetAttributes(&at, (const void*)kern); *regs = at.numRegs;
  dim3 g((M / BM) * (N / BN)), b(WM * WN * 32);
  kern<<<g, b, sh>>>(A, B, C, M, N, K); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 3; ++i) kern<<<g, b, sh>>>(A, B, C, M, N, K);
  CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) kern<<<g, b, sh>>>(A, B, C, M, N, K);
  cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / iters;
}

int main(int argc, char** argv) {
  CK(cudaSetDevice(argc > 1 ? atoi(argv[1]) : 0));
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));

  // ---- корректность на маленькой форме
  {
    const int M = 256, N = 128, K = 128;
    std::vector<__half> hA(M * K), hB(N * K);
    srand(3);
    for (auto& x : hA) x = __float2half((rand() / (float)RAND_MAX - .5f) * 2.f);
    for (auto& x : hB) x = __float2half((rand() / (float)RAND_MAX - .5f) * 2.f);
    __half *dA, *dB; float* dC;
    CK(cudaMalloc(&dA, hA.size() * 2)); CK(cudaMalloc(&dB, hB.size() * 2)); CK(cudaMalloc(&dC, (size_t)M * N * 4));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0x7f, (size_t)M * N * 4));
    int rg; run_sw<128, 128, 32, 2, 4, 4, 2>(dA, dB, dC, M, N, K, 1, &rg);
    std::vector<float> hC(M * N); CK(cudaMemcpy(hC.data(), dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    double emax = 0; int nan = 0;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double ref = 0;
        for (int k = 0; k < K; ++k) ref += (double)__half2float(hA[m * K + k]) * (double)__half2float(hB[n * K + k]);
        const float got = hC[(size_t)m * N + n];
        if (!isfinite(got)) { ++nan; continue; }
        emax = fmax(emax, fabs(got - ref) / fmax(fabs(ref), 1e-2));
      }
    printf("=== корректность 256x128x128: не записано/NaN %d, макс. относит. ошибка %.3e -> %s\n",
           nan, emax, (nan == 0 && emax < 3e-3) ? "ВЕРНО" : "ОТКАЗ");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }

  // ---- скорость против cuBLAS
  printf("\n=== перебор конфигураций (BMxBN/варпов)\n");
  struct Shape { int M, N, K; };
  const Shape shapes[] = {{4096, 4096, 4096}, {2048, 2048, 4096}, {1024, 1024, 8192},
                          {4096, 1024, 2048}, {512, 512, 16384}, {8192, 8192, 1024}};
  for (Shape s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    __half *dA, *dB; float* dC;
    CK(cudaMalloc(&dA, (size_t)M * K * 2)); CK(cudaMalloc(&dB, (size_t)N * K * 2)); CK(cudaMalloc(&dC, (size_t)M * N * 4));
    CK(cudaMemset(dA, 0x11, (size_t)M * K * 2)); CK(cudaMemset(dB, 0x11, (size_t)N * K * 2));
    const double flop = 2.0 * M * N * K;
    struct Res { const char* tag; double tf; int rg; };
    std::vector<Res> res;
    int rg = 0;
#define TRY(tag, BM, BN, BK, MB, NB, WM, WN)                                              \
    if (M % (BM) == 0 && N % (BN) == 0) {                                                 \
      const double t = run<BM, BN, BK, MB, NB, WM, WN>(dA, dB, dC, M, N, K, 20, &rg);     \
      res.push_back({tag, flop / (t * 1e-3) / 1e12, rg});                                 \
    }
#define TRYSW(tag, BM, BN, BK, MB, NB, WM, WN)                                            \
    if (M % (BM) == 0 && N % (BN) == 0) {                                                 \
      const double t = run_sw<BM, BN, BK, MB, NB, WM, WN>(dA, dB, dC, M, N, K, 20, &rg);  \
      res.push_back({tag, flop / (t * 1e-3) / 1e12, rg});                                 \
    }
    TRYSW("св128x128/4в", 128, 128, 32, 4, 4, 2, 2)
    TRYSW("св128x128/8в", 128, 128, 32, 2, 4, 4, 2)
    TRYSW("св256x128/8в", 256, 128, 32, 4, 4, 4, 2)
    TRYSW("св128x128/4в/K64", 128, 128, 64, 4, 4, 2, 2)
    TRY("128x128/8в", 128, 128, 32, 2, 4, 4, 2)
    TRY("128x128/4в", 128, 128, 32, 4, 4, 2, 2)
    TRY("256x128/8в", 256, 128, 32, 4, 4, 4, 2)
    TRY("256x128/16в", 256, 128, 32, 2, 4, 8, 2)
    TRY("128x256/16в", 128, 256, 32, 2, 4, 4, 4)
    TRY("256x256/16в", 256, 256, 32, 4, 4, 4, 4)
#undef TRY
    double best = 0; const char* btag = "";
    for (auto& r : res) if (r.tf > best) { best = r.tf; btag = r.tag; }
    const double ms = flop / (best * 1e12) * 1e3;
    // cuBLAS: C_cm[N][M] = B^T_cm * A_cm  (см. вывод в комментарии к main)
    const float al = 1.f, be = 0.f;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto cb = [&] { CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al, dB, CUDA_R_16F, K,
                                    dA, CUDA_R_16F, K, &be, dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP)); };
    cb(); CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) cb();
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float cms; cudaEventElapsedTime(&cms, e0, e1); cms /= 20;
    printf("%6d %5d %6d |", M, N, K);
    for (auto& r : res) printf(" %s %5.1f |", r.tag, r.tf);
    printf("  ЛУЧШИЙ %s %5.1f  cuBLAS %5.1f  x%.3f\n", btag, best, flop / (cms * 1e-3) / 1e12,
           best / (flop / (cms * 1e-3) / 1e12));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }
  cublasDestroy(h);
  return 0;
}
