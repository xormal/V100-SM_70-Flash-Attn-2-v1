// Проверка и замер рукописного варпового оператора (fa2_src/fmha_kernel/gemm/volta_warp_mma.h).
//
// Сборка и запуск (V100, такты зафиксированы):
//   nvcc -O3 -std=c++17 -arch=sm_70 -I ../fa2_src/fmha_kernel/gemm -o warp volta_warp_test.cu && ./warp
//
// ЧТО ПРОВЕРЯЕТСЯ
//   1. ПОКРЫТИЕ (а не только значения) на обоих вариантах инструкции и на формах, которые cutlass
//      отвергает: 32x16 и 64x16. Каждая ячейка помечается атомарным счётчиком — тест падает, если
//      хоть одна записана ноль или два раза. Сверка значений покрытие НЕ проверяет: она сравнивает
//      ровно те ячейки, которые вычисляются, и именно так однажды прошла версия, считавшая четверть.
//   2. Цепочка в форме FMHA в одном ядре: S = K*Q^T (.row.col) -> P = exp(S*scale - lse) -> dV = P^T*dO
//      (.row.row, dO читается КАК ЕСТЬ, без транспонирования через разделяемую память).
//   3. Потолок: окупает ли удвоение варпов потерю ILP на узкой плитке. Это и есть вопрос, ради
//      которого оператор писался.
//
// ИЗМЕРЕНО (V100 @ 1530 МГц, пик 125.3 ТФЛОП/с, 240 блоков):
//     плитка 32x32 / 8 варпов  (форма cutlass)   102.3 ТФЛОП/с  81.6%   60 регистров
//     плитка 64x64 / 8 варпов                    119.2          95.2%  190
//     плитка 32x16 / 16 варпов (цель)            115.1          91.9%   68
//     плитка 64x32 / 16 варпов                   112.2          89.5%  114
// Узкая плитка при 16 варпах ОБГОНЯЕТ cutlass-форму на 12.5%, отстаёт от самой широкой на 3.5% и
// стоит 68 регистров вместо 190. В backward при d>128 занятость держат именно регистры (и smem),
// см. docs/VOLTA_SM70.md §6b, поэтому важна не только скорость, но и цена в регистрах.
//
// NB о самом стенде: без дополнения строк (ld = K = 32) строки, отличающиеся на 2, попадают в один
// банк — 8-кратный конфликт, и первый прогон сравнивал конфликты, а не плитки (32x32/8в читалось как
// 41% пика вместо 81.6%). Дополнение до K+4 обязательно.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include "volta_warp_mma.h"

using namespace fa2_sm70;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

// ---------------------------------------------------------------- часть 1: плитка
template <int MB, int NB, BLayout BL, int K>
__global__ void k_tile(const __half* A, const __half* B, float* D, int* hits) {
    constexpr int M = MB * 16, N = NB * 16;
    constexpr int LDB = (BL == BLayout::kKMajor) ? K : N;
    constexpr int NB_ROWS = (BL == BLayout::kKMajor) ? N : K;
    __shared__ __half sA[M * K], sB[NB_ROWS * LDB];
    const int lane = threadIdx.x;
    for (int i = lane; i < M * K; i += 32) sA[i] = A[i];
    for (int i = lane; i < NB_ROWS * LDB; i += 32) sB[i] = B[i];
    __syncwarp();
    WarpMma<MB, NB, BL> op;
    op.clear();
    op.template accumulate<K>(sA, K, 0, sB, LDB, 0, lane);
    op.visit(lane, [&](float v, int r, int c) { D[r * N + c] = v; atomicAdd(&hits[r * N + c], 1); });
}

template <int MB, int NB, BLayout BL, int K>
void tile_case(const char* tag, int& fails) {
    constexpr int M = MB * 16, N = NB * 16;
    constexpr int NB_ROWS = (BL == BLayout::kKMajor) ? N : K;
    constexpr int LDB = (BL == BLayout::kKMajor) ? K : N;
    std::vector<__half> hA(M * K), hB(NB_ROWS * LDB);
    for (auto& x : hA) x = __float2half((rand() / (float)RAND_MAX - .5f) * 1.2f);
    for (auto& x : hB) x = __float2half((rand() / (float)RAND_MAX - .5f) * 1.2f);
    __half *dA, *dB; float* dD; int* dH;
    CK(cudaMalloc(&dA, hA.size() * 2)); CK(cudaMalloc(&dB, hB.size() * 2));
    CK(cudaMalloc(&dD, M * N * 4)); CK(cudaMalloc(&dH, M * N * 4));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dD, 0x7f, M * N * 4)); CK(cudaMemset(dH, 0, M * N * 4));
    k_tile<MB, NB, BL, K><<<1, 32>>>(dA, dB, dD, dH);
    CK(cudaDeviceSynchronize());
    std::vector<float> hD(M * N); std::vector<int> hH(M * N);
    CK(cudaMemcpy(hD.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hH.data(), dH, M * N * 4, cudaMemcpyDeviceToHost));
    int miss = 0, dup = 0; double emax = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            if (hH[m * N + n] == 0) ++miss;
            if (hH[m * N + n] > 1) ++dup;
            double ref = 0;
            for (int k = 0; k < K; ++k) {
                float b = (BL == BLayout::kKMajor) ? __half2float(hB[n * K + k])
                                                   : __half2float(hB[k * N + n]);
                ref += (double)__half2float(hA[m * K + k]) * b;
            }
            emax = fmax(emax, fabs(hD[m * N + n] - ref) / fmax(fabs(ref), 1e-2));
        }
    const bool ok = (miss == 0 && dup == 0 && emax < 3e-3);
    printf("  %-30s %3dx%-3d K=%-3d  пропущено %3d  дважды %3d  ошибка %.2e  %s\n",
           tag, M, N, K, miss, dup, emax, ok ? "ОК" : "ОТКАЗ");
    if (!ok) ++fails;
    cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(dH);
}

// ---------------------------------------------------------------- часть 2: цепочка FMHA
// S[key][query] = K*Q^T (оба k-непрерывны -> .row.col); P = exp(S*scale - lse[query]);
// dV[key][d] = P^T*dO, где dO[query][d] читается КАК ЕСТЬ (-> .row.row, без транспонирования).
template <int BK_, int BQ, int D>
__global__ void k_chain(const __half* Kk, const __half* Q, const __half* dO, const float* lse,
                        __half* Pout, float* dV) {
    __shared__ __half sK[BK_ * D], sQ[BQ * D], sP[BK_ * BQ], sdO[BQ * D];
    const int lane = threadIdx.x;
    for (int i = lane; i < BK_ * D; i += 32) sK[i] = Kk[i];
    for (int i = lane; i < BQ * D; i += 32) { sQ[i] = Q[i]; sdO[i] = dO[i]; }
    __syncwarp();

    WarpMma<BK_ / 16, BQ / 16, BLayout::kKMajor> qk;
    qk.clear();
    qk.template accumulate<D>(sK, D, 0, sQ, D, 0, lane);
    const float scale = 1.0f / sqrtf((float)D);
    qk.visit(lane, [&](float& v, int r, int c) { v = __expf(v * scale - lse[c]); });
    qk.store_half(sP, BQ, 0, 0, lane);      // P[key][query], k-непрерывен для следующего gemm
    __syncwarp();
    for (int i = lane; i < BK_ * BQ; i += 32) Pout[i] = sP[i];

    WarpMma<BK_ / 16, D / 16, BLayout::kNMajor> gv;
    gv.clear();
    gv.template accumulate<BQ>(sP, BQ, 0, sdO, D, 0, lane);   // A=P (k=query), B=dO[query][d]
    gv.visit(lane, [&](float v, int r, int c) { dV[r * D + c] = v; });
}


// ---------------------------------------------------------------- часть 3: потолок, узкая плитка vs широкая
// Смысл: 32x16 нужна ради 16 варпов в блоке. Но узкая плитка даёт меньше независимых mma на шаг k,
// то есть меньше ILP. Вопрос ровно в том, окупает ли удвоение варпов потерю ILP.
template <int MB, int NB, int WARPS, int K, int REP>
__global__ __launch_bounds__(WARPS * 32) void k_ceil(const __half* A, const __half* B, float* out) {
    // Дополнение до K+4: строки фрагмента ложатся на РАЗНЫЕ банки. Без него (ld = K = 32) строки,
    // отличающиеся на 2, попадают в один банк — 8-кратный конфликт, и сравнение конфигураций
    // измеряет конфликты, а не плитки.
    constexpr int LD = K + 4;
    __shared__ __half sA[64 * LD], sB[64 * LD];
    const int tid = threadIdx.x, lane = tid & 31;
    for (int i = tid; i < 64 * K; i += WARPS * 32) {
        sA[(i / K) * LD + i % K] = A[i & 1023];
        sB[(i / K) * LD + i % K] = B[i & 1023];
    }
    __syncthreads();
    WarpMma<MB, NB, BLayout::kKMajor> op;
    op.clear();
#pragma unroll 1
    for (int r = 0; r < REP; ++r) op.template accumulate<K>(sA, LD, 0, sB, LD, 0, lane);
    float s = 0.f;
    op.visit(lane, [&](float v, int, int) { s += v; });
    if (s == 1234.5678f) out[blockIdx.x] = s;
}

template <int MB, int NB, int WARPS, int K, int REP>
void ceil_case(const char* tag, const __half* A, const __half* B, float* out, int blocks) {
    auto f = [&] { k_ceil<MB, NB, WARPS, K, REP><<<blocks, WARPS * 32>>>(A, B, out); };
    f(); CK(cudaDeviceSynchronize());
    for (int i = 0; i < 3; ++i) f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) f();
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20;
    const double flop = (double)blocks * WARPS * (MB * 16.0) * (NB * 16.0) * K * 2.0 * REP;
    cudaFuncAttributes at; cudaFuncGetAttributes(&at, (const void*)k_ceil<MB, NB, WARPS, K, REP>);
    printf("  %-22s плитка %3dx%-3d  варпов %2d  %7.2f ТФЛОП/с  %5.1f%% пика  рег %3d\n",
           tag, MB * 16, NB * 16, WARPS, flop / (ms * 1e-3) / 1e12,
           100.0 * flop / (ms * 1e-3) / 1e12 / 125.3, at.numRegs);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main() {
    CK(cudaSetDevice(0));
    srand(11);
    int fails = 0;
    printf("=== часть 1: плитки, оба варианта (32x16 — форма, которую cutlass отвергает)\n");
    tile_case<1, 1, BLayout::kKMajor, 32>("k-непрерывный B", fails);
    tile_case<2, 1, BLayout::kKMajor, 32>("k-непрерывный B  <-- 32x16", fails);
    tile_case<2, 2, BLayout::kKMajor, 64>("k-непрерывный B", fails);
    tile_case<4, 1, BLayout::kKMajor, 32>("k-непрерывный B  <-- 64x16", fails);
    tile_case<4, 4, BLayout::kKMajor, 64>("k-непрерывный B", fails);
    tile_case<1, 1, BLayout::kNMajor, 32>("n-непрерывный B", fails);
    tile_case<2, 1, BLayout::kNMajor, 32>("n-непрерывный B  <-- 32x16", fails);
    tile_case<2, 2, BLayout::kNMajor, 64>("n-непрерывный B", fails);
    tile_case<4, 2, BLayout::kNMajor, 32>("n-непрерывный B", fails);
    tile_case<4, 4, BLayout::kNMajor, 64>("n-непрерывный B", fails);

    printf("\n=== часть 2: цепочка FMHA  S=K*Q^T -> P -> dV=P^T*dO  (оба варианта в одном ядре)\n");
    constexpr int BKk = 32, BQ = 32, D = 32;
    std::vector<__half> hK(BKk * D), hQ(BQ * D), hdO(BQ * D);
    std::vector<float> hlse(BQ);
    for (auto& x : hK) x = __float2half((rand() / (float)RAND_MAX - .5f) * 1.0f);
    for (auto& x : hQ) x = __float2half((rand() / (float)RAND_MAX - .5f) * 1.0f);
    for (auto& x : hdO) x = __float2half((rand() / (float)RAND_MAX - .5f) * 1.0f);
    for (auto& x : hlse) x = (rand() / (float)RAND_MAX) * 0.5f;
    __half *dK_, *dQ, *ddO, *dP; float *dlse, *ddV;
    CK(cudaMalloc(&dK_, hK.size() * 2)); CK(cudaMalloc(&dQ, hQ.size() * 2));
    CK(cudaMalloc(&ddO, hdO.size() * 2)); CK(cudaMalloc(&dP, BKk * BQ * 2));
    CK(cudaMalloc(&dlse, BQ * 4)); CK(cudaMalloc(&ddV, BKk * D * 4));
    CK(cudaMemcpy(dK_, hK.data(), hK.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dQ, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddO, hdO.data(), hdO.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlse, hlse.data(), BQ * 4, cudaMemcpyHostToDevice));
    k_chain<BKk, BQ, D><<<1, 32>>>(dK_, dQ, ddO, dlse, dP, ddV);
    CK(cudaDeviceSynchronize());
    std::vector<__half> hP(BKk * BQ); std::vector<float> hdV(BKk * D);
    CK(cudaMemcpy(hP.data(), dP, BKk * BQ * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdV.data(), ddV, BKk * D * 4, cudaMemcpyDeviceToHost));
    // эталон
    const double scale = 1.0 / sqrt((double)D);
    std::vector<double> P(BKk * BQ);
    double eP = 0;
    for (int k = 0; k < BKk; ++k)
        for (int q = 0; q < BQ; ++q) {
            double s = 0;
            for (int t = 0; t < D; ++t) s += (double)__half2float(hK[k * D + t]) * __half2float(hQ[q * D + t]);
            P[k * BQ + q] = exp(s * scale - hlse[q]);
            eP = fmax(eP, fabs(__half2float(hP[k * BQ + q]) - P[k * BQ + q]) / fmax(P[k * BQ + q], 1e-2));
        }
    double eV = 0;
    for (int k = 0; k < BKk; ++k)
        for (int t = 0; t < D; ++t) {
            double r = 0;
            for (int q = 0; q < BQ; ++q) r += __half2float(hP[k * BQ + q]) * __half2float(hdO[q * D + t]);
            eV = fmax(eV, fabs(hdV[k * D + t] - r) / fmax(fabs(r), 1e-2));
        }
    printf("  P (fp16, против fp64-эталона): %.2e   dV (против той же P): %.2e   %s\n",
           eP, eV, (eP < 3e-3 && eV < 3e-3) ? "ОК" : "ОТКАЗ");
    if (!(eP < 3e-3 && eV < 3e-3)) ++fails;


    printf("\n=== часть 3: потолок оператора — узкая плитка при 16 варпах против широкой при 8\n");
    __half *bA, *bB; float* bo;
    CK(cudaMalloc(&bA, 2048)); CK(cudaMalloc(&bB, 2048)); CK(cudaMalloc(&bo, 4096 * 4));
    CK(cudaMemcpy(bA, hK.data(), 2048, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bB, hQ.data(), 2048, cudaMemcpyHostToDevice));
    for (int blocks : {80, 240}) {
        printf("  -- блоков %d\n", blocks);
        ceil_case<2, 2, 8, 32, 64>("32x32 / 8в (cutlass)", bA, bB, bo, blocks);
        ceil_case<4, 4, 8, 32, 64>("64x64 / 8в", bA, bB, bo, blocks);
        ceil_case<2, 1, 16, 32, 64>("32x16 / 16в  <-- цель", bA, bB, bo, blocks);
        ceil_case<1, 2, 16, 32, 64>("16x32 / 16в", bA, bB, bo, blocks);
        ceil_case<2, 2, 16, 32, 64>("32x32 / 16в", bA, bB, bo, blocks);
        ceil_case<4, 2, 16, 32, 64>("64x32 / 16в", bA, bB, bo, blocks);
    }

    printf("\n%s\n", fails ? "ЕСТЬ ОТКАЗЫ" : "ВСЁ ПРОЙДЕНО");

    return fails ? 1 : 0;
}
