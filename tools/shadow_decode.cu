// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// СТЕНД ГЛУБИНЫ ТЕНИ, хозяин №3: ЗАДЕРЖКА (внимание в ДЕКОДЕ, 4.023 мс = 12.9% токена,
// занятость ~10%, простаивают ОБА ресурса -- 258 ГБ/с из 870).
//
// Ядро -- наш flash-decoding split-KV (kernels/decode/split_decode.cu, путь fp16 KVB=16), варп на
// ключ, боевая форма B=1, d=128, контекст 63K. Вопрос тот же: сколько ВЫДАЧ и сколько БАЙТ влезает,
// пока время не тронулось, и ЧТО кончается первым при 10% занятости -- выдача, задержка или регистры.
//
// ОСОБОЕ: у хозяина, упёртого в ЗАДЕРЖКУ, ожидание противоположно памятному -- добавленная
// НЕЗАВИСИМАЯ работа должна быть бесплатной ДОЛГО (её кладут в тень промаха), а добавленная
// ЗАВИСИМАЯ (ILP=1, длинная цепочка) удлиняет критический путь сразу. Поэтому оба варианта здесь.
//
// Сборка: nvcc -arch=sm_70 -O3 -std=c++17 --use_fast_math -o shadow_decode tools/shadow_decode.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ОТКАЗ %s: %s (%d)\n", #x, cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static constexpr int BLK = 128, NW = BLK / 32;

#define RK_NONE 0
#define RK_FFMA 1     // арифметика (конвейер FMA)
#define RK_LDG  2     // байты из DRAM (та же валюта, что у хозяина, но он в неё НЕ упёрт)
#define RK_LDS  3     // разделяемая
#define RK_MUFU 4     // XU

// R -- порций наездника на ОДИН ключ (на итерацию U=4 приходится 4R).
template <int EPT, int R, int RK, int ILP>
__global__ __launch_bounds__(BLK) void split_decode_shadow(
    const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    float* __restrict__ Op, float* __restrict__ Lp, const unsigned* __restrict__ RS,
    int H, int Hkv, int Sk, int d, float scale, int num_splits, int Lc, unsigned rmask, float eps) {
    const int c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (H / Hkv);
    const int k0 = c * Lc, k1 = min(k0 + Lc, Sk);
    const int oidx = ((b * H + h) * num_splits + c);
    __shared__ float sm[NW], sl[NW], sacc[NW * 32 * EPT];

    const __half* Qb = Q + (long)(b * H + h) * d;
    float qr[EPT], acc[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) { qr[e] = scale * __half2float(Qb[lane * EPT + e]); acc[e] = 0.f; }
    float m = -CUDART_INF_F, l = 0.f;
    const long kvbase = (long)(b * Hkv + hkv) * Sk * d;
    // состояние наездника
    float rf[ILP]; unsigned ru[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) { rf[i] = (float)(lane + i + 1); ru[i] = lane + i; }

    constexpr int U = 4;
    int j = k0 + warp;
    for (; j + (U - 1) * NW < k1; j += U * NW) {
        float part[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            int jj = j + u * NW; float ksl[EPT];
            const __half* pk = K + kvbase + (long)jj * d + lane * EPT;
#pragma unroll
            for (int e2 = 0; e2 < EPT / 2; ++e2) { __half2 hh = *reinterpret_cast<const __half2*>(pk + e2 * 2);
                ksl[e2 * 2] = __low2float(hh); ksl[e2 * 2 + 1] = __high2float(hh); }
            float pp = 0.f;
#pragma unroll
            for (int e = 0; e < EPT; ++e) pp += qr[e] * ksl[e];
            part[u] = pp;
            // ---------------- НАЕЗДНИК: R порций на КЛЮЧ ----------------
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if constexpr (RK == RK_FFMA) rf[r % ILP] = fmaf(rf[r % ILP], 1.0000001f, ksl[0]);
                if constexpr (RK == RK_LDG)  ru[r % ILP] += RS[((unsigned)(lane + jj * 32 + r * 4099)) & rmask];
                if constexpr (RK == RK_LDS)  rf[r % ILP] += sacc[(lane * 17 + r * 5 + u) & (NW * 32 * EPT - 1)];
                if constexpr (RK == RK_MUFU) rf[r % ILP] = exp2f(rf[r % ILP]);
            }
            // -------------------------------------------------------------
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
            for (int o = 16; o > 0; o >>= 1) part[u] += __shfl_xor_sync(0xffffffffu, part[u], o);
        float m_old = m;
#pragma unroll
        for (int u = 0; u < U; ++u) if (part[u] > m) m = part[u];
        float corr = (m_old == -CUDART_INF_F) ? 0.f : __expf(m_old - m);
        l *= corr;
#pragma unroll
        for (int e = 0; e < EPT; ++e) acc[e] *= corr;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            float p = __expf(part[u] - m);
            l += p; int jj = j + u * NW; float vsl[EPT];
            const __half* pv = V + kvbase + (long)jj * d + lane * EPT;
#pragma unroll
            for (int e2 = 0; e2 < EPT / 2; ++e2) { __half2 hh = *reinterpret_cast<const __half2*>(pv + e2 * 2);
                vsl[e2 * 2] = __low2float(hh); vsl[e2 * 2 + 1] = __high2float(hh); }
#pragma unroll
            for (int e = 0; e < EPT; ++e) acc[e] += p * vsl[e];
        }
    }
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    float sink = 0.f;
#pragma unroll
    for (int i = 0; i < ILP; ++i) sink += rf[i] + (float)ru[i];
#pragma unroll
    for (int e = 0; e < EPT; ++e) sacc[warp * 32 * EPT + lane * EPT + e] = acc[e] + eps * sink;
    __syncthreads();
    if (warp == 0) {
        float gm = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < NW; ++w) gm = fmaxf(gm, sm[w]);
        float gl = 0.f, ga[EPT];
#pragma unroll
        for (int e = 0; e < EPT; ++e) ga[e] = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) { float cw = __expf(sm[w] - gm); gl += cw * sl[w];
#pragma unroll
            for (int e = 0; e < EPT; ++e) ga[e] += cw * sacc[w * 32 * EPT + lane * EPT + e]; }
#pragma unroll
        for (int e = 0; e < EPT; ++e) Op[(long)oidx * d + lane * EPT + e] = ga[e] / (gl + 1e-30f);
        if (lane == 0) Lp[oidx] = gm + __logf(gl + 1e-30f);
    }
}

// -------------------------------------------------------------------------------------------------
static const __half *g_q, *g_k, *g_v; static float *g_op, *g_lp; static unsigned* g_rs;
static int g_H, g_Hkv, g_Sk, g_ns, g_Lc; static unsigned g_rmask;
static const int D_DIM = 128, EPT_C = D_DIM / 32;

template <int R, int RK, int ILP>
static double bench(int iters) {
    auto kern = split_decode_shadow<EPT_C, R, RK, ILP>;
    dim3 g(g_ns, g_H, 1);
    auto go = [&] { kern<<<g, BLK>>>(g_q, g_k, g_v, g_op, g_lp, g_rs, g_H, g_Hkv, g_Sk, D_DIM,
                                     0.088388f, g_ns, g_Lc, g_rmask, 0.f); };
    go(); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a)); for (int i = 0; i < iters; ++i) go();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float t; CK(cudaEventElapsedTime(&t, a, b));
    static bool once = true;
    if (once) { cudaFuncAttributes at; cudaFuncGetAttributes(&at, (const void*)kern);
        int occ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, BLK, 0);
        printf("# базовое ядро: рег %d, разлив %d Б, smem %d Б, блоков/SM %d (варпов/SM %d из 64)\n",
               at.numRegs, (int)at.localSizeBytes, (int)at.sharedSizeBytes, occ, occ * NW);
        once = false; }
    return t * 1e3 / iters;
}

int main(int argc, char** argv) {
    CK(cudaSetDevice(0));
    const int reps = (argc > 1) ? atoi(argv[1]) : 9;
    g_Sk = (argc > 2) ? atoi(argv[2]) : 63488;
    g_ns = (argc > 3) ? atoi(argv[3]) : 8;
    g_H = 32; g_Hkv = 8;
    g_Lc = (g_Sk + g_ns - 1) / g_ns;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("# карта %s, SM %d; B=1 H=%d Hkv=%d d=%d Sk=%d сплитов=%d -> блоков %d (%.2f на SM)\n",
           p.name, p.multiProcessorCount, g_H, g_Hkv, D_DIM, g_Sk, g_ns, g_ns * g_H, g_ns * g_H / 80.0);
    const size_t kv = (size_t)g_Hkv * g_Sk * D_DIM;
    std::vector<__half> h(kv);
    for (size_t i = 0; i < kv; ++i) h[i] = __float2half(0.01f * (float)((i % 13) - 6));
    void *pq, *pk, *pv, *po, *pl, *prs;
    CK(cudaMalloc(&pk, kv * 2)); CK(cudaMemcpy(pk, h.data(), kv * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&pv, kv * 2)); CK(cudaMemcpy(pv, h.data(), kv * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&pq, (size_t)g_H * D_DIM * 2)); CK(cudaMemset(pq, 0x11, (size_t)g_H * D_DIM * 2));
    CK(cudaMalloc(&po, (size_t)g_H * g_ns * D_DIM * 4));
    CK(cudaMalloc(&pl, (size_t)g_H * g_ns * 4));
    const size_t scr = 64u << 20;                       // 256 МБ мимо L2
    CK(cudaMalloc(&prs, scr * 4)); CK(cudaMemset(prs, 1, scr * 4));
    g_q = (const __half*)pq; g_k = (const __half*)pk; g_v = (const __half*)pv;
    g_op = (float*)po; g_lp = (float*)pl; g_rs = (unsigned*)prs; g_rmask = (unsigned)(scr - 1);

    struct Row { const char* nm; int r; int rk; double (*fn)(int); std::vector<double> t; };
    std::vector<Row> R = {
        {"база",          0, 0, bench<0,0,1>, {}},
        {"FFMA x1 ilp1",  1, 1, bench<1,RK_FFMA,1>, {}}, {"FFMA x2 ilp1",  2, 1, bench<2,RK_FFMA,1>, {}},
        {"FFMA x4 ilp1",  4, 1, bench<4,RK_FFMA,1>, {}}, {"FFMA x8 ilp1",  8, 1, bench<8,RK_FFMA,1>, {}},
        {"FFMA x16 ilp1",16, 1, bench<16,RK_FFMA,1>,{}}, {"FFMA x32 ilp1",32, 1, bench<32,RK_FFMA,1>,{}},
        {"FFMA x64 ilp1",64, 1, bench<64,RK_FFMA,1>,{}}, {"FFMA x128 ilp1",128,1,bench<128,RK_FFMA,1>,{}},
        {"FFMA x8 ilp4",  8, 1, bench<8,RK_FFMA,4>, {}}, {"FFMA x32 ilp4",32, 1, bench<32,RK_FFMA,4>,{}},
        {"FFMA x128 ilp4",128,1,bench<128,RK_FFMA,4>,{}},
        {"LDG x1",        1, 2, bench<1,RK_LDG,1>, {}}, {"LDG x2",  2, 2, bench<2,RK_LDG,1>, {}},
        {"LDG x4",        4, 2, bench<4,RK_LDG,1>, {}}, {"LDG x8",  8, 2, bench<8,RK_LDG,1>, {}},
        {"LDS x2",        2, 3, bench<2,RK_LDS,1>, {}}, {"LDS x8",  8, 3, bench<8,RK_LDS,1>, {}},
        {"LDS x32",      32, 3, bench<32,RK_LDS,1>,{}},
        {"MUFU x2",       2, 4, bench<2,RK_MUFU,1>,{}}, {"MUFU x8", 8, 4, bench<8,RK_MUFU,1>,{}},
        {"MUFU x32",     32, 4, bench<32,RK_MUFU,1>,{}},
    };
    for (auto& r : R) r.fn(3);
    for (int it = 0; it < reps; ++it) for (auto& r : R) r.t.push_back(r.fn(20));
    const double bytes = 2.0 * (double)kv * 2.0;        // K и V по kv элементов fp16
    printf("# конфигурация; мкс(медиана); ГБ/с; отн.базы; инстр.наездника на ключ на поток\n");
    double base = 0;
    for (auto& r : R) {
        std::sort(r.t.begin(), r.t.end());
        const double us = r.t[r.t.size() / 2]; if (base == 0) base = us;
        printf("%-16s %9.2f %8.1f %8.4f %8d\n", r.nm, us, bytes / (us * 1e3), us / base, r.r);
    }
    return 0;
}
