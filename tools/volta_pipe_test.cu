// Программный cp.async: варпы-загрузчики + именованные барьеры, против «все грузят и считают».
// Проверяется ОДНА величина — занятость тензорного конвейера. Геометрия счёта в обоих вариантах
// идентична (16 счётных варпов, плитка варпа 32x32), различается только КТО делает глобальные загрузки.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include "volta_warp_mma.h"

using namespace fa2_sm70;
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 4;
constexpr int WC = 16;                       // счётных варпов в обоих вариантах
constexpr int MB = 2, NB = 2;                // плитка варпа 32x32 -> сетка 4x4

// ---- вариант A: 16 варпов, каждый и грузит, и считает (как во всех наших ядрах)
__global__ __launch_bounds__(WC * 32) void k_fused(const __half* __restrict__ A,
                                                   const __half* __restrict__ Bm,
                                                   float* __restrict__ out, int K, int ns) {
  __shared__ __half sA[2][BM * LD], sB[2][BN * LD];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  constexpr int kThr = WC * 32, kChkA = BM * BK / 8, kChkRow = BK / 8;
  auto stage = [&](int kt, int b) {
    for (int i = tid; i < kChkA; i += kThr) {
      const int r = i / kChkRow, c = (i % kChkRow) * 8;
      const uint4 va = *reinterpret_cast<const uint4*>(&A[(long)(blockIdx.x % ns) * BM * K + (long)r * K + kt * BK + c]);
      const uint4 vb = *reinterpret_cast<const uint4*>(&Bm[(long)(blockIdx.x % ns) * BN * K + (long)r * K + kt * BK + c]);
      __half* pa = &sA[b][r * LD + c]; __half* pb = &sB[b][r * LD + c];
      *reinterpret_cast<uint2*>(pa) = make_uint2(va.x, va.y);
      *reinterpret_cast<uint2*>(pa + 4) = make_uint2(va.z, va.w);
      *reinterpret_cast<uint2*>(pb) = make_uint2(vb.x, vb.y);
      *reinterpret_cast<uint2*>(pb + 4) = make_uint2(vb.z, vb.w);
    }
  };
  WarpMma<MB, NB, BLayout::kKMajor> op; op.clear();
  const int nt = K / BK;
  stage(0, 0); __syncthreads();
  for (int t = 0; t < nt; ++t) {
    if (t + 1 < nt) stage(t + 1, (t + 1) & 1);
    op.accumulate<BK>(sA[t & 1], LD, wm * MB * 16, sB[t & 1], LD, wn * NB * 16, lane);
    __syncthreads();
  }
  float s = 0.f;
  op.visit(lane, [&](float v, int, int) { s += v; });
  atomicAdd(&out[blockIdx.x], s);        // настоящая сумма плитки: даёт сверку A против B
}

// ---- вариант B: 16 счётных + WL варпов-загрузчиков. Счётные варпы НЕ выдают ни одной LDG:
// они не могут стоять на глобальной памяти, и их регистры не заняты стадированием.
// Синхронизация — обычный __syncthreads(): все потоки достигают его в каждой итерации, поэтому
// дедлок невозможен по построению. (Кольцо на именованных барьерах ЗАВИСЛО — см. урок ниже;
// здесь проверяется ровно одна переменная: кто грузит.)
template <int WL>
__global__ __launch_bounds__((WC + WL) * 32) void k_pipe(const __half* __restrict__ A,
                                                         const __half* __restrict__ Bm,
                                                         float* __restrict__ out, int K, int ns) {
  __shared__ __half sA[2][BM * LD], sB[2][BN * LD];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nt = K / BK;
  const bool loader = (warp >= WC);
  const int lt = tid - WC * 32;
  constexpr int kThrL = WL * 32, kChk = BM * BK / 8, kChkRow = BK / 8;

  auto load = [&](int t, int b) {
    for (int i = lt; i < kChk; i += kThrL) {
      const int r = i / kChkRow, c = (i % kChkRow) * 8;
      const uint4 va = *reinterpret_cast<const uint4*>(&A[(long)(blockIdx.x % ns) * BM * K + (long)r * K + t * BK + c]);
      const uint4 vb = *reinterpret_cast<const uint4*>(&Bm[(long)(blockIdx.x % ns) * BN * K + (long)r * K + t * BK + c]);
      __half* pa = &sA[b][r * LD + c]; __half* pb = &sB[b][r * LD + c];
      *reinterpret_cast<uint2*>(pa) = make_uint2(va.x, va.y);
      *reinterpret_cast<uint2*>(pa + 4) = make_uint2(va.z, va.w);
      *reinterpret_cast<uint2*>(pb) = make_uint2(vb.x, vb.y);
      *reinterpret_cast<uint2*>(pb + 4) = make_uint2(vb.z, vb.w);
    }
  };

  if (loader) load(0, 0);
  __syncthreads();

  const int wm = warp >> 2, wn = warp & 3;
  WarpMma<MB, NB, BLayout::kKMajor> op; op.clear();
  for (int t = 0; t < nt; ++t) {
    if (loader) { if (t + 1 < nt) load(t + 1, (t + 1) & 1); }
    else        op.accumulate<BK>(sA[t & 1], LD, wm * MB * 16, sB[t & 1], LD, wn * NB * 16, lane);
    __syncthreads();
  }
  if (!loader) {
    float s = 0.f;
    op.visit(lane, [&](float v, int, int) { s += v; });
    atomicAdd(&out[blockIdx.x], s);
  }
}


// ---- вариант C: softDMA. НЕ копия cp.async, а своё: все 16 варпов грузят (значит максимум
// запросов в полёте И самопокрытие сохранены), но выдача РАЗМАЗАНА по шагам mma детерминированно:
// на каждом шаге k выдаётся ровно одна порция. Слоты на шине известны заранее, всплесков очереди
// нет, а дистанция «выдача -> потребление» задаётся структурой, а не планировщиком.
__global__ __launch_bounds__(WC * 32) void k_dma(const __half* __restrict__ A,
                                                 const __half* __restrict__ Bm,
                                                 float* __restrict__ out, int K, int ns) {
  __shared__ __half sA[2][BM * LD], sB[2][BN * LD];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  constexpr int kThr = WC * 32, kChk = BM * BK / 8, kChkRow = BK / 8;
  constexpr int kPer = kChk / kThr;              // порций на поток за плитку
  constexpr int kSteps = BK / 4;                 // шагов mma на плитку
  static_assert(kPer <= kSteps, "порций больше, чем шагов: слот не выделить");
  const int r0 = (tid / kChkRow), c0 = (tid % kChkRow) * 8;

  uint4 sa[kPer], sb[kPer];
  auto issue = [&](int t, int p) {               // выдать порцию p плитки t (СЛОТ на шине)
    const int r = r0 + p * (kThr / kChkRow);
    sa[p] = *reinterpret_cast<const uint4*>(&A[(long)(blockIdx.x % ns) * BM * K + (long)r * K + t * BK + c0]);
    sb[p] = *reinterpret_cast<const uint4*>(&Bm[(long)(blockIdx.x % ns) * BN * K + (long)r * K + t * BK + c0]);
  };
  auto commit = [&](int b) {
#pragma unroll
    for (int p = 0; p < kPer; ++p) {
      const int r = r0 + p * (kThr / kChkRow);
      __half* pa = &sA[b][r * LD + c0]; __half* pb = &sB[b][r * LD + c0];
      *reinterpret_cast<uint2*>(pa) = make_uint2(sa[p].x, sa[p].y);
      *reinterpret_cast<uint2*>(pa + 4) = make_uint2(sa[p].z, sa[p].w);
      *reinterpret_cast<uint2*>(pb) = make_uint2(sb[p].x, sb[p].y);
      *reinterpret_cast<uint2*>(pb + 4) = make_uint2(sb[p].z, sb[p].w);
    }
  };

  WarpMma<MB, NB, BLayout::kKMajor> op; op.clear();
  const int nt = K / BK;
#pragma unroll
  for (int p = 0; p < kPer; ++p) issue(0, p);
  commit(0);
  __syncthreads();
  for (int t = 0; t < nt; ++t) {
    const int b = t & 1;
    const __half* pa = sA[b] + (wm * MB * 16) * LD;
    const __half* pb = sB[b] + (wn * NB * 16) * LD;
    uint32_t fa[MB][2], fb[NB][2];
    // шаги mma, между которыми РАВНОМЕРНО расставлены выдачи следующей плитки
#pragma unroll
    for (int k = 0; k < BK; k += 4) {
      const int step = k >> 2;
      if (t + 1 < nt && step < kPer) issue(t + 1, step);      // ровно один слот на шаг
      op.load_a(fa, pa, LD, 0, lane, k);
      op.load_b(fb, pb, LD, 0, lane, k);
      op.mma(fa, fb);
    }
    if (t + 1 < nt) { __syncthreads(); commit((t + 1) & 1); }
    __syncthreads();
  }
  float s = 0.f;
  op.visit(lane, [&](float v, int, int) { s += v; });
  atomicAdd(&out[blockIdx.x], s);
}


// ---- вариант D: LDG и STS РАЗНЕСЕНЫ (как в FMHA-ядре), но выдача НЕ размазана — всплеском в
// начале итерации. Нужен, чтобы отделить два эффекта: «разнести загрузку и запись» против
// «размазать выдачу по шагам mma». Вариант A их смешивал: там stage() делал LDG и STS подряд.
__global__ __launch_bounds__(WC * 32) void k_split(const __half* __restrict__ A,
                                                   const __half* __restrict__ Bm,
                                                   float* __restrict__ out, int K, int ns) {
  __shared__ __half sA[2][BM * LD], sB[2][BN * LD];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  constexpr int kThr = WC * 32, kChk = BM * BK / 8, kChkRow = BK / 8, kPer = kChk / kThr;
  const int r0 = tid / kChkRow, c0 = (tid % kChkRow) * 8;
  uint4 sa[kPer], sb[kPer];
  auto ldg = [&](int t) {
#pragma unroll
    for (int p = 0; p < kPer; ++p) {
      const int r = r0 + p * (kThr / kChkRow);
      sa[p] = *reinterpret_cast<const uint4*>(&A[(long)(blockIdx.x % ns) * BM * K + (long)r * K + t * BK + c0]);
      sb[p] = *reinterpret_cast<const uint4*>(&Bm[(long)(blockIdx.x % ns) * BN * K + (long)r * K + t * BK + c0]);
    }
  };
  auto sts = [&](int b) {
#pragma unroll
    for (int p = 0; p < kPer; ++p) {
      const int r = r0 + p * (kThr / kChkRow);
      __half* pa = &sA[b][r * LD + c0]; __half* pb = &sB[b][r * LD + c0];
      *reinterpret_cast<uint2*>(pa) = make_uint2(sa[p].x, sa[p].y);
      *reinterpret_cast<uint2*>(pa + 4) = make_uint2(sa[p].z, sa[p].w);
      *reinterpret_cast<uint2*>(pb) = make_uint2(sb[p].x, sb[p].y);
      *reinterpret_cast<uint2*>(pb + 4) = make_uint2(sb[p].z, sb[p].w);
    }
  };
  WarpMma<MB, NB, BLayout::kKMajor> op; op.clear();
  const int nt = K / BK;
  ldg(0); sts(0); __syncthreads();
  for (int t = 0; t < nt; ++t) {
    if (t + 1 < nt) ldg(t + 1);                        // ВСПЛЕСК: все порции разом, но до mma
    op.accumulate<BK>(sA[t & 1], LD, wm * MB * 16, sB[t & 1], LD, wn * NB * 16, lane);
    if (t + 1 < nt) { __syncthreads(); sts((t + 1) & 1); }
    __syncthreads();
  }
  float s = 0.f;
  op.visit(lane, [&](float v, int, int) { s += v; });
  atomicAdd(&out[blockIdx.x], s);
}

template <typename F>
void bench(const char* tag, F f, int threads, int blocks, int K, void* fn) {
  f(); CK(cudaDeviceSynchronize());
  for (int i = 0; i < 3; ++i) f();
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
  for (int i = 0; i < 20; ++i) f();
  cudaEventRecord(b); CK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms, a, b); ms /= 20;
  const double flop = (double)blocks * WC * (MB * 16.0) * (NB * 16.0) * K * 2.0;
  cudaFuncAttributes at; cudaFuncGetAttributes(&at, fn);
  printf("  %-34s потоков %3d  %7.4f мс  %6.1f ТФЛОП/с  %4.1f%% пика  рег %3d\n",
         tag, threads, ms, flop / (ms * 1e-3) / 1e12, 100.0 * flop / (ms * 1e-3) / 1e12 / 125.3, at.numRegs);
  cudaEventDestroy(a); cudaEventDestroy(b);
}

int main(int argc, char** argv) {
  CK(cudaSetDevice(0));
  const int K = 2048, blocks = 160;
  // Каждый блок читает СВОЮ плитку: рабочее множество 2 x 160 x 128 x 2048 x 2 Б = 160 МБ,
  // это заведомо больше L2 (6 МБ), поэтому меряется настоящая задержка DRAM, а не L2.
  const size_t nA = (size_t)blocks * BM * K, nB = (size_t)blocks * BN * K;
  __half *dA, *dB; float* dO;
  CK(cudaMalloc(&dA, nA * 2)); CK(cudaMalloc(&dB, nB * 2)); CK(cudaMalloc(&dO, blocks * 4));
  CK(cudaMemset(dA, 0x11, nA * 2)); CK(cudaMemset(dB, 0x11, nB * 2));
  printf("   рабочее множество %.0f МБ (L2 = 6 МБ)\n", (nA + nB) * 2 / 1e6);
  {   // --- КОРРЕКТНОСТЬ: B должен давать то же, что A. Раньше это не проверялось вовсе.
    std::vector<float> hA(blocks), hB(blocks);
    CK(cudaMemset(dO, 0, blocks * 4));
    k_fused<<<blocks, WC*32>>>(dA,dB,dO,K,8); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hA.data(), dO, blocks*4, cudaMemcpyDeviceToHost));
    CK(cudaMemset(dO, 0, blocks * 4));
    k_pipe<8><<<blocks,(WC+8)*32>>>(dA,dB,dO,K,8); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hB.data(), dO, blocks*4, cudaMemcpyDeviceToHost));
    double e = 0, n = 0; int bad = 0;
    for (int i = 0; i < blocks; ++i) {
      if (!isfinite(hA[i]) || !isfinite(hB[i])) ++bad;
      e += (hA[i]-hB[i])*(hA[i]-hB[i]); n += (double)hA[i]*hA[i];
    }
    printf("  сверка A против B: relL2 %.2e, нечисел %d, |A[0]| %.3e  %s\n",
           sqrt(e/fmax(n,1e-30)), bad, fabs(hA[0]),
           (sqrt(e/fmax(n,1e-30)) < 1e-6 && bad == 0 && fabs(hA[0]) > 0) ? "СОВПАДАЮТ" : "РАСХОЖДЕНИЕ");
  }
  {   // сверка C против A
    std::vector<float> ha(blocks), hc(blocks);
    CK(cudaMemset(dO, 0, blocks*4)); k_fused<<<blocks, WC*32>>>(dA,dB,dO,K,8); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ha.data(), dO, blocks*4, cudaMemcpyDeviceToHost));
    CK(cudaMemset(dO, 0, blocks*4)); k_dma<<<blocks, WC*32>>>(dA,dB,dO,K,8); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hc.data(), dO, blocks*4, cudaMemcpyDeviceToHost));
    double e=0,n=0; for (int i=0;i<blocks;++i){e+=(ha[i]-hc[i])*(ha[i]-hc[i]); n+=(double)ha[i]*ha[i];}
    printf("  сверка A против C: relL2 %.2e  %s\n", sqrt(e/fmax(n,1e-30)),
           sqrt(e/fmax(n,1e-30)) < 1e-6 ? "СОВПАДАЮТ" : "РАСХОЖДЕНИЕ");
  }
  printf("=== знак эффекта против размера рабочего множества (счётных варпов 16 везде)\n");
  printf("  %-14s %7s %7s %7s %7s | %s\n", "раб.множество", "A слито", "B роли", "D разнес", "C softDMA",
         "D/A = что даёт разнесение, C/D = что даёт размазывание");
  for (int ns : {1, 2, 4, 8, 16, 40, 160}) {
    const double mbs = (double)ns * (BM + BN) * K * 2 / 1e6;
    double t[4];
    for (int v = 0; v < 4; ++v) {
      auto go = [&]{ if (v==0) k_fused<<<blocks, WC*32>>>(dA,dB,dO,K,ns);
                     else if (v==1) k_pipe<8><<<blocks,(WC+8)*32>>>(dA,dB,dO,K,ns);
                     else if (v==2) k_dma<<<blocks, WC*32>>>(dA,dB,dO,K,ns);
                     else      k_split<<<blocks, WC*32>>>(dA,dB,dO,K,ns); };
      go(); CK(cudaDeviceSynchronize());
      for (int i=0;i<3;++i) go(); CK(cudaDeviceSynchronize());
      cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
      for (int i=0;i<20;++i) go();
      cudaEventRecord(b); CK(cudaEventSynchronize(b));
      float ms; cudaEventElapsedTime(&ms,a,b); t[v]=ms/20;
      cudaEventDestroy(a); cudaEventDestroy(b);
    }
    const double fl = (double)blocks * WC * (MB*16.0) * (NB*16.0) * K * 2.0;
    printf("  %8.1f МБ %7.1f %7.1f %7.1f %7.1f | D/A x%.3f  C/D x%.3f\n", mbs,
           fl/(t[0]*1e-3)/1e12, fl/(t[1]*1e-3)/1e12, fl/(t[3]*1e-3)/1e12, fl/(t[2]*1e-3)/1e12,
           t[0]/t[3], t[3]/t[2]);
  }
  return 0;
}
