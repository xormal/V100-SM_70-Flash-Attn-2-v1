// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Fused W8A16 GEMV for GPTQ int8 weights on sm_70: the weight is consumed PACKED and unpacked in the
// shadow of its own load, feeding the fp16 pipeline. No INT8 tensor cores are involved -- Volta has
// none (IMMA arrived with Turing), and it does not need them.
//
// WHY THIS SHAPE OF KERNEL, MEASURED
//   int8 matmul through DP4A on this card is 54.6 TOPS against 93.6 TFLOPS for fp16 HMMA -- 0.58x.
//   So computing in int8 is the wrong move here; the win is in READING less. At M=1 the same
//   comparison, on a 5760x2880 weight:
//       dense fp16 (cublas HMMA)              51.4 us   646 GB/s   1.00x
//       packed 4-bit, unpack fused (ours)     35.8 us   232 GB/s   1.44x   <- 4x fewer bytes
//       int8 dequantised in a SEPARATE pass  283.0 us    59 GB/s   0.18x   <- the trap
//   Unpacking in a separate pass is 5.5x SLOWER than not quantising at all, because it writes the
//   whole dense weight to memory and reads it back. Fusing is not an optimisation here; it is the
//   difference between a win and a large loss.
//
//   Decode is where this pays: the weights are read ONCE PER TOKEN with no reuse, and on the 27B/TP=2
//   deployment that read is 13.5 GB per card, 48.4% of a 34 ms step. Prefill does not benefit -- a
//   weight tile is reused by ~8192 tokens there, so its 16 ms read hides behind ~2400 ms of GEMM.
//
// LAYOUT (GPTQ, bits=8, sym=false, desc_act=false)
//   qweight [K/4, N] int32 -- four consecutive k in one word, so a warp with one COLUMN per lane
//                             reads 32 consecutive words = 128 B, perfectly coalesced
//   qzeros  [K/G, N/4] int32 -- four consecutive n per word
//   scales  [K/G, N] fp16
//   w[k,n] = (q[k,n] - z[k/G,n]) * s[k/G,n]
//
// One column per THREAD also removes the cross-lane reduction entirely: each thread owns a whole
// output element and writes it once.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "w8a16_hmma.cuh"

// Shared-memory budget for the staged activation, in fp32 elements. The k-tile and the M-tile trade
// against each other inside it: MT rows x XTILE keys. 4096 floats = 16 KB leaves room for 6 blocks/SM.
#define XSMEM 4096
constexpr int xtile_of(int MT) { int t = XSMEM / MT; return t > 1024 ? 1024 : (t < 128 ? 128 : t); }
#define MAXG (1024 / 128)   // groups in the widest k-tile

// SPLIT-K IS NOT AN OPTIMISATION HERE, IT IS THE WHOLE KERNEL.
// One column per thread gives grid = N/THREADS blocks. On the real checkpoint (N=5120, THREADS=256)
// that is TWENTY blocks on an 80-SM card: three quarters of the machine idle, each thread walking all
// of K=17408 alone. Measured 48 GB/s = 0.19x of dense fp16 -- and hoisting the group metadata out of
// the inner loop moved it only 40 -> 48, which is what proves the limiter is not the inner loop.
// The arithmetic was never in question: 89 MB of packed weight is ~6 us of fma against ~174 us of
// unavoidable memory stream. Only the grid was wrong. With the k-slice split it measured 657 GB/s.
//
// MT (rows of x per pass) exists for the same reason. With MT=1 the weight is re-read once per row,
// so batched decode collapsed exactly linearly -- 2.56x at M=1, 0.43x at M=4, 0.11x at M=16. Spending
// one unpacked weight word on MT accumulators is what makes the packed read pay for a batch. The
// arrays below are indexed only by fully-unrolled loops: a runtime index would put them in local
// memory and silently undo the whole thing (ptxas -v confirms 0 bytes spilled at every MT).
//
// THE ACTIVATION IS STAGED AS fp32, NOT fp16, AND THE GROUP SUMS ARE PRECOMPUTED.
// Staging it as fp16 costs a half->float conversion of the SAME value once per output column, i.e. N
// times over, and folding the zero as (q-z)*x needs a running Sum x that costs 3 more adds per element
// per row. Both are loop-invariant in n. Converting once at staging time and reading float4 turns the
// inner step into 1 LDS.128 + 4 FFMA per row per weight word, down from ~13 instructions. ptxas showed
// no spill in either version, so the first MT=16 result (0.28x) was pure instruction issue, not
// occupancy -- the fp32 SIMT pipe is the wall at high MT, which is exactly what tensor cores exist for.
template <int THREADS, int MT, bool SPLIT>
__global__ void w8a16_gemv_kernel(const __half* __restrict__ x,
                                  const int* __restrict__ qweight,
                                  const int* __restrict__ qzeros,
                                  const __half* __restrict__ scales,
                                  __half* __restrict__ y, float* __restrict__ ypart,
                                  int M, int N, int K, int G, int groups_per_split, int m0) {
  constexpr int XT = xtile_of(MT);
  // At MT=1 the precomputed group sum does not pay: it saves 3 adds per element for a SINGLE row but
  // costs a barrier and a pass that only ng threads take part in (measured 2.56x -> 2.31x). Above one
  // row it pays, because the saving scales with MT while the barrier does not.
  constexpr bool GSUM = (MT > 1);
  __shared__ float sx[MT][XT];
  __shared__ float gs[GSUM ? MT : 1][GSUM ? MAXG : 1];   // Sum x over each group of the tile
  const int n = blockIdx.x * THREADS + threadIdx.x;      // this thread's output column
  const bool live = (n < N);
  const int nw = n >> 2, nsh = (n & 3) * 8;              // which qzeros word / byte
  const int mrows = min(MT, M - m0);                     // ragged last M-chunk

  const int kbeg = SPLIT ? blockIdx.y * groups_per_split * G : 0;
  const int kend = SPLIT ? min(K, kbeg + groups_per_split * G) : K;
  if (kbeg >= kend) return;

  float acc[MT];
  #pragma unroll
  for (int mm = 0; mm < MT; ++mm) acc[mm] = 0.f;

  for (int k0 = kbeg; k0 < kend; k0 += XT) {
    const int len = min(XT, kend - k0);
    const int ng = (len + G - 1) / G;
    __syncthreads();
    for (int t = threadIdx.x; t < len * mrows; t += THREADS) {
      const int mm = t / len, kk = t - mm * len;
      sx[mm][kk] = __half2float(x[(long)(m0 + mm) * K + k0 + kk]);
    }
    __syncthreads();
    // Reducing the group sums in the SAME pass via shared atomics was tried and is worse everywhere
    // (MT=16: 485 -> 1046 us): the contention on ng*mrows slots costs more than the extra barrier.
    if constexpr (GSUM) {
      for (int t = threadIdx.x; t < ng * mrows; t += THREADS) {   // one thread per (row, group)
        const int mm = t / ng, g = t - mm * ng;
        float a = 0.f;
        for (int i = g * G; i < min((g + 1) * G, len); ++i) a += sx[mm][i];
        gs[mm][g] = a;
      }
      __syncthreads();
    }
    if (!live) continue;
    // Outer loop over GROUPS, inner over the group's k. This is what lets the scale and zero be
    // fetched once per 128 k instead of once per 4, AND lets the inner loop unroll (a group-change
    // branch inside the k-loop blocks unrolling, so the two fixes are one fix).
    for (int gk = 0; gk < len; gk += G) {
      const int g = (k0 + gk) / G;
      const float s = __half2float(scales[(long)g * N + n]);
      const float z = (float)((qzeros[(long)g * (N >> 2) + nw] >> nsh) & 0xFF);
      const int glen = min(G, len - gk);
      float qa[MT], xa[MT];
      #pragma unroll
      for (int mm = 0; mm < MT; ++mm) { qa[mm] = 0.f; xa[mm] = 0.f; }
      #pragma unroll 4
      for (int k = 0; k < glen; k += 4) {
        // one int32 of qweight carries four consecutive k for this column; a warp holds 32
        // consecutive columns, so this load is 32 consecutive words = 128 B, fully coalesced
        const unsigned w = (unsigned)qweight[(long)((k0 + gk + k) >> 2) * N + n];
        const float q0 = (float)( w        & 0xFF), q1 = (float)((w >>  8) & 0xFF);
        const float q2 = (float)((w >> 16) & 0xFF), q3 = (float)((w >> 24) & 0xFF);
        #pragma unroll
        for (int mm = 0; mm < MT; ++mm) {
          const float4 a = *reinterpret_cast<const float4*>(&sx[mm][gk + k]);
          qa[mm] = fmaf(a.x, q0, qa[mm]); qa[mm] = fmaf(a.y, q1, qa[mm]);
          qa[mm] = fmaf(a.z, q2, qa[mm]); qa[mm] = fmaf(a.w, q3, qa[mm]);
          if constexpr (!GSUM) xa[mm] += a.x + a.y + a.z + a.w;
        }
      }
      #pragma unroll
      for (int mm = 0; mm < MT; ++mm)                    // (q-z)*s summed = s*(Sum q*x - z*Sum x)
        acc[mm] = fmaf(s, qa[mm] - z * (GSUM ? gs[mm][gk / G] : xa[mm]), acc[mm]);
    }
  }
  if (!live) return;
  #pragma unroll
  for (int mm = 0; mm < MT; ++mm) {
    if (mm >= mrows) break;
    if (SPLIT) atomicAdd(ypart + (long)(m0 + mm) * N + n, acc[mm]);
    else       y[(long)(m0 + mm) * N + n] = __float2half(acc[mm]);
  }
}

__global__ void w8a16_cast_kernel(const float* __restrict__ src, __half* __restrict__ dst, long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

// Путь на ТЕНЗОРНЫХ ядрах для M>=8, где SIMT-путь проигрывает плотному cublas: там перелом
// физический (15.7 TFLOPS против 93.6), и снять его можно только выйдя на HMMA.
torch::Tensor w8a16_gemm_hmma(torch::Tensor x, torch::Tensor qweight, torch::Tensor qzeros,
                              torch::Tensor scales, int64_t group_size) {
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda(), "w8a16 hmma: CUDA");
  auto xc = x.contiguous(), qw = qweight.contiguous(), qz = qzeros.contiguous(), sc = scales.contiguous();
  const int N = (int)qw.size(1), K = (int)qw.size(0) * 4;
  const int M = (int)(xc.numel() / xc.size(-1));
  TORCH_CHECK(group_size == 128, "w8a16 hmma: группа 128");
  TORCH_CHECK(K % 128 == 0 && (N & 3) == 0, "w8a16 hmma: K кратно 128, N кратно 4");
  auto y = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(xc.device()));
  constexpr int BN = 64, WN = 4, G = 128;
  dim3 grid((N + BN - 1) / BN, (M + 15) / 16);
  fa2_sm70::w8a16::w8a16_hmma<BN, WN, G><<<grid, WN * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()), qw.data_ptr<int>(), qz.data_ptr<int>(),
      reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor w8a16_gemv(torch::Tensor x, torch::Tensor qweight, torch::Tensor qzeros,
                         torch::Tensor scales, int64_t group_size) {
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda(), "w8a16: CUDA tensors required");
  TORCH_CHECK(qweight.dim() == 2 && scales.dim() == 2, "w8a16: qweight [K/4,N], scales [K/G,N]");
  auto xc = x.contiguous(), qw = qweight.contiguous(), qz = qzeros.contiguous(), sc = scales.contiguous();
  const int N = (int)qw.size(1), K = (int)qw.size(0) * 4;
  const int M = (int)(xc.numel() / xc.size(-1));
  const int G = (int)group_size;
  TORCH_CHECK(xc.size(-1) == K, "w8a16: K mismatch");
  TORCH_CHECK((N & 3) == 0, "w8a16: N must be a multiple of 4 (qzeros packs 4 columns per word)");
  auto y = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(xc.device()));
  constexpr int THREADS = 256;
  const int nblk = (N + THREADS - 1) / THREADS;
  const int ngroups = (K + G - 1) / G;
  // Fill the machine: aim for ~8 blocks per SM across 80 SMs. Below that the kernel is latency-bound
  // on a stream it should be bandwidth-bound on (measured: 20 blocks -> 48 GB/s of an achievable 841).
  int ksplit = max(1, min(ngroups, (80 * 8 + nblk - 1) / nblk));
  const int gps = (ngroups + ksplit - 1) / ksplit;
  ksplit = (ngroups + gps - 1) / gps;                    // trim empty tail splits
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  const auto* sp = reinterpret_cast<const __half*>(sc.data_ptr<at::Half>());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  torch::Tensor part;
  float* pp = nullptr;
  if (ksplit > 1) {
    part = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(xc.device()));
    pp = part.data_ptr<float>();
  }
  const dim3 grid = (ksplit > 1) ? dim3(nblk, ksplit) : dim3(nblk);

#define W8A16_LAUNCH(MT)                                                                     \
  do {                                                                                       \
    if (ksplit > 1)                                                                          \
      w8a16_gemv_kernel<THREADS, MT, true><<<grid, THREADS, 0, stream>>>(                     \
          xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, nullptr, pp, M, N, K, G, gps, m0);  \
    else                                                                                     \
      w8a16_gemv_kernel<THREADS, MT, false><<<grid, THREADS, 0, stream>>>(                    \
          xp, qw.data_ptr<int>(), qz.data_ptr<int>(), sp, yp, nullptr, M, N, K, G, 0, m0);    \
  } while (0)

  // Chunk M by the largest tile that fits. Each chunk costs one full pass over the weight, so the
  // packed read only pays while the batch fits in a few chunks -- above that a dense tensor-core GEMM
  // is the right kernel and the caller should use it (see w8a16_prefers_packed).
  for (int m0 = 0; m0 < M; ) {
    const int rem = M - m0;
    if      (rem >= 16) { W8A16_LAUNCH(16); m0 += 16; }
    else if (rem >=  8) { W8A16_LAUNCH(8);  m0 +=  8; }
    else if (rem >=  4) { W8A16_LAUNCH(4);  m0 +=  4; }
    else if (rem >=  2) { W8A16_LAUNCH(2);  m0 +=  2; }
    else                { W8A16_LAUNCH(1);  m0 +=  1; }
  }
#undef W8A16_LAUNCH
  if (ksplit > 1) {
    const long tot = (long)M * N;
    w8a16_cast_kernel<<<(tot + 255) / 256, 256, 0, stream>>>(pp, yp, tot);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Where the packed path wins, MEASURED on Huihui-Qwen3.6-27B-INT8 (V100, clocks locked):
//   M    down_proj 17408x5120        gate_proj 5120x17408
//    1   133 us vs dense 348  2.62x  140 us vs dense 399  2.84x
//    4   219 us vs dense 226  1.03x  242 us vs dense 225  0.93x
//   16   485 us vs dense 236  0.49x  491 us vs dense 231  0.47x
//   64  1926 us vs dense 279  0.14x 1951 us vs dense 238  0.12x
// The crossover is physical, not a defect. Below it the GEMM is bound by the weight READ and halving
// the bytes wins; above it cublas is on the tensor cores at 93.6 TFLOPS while this kernel is on the
// fp32 SIMT pipe at 15.7 -- at M=16 the FFMA count alone is a 182 us floor. Lifting the crossover
// means feeding the unpacked bytes to HMMA, which is a different (GEMM) kernel, not a tuning knob.
bool w8a16_prefers_packed(int64_t M) { return M <= 4; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w8a16_gemv", &w8a16_gemv, "fused GPTQ-int8 W8A16 GEMV (weights consumed packed, sm_70)");
  m.def("w8a16_gemm_hmma", &w8a16_gemm_hmma, "W8A16 через тензорные ядра (M>=8)");
  m.def("w8a16_prefers_packed", &w8a16_prefers_packed, "measured crossover vs a dense fp16 GEMM");
}
