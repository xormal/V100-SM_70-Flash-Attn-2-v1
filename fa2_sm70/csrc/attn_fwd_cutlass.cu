// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// FA2-sm70 forward via cutlass's own FMHA warp-GEMM machinery (task #7, cutlass-grade path).
// After R17-R20 proved every piecemeal hand-lever caps at ~3.7% (the sm70 limiter is latency at
// the 20%-occupancy cap, hidden only by cutlass's INTEGRATED register-scheduled MmaVoltaTensorOp
// pipeline), this instantiates that exact machinery: cutlass::AttentionKernel<half_t, Sm70, ...>
// from examples/41 -- the same fmha_cutlassF_*_sm70 kernel torch SDPA dispatches to (~13.5%).
// Wrapped under our attn_fwd API (BHSD in via strides; out is BSHD then permuted; GQA expanded).
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "volta_fwd_block.cuh"
#include "volta_fwd_i8.cuh"
#include "volta_fwd_ws.cuh"
#include <math_constants.h>
#include "kernel_forward.h"

// [ПОРЯДОК БЛОКОВ] При причинной маске вес плитки запросов растёт с её номером: первая считает одну
// плитку ключей, последняя -- все. Диспетчер раздаёт блоки по возрастанию blockIdx, поэтому самые
// тяжёлые попадают в ОГРЫЗОК последней волны и работают на почти пустой машине. Разворот кладёт их в
// первую полную волну (longest-processing-time-first). Замер на прототипе (tools/volta_fwd_mainloop.cu):
// S=4096 +20.2%, S=8192 +29.8%, короткие формы без изменений. FA2SM70_FWD_REV=0 отключает.
static bool fwd_reverse_blocks(bool causal) {
  if (!causal) return false;                       // без маски все блоки равны -- разворот бессмыслен
  static const int env = [] {
    const char* e = std::getenv("FA2SM70_FWD_REV");
    return e ? std::atoi(e) : 1;
  }();
  return env != 0;
}



// Свод срезов split-K. Каждый срез s дал O_s -- softmax-среднее ТОЛЬКО по своим ключам -- и lse_s.
// Истинный результат: O = sum_s w_s * O_s, где w_s = exp(lse_s - lsemax) / sum_t exp(lse_t - lsemax).
// Пустые срезы имеют lse = -inf (буфер так инициализирован) и выпадают сами.
// Один блок обслуживает одну строку (bh, q); нити идут по d.
__global__ void fwd_combine_splits_kernel(
    const float* __restrict__ Opart, const float* __restrict__ LSEpart,
    __half* __restrict__ O, float* __restrict__ Lout,
    int nsplits, int BH, int Sq, int D, int lse_dim,
    long o_split_stride, long lse_split_stride) {
  // [SASS] Первая редакция считала веса В КАЖДОЙ НИТИ и тремя проходами по LSEpart. Разбор показал
  // на 8 полезных FFMA -- 72 LDG, 36 MUFU.EX2 и 66 FSETP: веса одинаковы для всей строки, а работа
  // множилась на число нитей. Здесь веса считаются ОДИН раз на блок в разделяемой памяти, и
  // внутренний цикл сводится к LDG + FFMA.
  // [ЛОВУШКА] Пустые срезы НЕ ПИШУТ в Opart (блок выходит в advance_to_block), а Opart -- torch::empty,
  // то есть неинициализирован. Гасить их нулевым весом НЕЛЬЗЯ: 0 * NaN = NaN, и мусор из повторно
  // использованной аллокации torch отравляет строку. Поэтому активные срезы СЖИМАЮТСЯ в список, и
  // незаписанная память вообще не читается.
  extern __shared__ float ss[];                     // [0..n-1] веса, [nsplits..] индексы, затем 1/сумма и lse
  float* sw = ss;
  int* sidx = (int*)(ss + nsplits);
  const int row = blockIdx.x;
  if (row >= BH * Sq) return;
  const int bh = row / Sq, q = row - bh * Sq;
  const long lse_base = (long)bh * lse_dim + q;
  const long o_base = (long)row * D;

  if (threadIdx.x < nsplits)
    sw[threadIdx.x] = LSEpart[lse_base + (long)threadIdx.x * lse_split_stride];
  __syncthreads();
  if (threadIdx.x == 0) {                           // nsplits <= 16: последовательно и дёшево
    float m = -CUDART_INF_F;
    for (int s = 0; s < nsplits; ++s) { const float l = sw[s]; if (l > m && isfinite(l)) m = l; }
    int n = 0; float ws = 0.f;
    if (m > -CUDART_INF_F) {
      for (int s = 0; s < nsplits; ++s) {
        const float l = sw[s];
        if (!isfinite(l)) continue;                 // -inf = срез пуст, +inf = строка-заполнитель
        const float e = __expf(l - m);
        sw[n] = e; sidx[n] = s; ++n; ws += e;
      }
    }
    sidx[nsplits] = n;                              // сколько срезов реально участвует
    ss[2 * nsplits + 1] = (ws > 0.f) ? (1.f / ws) : 0.f;
    ss[2 * nsplits + 2] = (ws > 0.f) ? (m + logf(ws)) : -CUDART_INF_F;
  }
  __syncthreads();

  const int nact = sidx[nsplits];
  const float inv = ss[2 * nsplits + 1];
  for (int t = threadIdx.x; t < D; t += blockDim.x) {
    float acc = 0.f;
    const float* pv = Opart + o_base + t;
#pragma unroll 4
    for (int s = 0; s < nact; ++s) acc = fmaf(sw[s], pv[(long)sidx[s] * o_split_stride], acc);
    O[o_base + t] = __float2half(acc * inv);
  }
  if (Lout && threadIdx.x == 0) Lout[(long)bh * Sq + q] = ss[2 * nsplits + 2];
}

// [SPLIT-K ДЛЯ PREFILL] Выбор числа срезов по ключам.
//
// Измерено на боевом ядре (GPU0 @1380 МГц), мкс на голову при S=2048: H=1 130.1, H=2 78.9, H=4 44.2,
// H=8 40.5, H=32 31.4 -- при H=1 мы платим В 4.1 РАЗА больше за ту же работу. Причина не в порядке
// блоков (разворот уже отгружен), а в законе плейбука §406: расписание не короче САМОГО ДЛИННОГО
// задания. При S=2048, kQ=32, kK=128 вес плитки запросов i равен ceil((i+1)*32/128) плиток ключей,
// максимум 16, сумма 544; на 240 слотов (80 SM x 3 блока) идеал 2.27 -- дисбаланс 7x.
//
// Лечится ТОЛЬКО дроблением тяжёлых заданий. Берём наименьший split, при котором
// макс_вес/split <= идеал, и ни на шаг больше: лишние срезы добавляют пустые блоки и свод.
// Платим перечитыванием K/V -- оно почти бесплатно (L2 hit 97.7%, HBM 3.26% от пика).
static int fwd_num_splits(int B, int H, int Sq, int Sk, int d, int kQ, int kK, bool causal, int blocks_per_sm) {
  static const int env = [] {
    const char* e = std::getenv("FA2SM70_FWD_SPLITS");
    return e ? std::atoi(e) : -1;                  // -1 = выбирать по модели, 0/1 = выключено
  }();
  if (env >= 0) return env < 1 ? 1 : env;
  if (!causal) return 1;                           // без маски все плитки равны -- дробить нечего
  // HEAD-DIM GATE. Above d = kKeysPerBlock the kernel does NOT keep the output in registers
  // (kKeepOutputInRF is false) and accumulates it across V iterations through the gmem
  // output-accumulator; the split path hands that accumulator a per-split partial buffer, and the
  // two arrangements collide. Measured consequence with the gate absent: d=256 and d=512 produced
  // relL2 1.2e+01 .. 4.8e+01 -- destroyed, not merely inaccurate -- while the LSE stayed CORRECT,
  // so in training it surfaced three steps downstream as wrong dQ/dK with dV intact (delta =
  // (dO*O).sum inherits the bad O; dV never touches delta).
  // The gate is set where the measurements are: every split-K win recorded in the README is d<=128.
  if (d > 128) return 1;
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  if (nsm <= 0) nsm = 80;
  const long slots = (long)nsm * blocks_per_sm;
  const int nq = (Sq + kQ - 1) / kQ;
  const long off = (long)Sk - Sq;                  // CausalFromBottomRight
  long total = 0, maxw = 0;
  for (int i = 0; i < nq; ++i) {                   // вес плитки i = число плиток ключей после обрезки
    long keys = std::min<long>(Sk, (long)i * kQ + off + kQ);
    long w = (keys + kK - 1) / kK;
    if (w < 0) w = 0;
    total += w * (long)B * H;
    if (w > maxw) maxw = w;
  }
  if (total <= 0 || maxw <= 0) return 1;
  const double ideal = (double)total / (double)slots;
  if ((double)maxw <= ideal) return 1;             // машина уже набита -- дробление только навредит

  // Порог из ЗАМЕРА, а не из модели. Два случая с ОДИНАКОВЫМ числом блоков (256 против 240 слотов) и
  // почти одинаковым дисбалансом дали противоположный исход: 1,1,8192,128 +37%, а 1,4,2048,128 -4%.
  // Различает их СРЕДНЯЯ ТЯЖЕСТЬ блока (32.5 против 8.5 плиток): свод стоит (s+1)*B*H*Sq*d*4 байт
  // независимо от тяжести, поэтому у лёгких блоков он съедает выигрыш (9% времени ядра против 2.5%).
  const long nblocks = (long)nq * B * H;
  const double avgw = (double)total / (double)nblocks;
  if ((double)nblocks > 0.9 * (double)slots && avgw < 16.0) return 1;

  int s = (int)std::ceil((double)maxw / ideal);
  if (s > 16) s = 16;                              // выше 16 растут пустые срезы и цена свода
  return s < 2 ? 1 : s;
}


// Путь split-K: то же ядро, но выход fp32 и num_splits срезов по ключам, затем свод.
// Применяется ТОЛЬКО к чистому причинному prefill (без окна, softcap и ALiBi) -- там, где измерен
// недобор машины; остальные режимы идут прежним путём без изменений.
template <int kQ, int kK, int kMaxK>
static std::tuple<torch::Tensor, torch::Tensor> run_fmha_splitk(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale, bool causal, int nsplits) {
    const c10::cuda::CUDAGuard device_guard(Q.device());
    using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, kQ, kK, kMaxK,
                                      false, false, DefaultToBatchHook, /*kOutputBHSD*/true,
                                      /*output_t*/float>;
    int B = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Sk = K.size(2), Hkv = K.size(1);
    const int lse_dim = ((Sq + Attention::kAlignLSE - 1) / Attention::kAlignLSE) * Attention::kAlignLSE;
    auto fopt = torch::dtype(torch::kFloat32).device(Q.device());
    auto Opart = torch::empty({nsplits, B, H, Sq, d}, fopt);                       // частичные O (fp32)
    auto Lpart = torch::full({nsplits, B, H, lse_dim},
                             -std::numeric_limits<float>::infinity(), fopt);       // пустые срезы = -inf
    typename Attention::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t*>(Q.data_ptr<at::Half>());
    p.key_ptr = reinterpret_cast<cutlass::half_t*>(K.data_ptr<at::Half>());
    p.value_ptr = reinterpret_cast<cutlass::half_t*>(V.data_ptr<at::Half>());
    p.output_ptr = Opart.data_ptr<float>();
    p.output_accum_ptr = nullptr;                     // output_t уже fp32 -- накопитель идёт прямо в O
    p.logsumexp_ptr = Lpart.data_ptr<float>(); p.attn_bias_ptr = nullptr;
    p.scale = (float)scale;
    p.num_heads = H; p.num_batches = B; p.head_dim = d; p.head_dim_value = d;
    p.num_queries = Sq; p.num_keys = Sk; p.num_keys_absolute = Sk;
    p.custom_mask_type = causal ? Attention::CausalFromBottomRight : Attention::NoCustomMask;
    p.reverse_blocks = fwd_reverse_blocks(causal);
    p.window_left = -1; p.window_right = -1; p.logit_softcap = 0.f; p.alibi_slopes_ptr = nullptr;
    p.kv_head_ratio = H / Hkv;
    p.q_strideM = d; p.k_strideM = d; p.v_strideM = d;
    p.q_strideH = Sq * d; p.k_strideH = Sk * d; p.v_strideH = Sk * d;
    p.q_strideB = (long)H * Sq * d; p.k_strideB = (long)Hkv * Sk * d; p.v_strideB = (long)Hkv * Sk * d;
    p.o_strideM = d;
    p.num_splits = nsplits;
    p.o_split_stride = (long)B * H * Sq * d;          // шаг между срезами в Opart
    p.lse_split_stride = (long)B * H * lse_dim;       // шаг между срезами в Lpart
    TORCH_CHECK(Attention::check_supported(p), "cutlass FMHA split-K: params unsupported");
    auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem = int(sizeof(typename Attention::SharedStorage));
    if (smem > 0xc000) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto O = torch::empty({B, H, Sq, d}, torch::dtype(torch::kFloat16).device(Q.device()));
    auto L = torch::empty({B, H, Sq}, fopt);
    const int rows = B * H * Sq, thr = (d >= 128 ? 128 : 64);
    fwd_combine_splits_kernel<<<rows, thr, (2 * nsplits + 3) * sizeof(float), stream>>>(
        Opart.data_ptr<float>(), Lpart.data_ptr<float>(),
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()), L.data_ptr<float>(),
        nsplits, B * H, Sq, d, lse_dim, p.o_split_stride, p.lse_split_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(O, L);
}

template <int kQ, int kK, int kMaxK>

static std::tuple<torch::Tensor, torch::Tensor> run_fmha(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale, bool causal, int64_t window, double softcap, const float* alibi_ptr = nullptr, int64_t window_right = -1) {
    // kOutputBHSD=true: the kernel writes O straight into a [B,H,Sq,D] tensor, so we skip the
    // ~250us post-kernel permute+contiguous (a full copy of O that SDPA never pays -- its FMHA writes
    // BHSD natively). Bit-identical to the old BSHD-then-transpose; just no extra copy.
    const c10::cuda::CUDAGuard device_guard(Q.device());   // multi-GPU: launch on the tensors' device
    using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, kQ, kK, kMaxK, false, false, DefaultToBatchHook, /*kOutputBHSD*/true>;
    int B = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Sk = K.size(2), Hkv = K.size(1);
    // split-K только для чистого причинного prefill: остальные режимы идут прежним путём без изменений
    if (causal && window <= 0 && window_right < 0 && softcap == 0.0 && alibi_ptr == nullptr) {
        int occ = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, (const void*)attention_kernel_batched_impl<Attention>,
            (int)(Attention::kNumThreads), (size_t)sizeof(typename Attention::SharedStorage));
        if (occ < 1) occ = 1;
        const int ns = fwd_num_splits(B, H, Sq, Sk, d, kQ, kK, causal, occ);
        if (ns > 1) return run_fmha_splitk<kQ, kK, kMaxK>(Q, K, V, scale, causal, ns);
    }
    auto O = torch::empty({B, H, Sq, d}, torch::dtype(torch::kFloat16).device(Q.device()));  // BHSD out (direct)
    const int lse_dim = ((Sq + Attention::kAlignLSE - 1) / Attention::kAlignLSE) * Attention::kAlignLSE;  // padded to 32
    auto lse = torch::empty({B, H, lse_dim}, torch::dtype(torch::kFloat32).device(Q.device()));           // [B,H,lse_dim], natural-log
    typename Attention::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t*>(Q.data_ptr<at::Half>());
    p.key_ptr = reinterpret_cast<cutlass::half_t*>(K.data_ptr<at::Half>());
    p.value_ptr = reinterpret_cast<cutlass::half_t*>(V.data_ptr<at::Half>());
    p.output_ptr = reinterpret_cast<cutlass::half_t*>(O.data_ptr<at::Half>());
    torch::Tensor accum;                                                 // large head_dim (kMaxK>kKeysPerBlock)
    p.output_accum_ptr = nullptr;                                        // needs an fp32 output-accumulator buffer
    if (Attention::kNeedsOutputAccumulatorBuffer) {
        accum = torch::empty({B, Sq, H, d}, torch::dtype(torch::kFloat32).device(Q.device()));  // internal accum stays BSHD-packed
        p.output_accum_ptr = reinterpret_cast<typename Attention::output_accum_t*>(accum.data_ptr<float>());
    }
    p.logsumexp_ptr = lse.data_ptr<float>(); p.attn_bias_ptr = nullptr;
    p.scale = (float)scale;
    p.num_heads = H; p.num_batches = B; p.head_dim = d; p.head_dim_value = d;
    p.num_queries = Sq; p.num_keys = Sk; p.num_keys_absolute = Sk;
    p.custom_mask_type = causal ? Attention::CausalFromBottomRight : Attention::NoCustomMask;
    p.reverse_blocks = fwd_reverse_blocks(causal);
    // Local window: `window`>0 is the causal past-window W (keys [i-W+1, i]) -> window_left = W-1;
    // `window_right`>=0 additionally allows R future keys (two-sided). -1 on a side = unbounded.
    p.window_left = (window > 0) ? (int32_t)(window - 1) : -1;
    p.window_right = (int32_t)window_right;
    p.logit_softcap = (float)softcap;                                    // attention logit soft-cap (0 = off)
    p.alibi_slopes_ptr = alibi_ptr;                                      // per-head ALiBi slopes (nullptr = off)
    // Native GQA/MQA: kv_head_ratio = H/Hkv maps query head -> KV head (no K/V expansion for any Hkv).
    p.kv_head_ratio = H / Hkv;
    p.q_strideM = d; p.k_strideM = d; p.v_strideM = d;                    // BHSD: step to next seq pos
    p.q_strideH = Sq * d; p.k_strideH = Sk * d; p.v_strideH = Sk * d;     // per-(KV)-head stride
    p.q_strideB = (long)H * Sq * d; p.k_strideB = (long)Hkv * Sk * d; p.v_strideB = (long)Hkv * Sk * d;
    p.o_strideM = d;                                                      // BHSD out: step to next query = D
    TORCH_CHECK(Attention::check_supported(p), "cutlass FMHA: params unsupported");
    auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem = int(sizeof(typename Attention::SharedStorage));
    if (smem > 0xc000) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto L = lse.narrow(2, 0, Sq).contiguous();                          // [B,H,Sq] natural-log LSE
    return std::make_tuple(O, L);
}

// ---- Attention dropout (training; reproducible Philox) ------------------------------------------
// Separate instantiation (kSupportsDropout=true) so the default inference path stays dropout-free
// (the cutlass comment notes dropout is "quite slower on V100"). The forward applies a Bernoulli
// mask to the attention weights P with scale 1/(1-p); the backward replays the SAME mask via an
// identical Philox (seed, offset) and the matching per-(batch,head) rng offset formula.
template <int kQ, int kK, int kMaxK>
static std::tuple<torch::Tensor, torch::Tensor> run_fmha_drop(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale, bool causal,
    double dropout_p, int64_t seed, int64_t offset) {
    // kOutputBHSD=true: write O straight to [B,H,Sq,D] -> skip the post-kernel permute+contiguous copy.
    const c10::cuda::CUDAGuard device_guard(Q.device());   // multi-GPU: launch on the tensors' device
    using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, kQ, kK, kMaxK, /*dropout*/true, /*bias*/false, DefaultToBatchHook, /*kOutputBHSD*/true>;
    int B = Q.size(0), H = Q.size(1), Sq = Q.size(2), d = Q.size(3), Sk = K.size(2), Hkv = K.size(1);
    auto O = torch::empty({B, H, Sq, d}, torch::dtype(torch::kFloat16).device(Q.device()));  // BHSD out (direct)
    const int lse_dim = ((Sq + Attention::kAlignLSE - 1) / Attention::kAlignLSE) * Attention::kAlignLSE;
    auto lse = torch::empty({B, H, lse_dim}, torch::dtype(torch::kFloat32).device(Q.device()));
    typename Attention::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t*>(Q.data_ptr<at::Half>());
    p.key_ptr = reinterpret_cast<cutlass::half_t*>(K.data_ptr<at::Half>());
    p.value_ptr = reinterpret_cast<cutlass::half_t*>(V.data_ptr<at::Half>());
    p.output_ptr = reinterpret_cast<cutlass::half_t*>(O.data_ptr<at::Half>());
    p.output_accum_ptr = nullptr;
    torch::Tensor accum;
    if (Attention::kNeedsOutputAccumulatorBuffer) {
        accum = torch::empty({B, Sq, H, d}, torch::dtype(torch::kFloat32).device(Q.device()));
        p.output_accum_ptr = reinterpret_cast<typename Attention::output_accum_t*>(accum.data_ptr<float>());
    }
    p.logsumexp_ptr = lse.data_ptr<float>(); p.attn_bias_ptr = nullptr;
    p.scale = (float)scale;
    p.num_heads = H; p.num_batches = B; p.head_dim = d; p.head_dim_value = d;
    p.num_queries = Sq; p.num_keys = Sk; p.num_keys_absolute = Sk;
    p.custom_mask_type = causal ? Attention::CausalFromBottomRight : Attention::NoCustomMask;
    p.reverse_blocks = fwd_reverse_blocks(causal);
    p.window_left = -1; p.window_right = -1; p.logit_softcap = 0.0f;
    p.kv_head_ratio = H / Hkv;
    p.q_strideM = d; p.k_strideM = d; p.v_strideM = d;
    p.q_strideH = Sq * d; p.k_strideH = Sk * d; p.v_strideH = Sk * d;
    p.q_strideB = (long)H * Sq * d; p.k_strideB = (long)Hkv * Sk * d; p.v_strideB = (long)Hkv * Sk * d;
    p.o_strideM = d;                                                     // BHSD out: step to next query = D
    p.use_dropout = dropout_p > 0.0;                                     // reproducible Philox mask
    p.dropout_prob = (float)dropout_p;
    p.rng_engine_inputs = at::PhiloxCudaState((uint64_t)seed, (uint64_t)offset);
    TORCH_CHECK(Attention::check_supported(p), "cutlass FMHA dropout: params unsupported");
    auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem = int(sizeof(typename Attention::SharedStorage));
    if (smem > 0xc000) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto L = lse.narrow(2, 0, Sq).contiguous();
    return std::make_tuple(O, L);
}

std::tuple<torch::Tensor, torch::Tensor> attn_fwd_cutlass_dropout(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale, bool causal,
    double dropout_p, int64_t seed, int64_t offset) {
    int H = Q.size(1), d = Q.size(3), Hkv = K.size(1);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && H % Hkv == 0, "dropout fwd: fp16/GQA");
    TORCH_CHECK(d <= 128, "dropout fwd: head_dim <= 128 (training path)");
    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    if (d <= 64) return run_fmha_drop<64, 64, 64>(Qc, Kc, Vc, scale, causal, dropout_p, seed, offset);
    return run_fmha_drop<32, 128, 128>(Qc, Kc, Vc, scale, causal, dropout_p, seed, offset);
}

// ---- Variable-length / packed attention (cu_seqlens) --------------------------------------------
// Packed tokens: Q [Tq, H, D], K/V [Tk, Hkv, D]; cu_seqlens_{q,k} [nseq+1] int32 give per-sequence
// offsets. The cutlass FMHA forward supports this natively (seqstart_q/k_ptr): batch_id indexes a
// sequence, advance_to_block jumps to its q_start/k_start and sets num_queries/num_keys per-sequence.
// grid.x = ceil_div(max_seqlen_q, kQ) covers the longest sequence; shorter ones early-exit.
template <int kQ, int kK, int kMaxK>
static std::tuple<torch::Tensor, torch::Tensor> run_fmha_varlen(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor cu_q, torch::Tensor cu_k, int64_t max_q, int64_t max_k,
    double scale, bool causal) {
    const c10::cuda::CUDAGuard device_guard(Q.device());   // multi-GPU: launch on the tensors' device
    using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, kQ, kK, kMaxK, false, false>;
    int Tq = Q.size(0), H = Q.size(1), d = Q.size(2), Hkv = K.size(1);
    int nseq = cu_q.size(0) - 1;
    auto O = torch::empty({Tq, H, d}, torch::dtype(torch::kFloat16).device(Q.device()));       // packed [Tq,H,D]
    const int lse_dim = ((int(max_q) + Attention::kAlignLSE - 1) / Attention::kAlignLSE) * Attention::kAlignLSE;
    auto lse = torch::empty({nseq, H, lse_dim}, torch::dtype(torch::kFloat32).device(Q.device()));  // [nseq,H,lse_dim]
    typename Attention::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t*>(Q.data_ptr<at::Half>());
    p.key_ptr = reinterpret_cast<cutlass::half_t*>(K.data_ptr<at::Half>());
    p.value_ptr = reinterpret_cast<cutlass::half_t*>(V.data_ptr<at::Half>());
    p.output_ptr = reinterpret_cast<cutlass::half_t*>(O.data_ptr<at::Half>());
    torch::Tensor accum;
    p.output_accum_ptr = nullptr;
    if (Attention::kNeedsOutputAccumulatorBuffer) {
        accum = torch::empty({Tq, H, d}, torch::dtype(torch::kFloat32).device(Q.device()));     // packed accum
        p.output_accum_ptr = reinterpret_cast<typename Attention::output_accum_t*>(accum.data_ptr<float>());
    }
    p.logsumexp_ptr = lse.data_ptr<float>(); p.attn_bias_ptr = nullptr;
    p.seqstart_q_ptr = cu_q.data_ptr<int32_t>();                    // varlen: per-sequence offsets
    p.seqstart_k_ptr = cu_k.data_ptr<int32_t>();
    p.scale = (float)scale;
    p.num_heads = H; p.num_batches = nseq; p.head_dim = d; p.head_dim_value = d;
    p.num_queries = (int)max_q; p.num_keys = (int)max_k; p.num_keys_absolute = (int)max_k;   // grid sizing / caps
    p.custom_mask_type = causal ? Attention::CausalFromBottomRight : Attention::NoCustomMask;
    p.reverse_blocks = fwd_reverse_blocks(causal);
    p.window_left = -1; p.window_right = -1; p.logit_softcap = 0.0f;
    p.kv_head_ratio = H / Hkv;
    p.q_strideM = H * d; p.k_strideM = Hkv * d; p.v_strideM = Hkv * d;   // packed [T, heads, D]: token stride
    p.q_strideH = d; p.k_strideH = d; p.v_strideH = d;                   // head stride
    p.q_strideB = 0; p.k_strideB = 0; p.v_strideB = 0;                   // unused under seqstart
    p.o_strideM = H * d;
    TORCH_CHECK(Attention::check_supported(p), "cutlass FMHA varlen: params unsupported");
    auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem = int(sizeof(typename Attention::SharedStorage));
    if (smem > 0xc000) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(O, lse);                                      // O [Tq,H,D], LSE [nseq,H,lse_dim]
}

std::tuple<torch::Tensor, torch::Tensor> attn_fwd_cutlass_varlen(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor cu_q, torch::Tensor cu_k, int64_t max_q, int64_t max_k,
    double scale, bool causal) {
    int H = Q.size(1), d = Q.size(2), Hkv = K.size(1);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && H % Hkv == 0, "varlen: fp16/GQA");
    TORCH_CHECK(cu_q.scalar_type() == torch::kInt32 && cu_k.scalar_type() == torch::kInt32, "cu_seqlens int32");
    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    auto cq = cu_q.contiguous(), ck = cu_k.contiguous();
    if (d <= 64)  return run_fmha_varlen<64, 64, 64>(Qc, Kc, Vc, cq, ck, max_q, max_k, scale, causal);
    if (d <= 128) return run_fmha_varlen<32, 128, 128>(Qc, Kc, Vc, cq, ck, max_q, max_k, scale, causal);
    if (d <= 256) return run_fmha_varlen<32, 128, 256>(Qc, Kc, Vc, cq, ck, max_q, max_k, scale, causal);
    if (d <= 512) return run_fmha_varlen<32, 128, 512>(Qc, Kc, Vc, cq, ck, max_q, max_k, scale, causal);
    TORCH_CHECK(false, "varlen: d<=512");
}

// [serving] Chunked-prefill entry: Q arrives BSHD [B, Sq, H, D] and the output is written BSHD too,
// so NEITHER side needs a transpose. Without this the shim had to hand the kernel `q.transpose(1,2)`,
// which `Q.contiguous()` then materialised -- 400 MB for a 32K-token chunk, on top of everything
// else. That copy is pure overhead in time and, more importantly, in PEAK memory: it is what made
// larger prefill chunks OOM next to 17.7 GB of int8 weights, and chunk size is exactly the knob that
// controls how often the whole prefix is re-touched.
// K/V stay BHSD [B, Hkv, Sk, D] -- our paged gather already emits that layout.
template <int kQ, int kK, int kMaxK>
static std::tuple<torch::Tensor, torch::Tensor> run_fmha_qbshd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale, bool causal) {
    const c10::cuda::CUDAGuard device_guard(Q.device());
    using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, kQ, kK, kMaxK,
                                      false, false, DefaultToBatchHook, /*kOutputBHSD*/ false>;
    int B = Q.size(0), Sq = Q.size(1), H = Q.size(2), d = Q.size(3);
    int Hkv = K.size(1), Sk = K.size(2);
    auto O = torch::empty({B, Sq, H, d}, torch::dtype(torch::kFloat16).device(Q.device()));   // BSHD out
    const int lse_dim = ((Sq + Attention::kAlignLSE - 1) / Attention::kAlignLSE) * Attention::kAlignLSE;
    auto lse = torch::empty({B, H, lse_dim}, torch::dtype(torch::kFloat32).device(Q.device()));
    typename Attention::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t*>(Q.data_ptr<at::Half>());
    p.key_ptr = reinterpret_cast<cutlass::half_t*>(K.data_ptr<at::Half>());
    p.value_ptr = reinterpret_cast<cutlass::half_t*>(V.data_ptr<at::Half>());
    p.output_ptr = reinterpret_cast<cutlass::half_t*>(O.data_ptr<at::Half>());
    torch::Tensor accum; p.output_accum_ptr = nullptr;
    if (Attention::kNeedsOutputAccumulatorBuffer) {
        accum = torch::empty({B, Sq, H, d}, torch::dtype(torch::kFloat32).device(Q.device()));
        p.output_accum_ptr = reinterpret_cast<typename Attention::output_accum_t*>(accum.data_ptr<float>());
    }
    p.logsumexp_ptr = lse.data_ptr<float>(); p.attn_bias_ptr = nullptr;
    p.scale = (float)scale;
    p.num_heads = H; p.num_batches = B; p.head_dim = d; p.head_dim_value = d;
    p.num_queries = Sq; p.num_keys = Sk; p.num_keys_absolute = Sk;
    p.custom_mask_type = causal ? Attention::CausalFromBottomRight : Attention::NoCustomMask;
    p.reverse_blocks = fwd_reverse_blocks(causal);
    p.window_left = -1; p.window_right = -1; p.logit_softcap = 0.0f; p.alibi_slopes_ptr = nullptr;
    p.kv_head_ratio = H / Hkv;
    p.q_strideM = (long)H * d; p.q_strideH = d; p.q_strideB = (long)Sq * H * d;   // BSHD query
    p.k_strideM = d; p.k_strideH = (long)Sk * d; p.k_strideB = (long)Hkv * Sk * d; // BHSD key/value
    p.v_strideM = d; p.v_strideH = (long)Sk * d; p.v_strideB = (long)Hkv * Sk * d;
    p.o_strideM = (long)H * d;                                                     // BSHD out
    TORCH_CHECK(Attention::check_supported(p), "cutlass FMHA (BSHD q): params unsupported");
    auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem = int(sizeof(typename Attention::SharedStorage));
    if (smem > 0xc000) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(O, lse.narrow(2, 0, Sq).contiguous());
}

// [d=256 VALUE SPLIT] The single biggest inefficiency at d=256 is that the output accumulator does
// not fit in registers: kKeepOutputInRF is kSingleValueIteration is `kMaxK <= kKeysPerBlock`, false
// for 256 against a 128-key tile, so every key iteration round-trips the accumulator through global
// memory. Measured at equal total work (S=16384, H*d held constant): d=128 runs 42.2 TFLOP/s and
// d=256 only 22.0 -- a 1.92x penalty for the same arithmetic.
//
// The way out on sm_70: `kMaxK` bounds only head_dim_VALUE here. MM0's iteration count comes from the
// runtime `p.head_dim` (kernel_forward.h: gemm_k_iterations from problem_size_0_k), and Volta's
// MakeCustomMma<MmaPipelined, kMaxK> specialisation IGNORES kMaxK entirely (only the Multistage one
// uses it, to trim stages). So we can instantiate kMaxK=128 -- putting the output in registers -- while
// still feeding head_dim=256 to the QK GEMM.
//
// Cost: the softmax depends only on Q,K, so splitting V in half recomputes QK twice -- 1.5x the
// arithmetic (2*QK + PV instead of QK + PV) for 1.9x the efficiency. Net ~1.28x expected.
template <int kQ, int kK>
static void run_fmha_vsplit_pass(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                 torch::Tensor O, torch::Tensor lse, int half,
                                 double scale, bool causal) {
    using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, kQ, kK, /*kMaxK*/ 128,
                                      false, false, DefaultToBatchHook, /*kOutputBHSD*/ false>;
    static_assert(Attention::kSingleValueIteration, "value split must keep the output in registers");
    int B = Q.size(0), Sq = Q.size(1), H = Q.size(2), d = Q.size(3);
    int Hkv = K.size(1), Sk = K.size(2);
    const int dv = d / 2;
    typename Attention::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t*>(Q.data_ptr<at::Half>());
    p.key_ptr = reinterpret_cast<cutlass::half_t*>(K.data_ptr<at::Half>());
    p.value_ptr = reinterpret_cast<cutlass::half_t*>(V.data_ptr<at::Half>()) + (long)half * dv;
    p.output_ptr = reinterpret_cast<cutlass::half_t*>(O.data_ptr<at::Half>());
    p.output_accum_ptr = nullptr;                  // in registers now -- the whole point
    p.logsumexp_ptr = lse.data_ptr<float>(); p.attn_bias_ptr = nullptr;
    p.scale = (float)scale;
    p.num_heads = H; p.num_batches = B;
    p.head_dim = d;                                // QK still sees the full 256
    p.head_dim_value = dv;                         // PV / output sees 128
    p.num_queries = Sq; p.num_keys = Sk; p.num_keys_absolute = Sk;
    p.custom_mask_type = causal ? Attention::CausalFromBottomRight : Attention::NoCustomMask;
    p.reverse_blocks = fwd_reverse_blocks(causal);
    p.window_left = -1; p.window_right = -1; p.logit_softcap = 0.0f; p.alibi_slopes_ptr = nullptr;
    p.kv_head_ratio = H / Hkv;
    p.q_strideM = (long)H * d; p.q_strideH = d; p.q_strideB = (long)Sq * H * d;      // BSHD query
    p.k_strideM = d; p.k_strideH = (long)Sk * d; p.k_strideB = (long)Hkv * Sk * d;   // BHSD key
    p.v_strideM = d; p.v_strideH = (long)Sk * d; p.v_strideB = (long)Hkv * Sk * d;   // stride is the FULL row
    p.o_strideM = (long)H * dv;                                                       // BSHD half-width out
    TORCH_CHECK(Attention::check_supported(p), "cutlass FMHA (value split): params unsupported");
    auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem = int(sizeof(typename Attention::SharedStorage));
    if (smem > 0xc000) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kQ, int kK>
static std::tuple<torch::Tensor, torch::Tensor> attn_fwd_vsplit_impl(torch::Tensor Q, torch::Tensor K,
                                                                     torch::Tensor V, double scale, bool causal) {
    const c10::cuda::CUDAGuard device_guard(Q.device());
    int B = Q.size(0), Sq = Q.size(1), H = Q.size(2), d = Q.size(3);
    TORCH_CHECK(d == 256, "value split is for head_dim == 256");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && H % K.size(1) == 0, "vsplit: fp16/GQA");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "vsplit: contiguous inputs");
    auto opt = torch::dtype(torch::kFloat16).device(Q.device());
    auto O0 = torch::empty({B, Sq, H, d / 2}, opt), O1 = torch::empty({B, Sq, H, d / 2}, opt);
    const int lse_dim = ((Sq + 32 - 1) / 32) * 32;
    auto lse = torch::empty({B, H, lse_dim}, torch::dtype(torch::kFloat32).device(Q.device()));
    // Both passes share Q and K, so both recompute the SAME softmax; the LSE from either is the one.
    run_fmha_vsplit_pass<kQ, kK>(Q, K, V, O0, lse, 0, scale, causal);
    run_fmha_vsplit_pass<kQ, kK>(Q, K, V, O1, lse, 1, scale, causal);
    auto O = torch::cat({O0, O1}, -1);
    return std::make_tuple(O, lse.narrow(2, 0, Sq).contiguous());
}

std::tuple<torch::Tensor, torch::Tensor> attn_fwd_vsplit(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                                          double scale, bool causal) {
    // QUERY TILE = 64, NOT 128. kQueriesPerBlock sets the thread count (threads = kQ*kK/32), so
    // kQ=128 gives 512 threads at 128 registers each -- 65536 registers, the ENTIRE file, hence
    // exactly ONE block per SM (ncu: sm__warps_active 25.0%). With one resident block every barrier
    // stalls the whole SM and there is nothing else to hide the K/V stream behind. kQ=64 halves the
    // block to 256 threads and fits two. Measured on this model's geometry (H=24/Hkv=4, d=256,
    // causal, 8192-query chunk), useful TFLOP/s:
    //         8K     16K    32K    64K   120K keys
    //  kQ=128 19.6   24.6   25.7   24.1   25.3
    //  kQ= 64 26.4   29.8   29.6   29.4   29.2      <- 1.15x .. 1.35x
    //  kQ= 32 25.1   24.5    -      -      20.7     <- over-shrinks: more K/V re-reads per key
    // Output is bit-identical (max 3.33e-04 vs an fp32 oracle either way) -- this is a scheduling
    // choice, not a numerical one. FA2SM70_VSPLIT_KQ overrides it for re-tuning.
    const char* e = getenv("FA2SM70_VSPLIT_KQ");
    const int kq = e ? atoi(e) : 64;
    if (kq == 32)  return attn_fwd_vsplit_impl<32, 128>(Q, K, V, scale, causal);
    if (kq == 128) return attn_fwd_vsplit_impl<128, 128>(Q, K, V, scale, causal);
    return attn_fwd_vsplit_impl<64, 128>(Q, K, V, scale, causal);
}


// ==============================================================================================
// [РУКОПИСНЫЙ МЕЙНЛУП, d=256] Точка входа в ядро из volta_fwd_block.cuh.
// Q [B,Sq,H,D] BSHD, K/V [B,Hkv,Sk,D] BHSD, O [B,Sq,H,D] BSHD, LSE [B,H,Sq_pad] -- ровно те
// раскладки, что даёт и ждёт vLLM, поэтому ни одного транспонирования снаружи.
// ==============================================================================================
std::tuple<torch::Tensor, torch::Tensor> attn_fwd_volta(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                                        double scale, bool causal) {
    const c10::cuda::CUDAGuard device_guard(Q.device());
    // [ТА ЖЕ ПОКУПКА ЗАНЯТОСТИ, ЧТО В int8-ЯДРЕ] BQ=32 дробит по ЗАПРОСАМ (ничего не пересчитывается),
    // Q подаётся кусками и DC=32 -- вместе это опускает разделяемую под 48 КБ, порог двух блоков.
    // Переключаемо, чтобы прежняя конфигурация оставалась доступной для сверки.
    // [ПОКУПКА ЗАНЯТОСТИ] BQ=32 дробит по ЗАПРОСАМ (ничего не пересчитывается), Q подаётся кусками
    // (QCH -- ШАБЛОННЫЙ параметр, не макрос) и DC=32: разделяемая 45 КБ, ниже порога двух блоков.
    // [ЗАМЕРЕНО: ДЛЯ fp16 ПОКУПКА ЗАНЯТОСТИ НЕ ОКУПАЕТСЯ, по умолчанию ВЫКЛЮЧЕНО]
    // Два резидентных блока требуют <= 48 КБ, а у fp16 это достижимо ТОЛЬКО при DC=32 (при DC=64
    // выходит 49.5 КБ даже с минимальным дополнением V -- не влезает ни при каком). Платить
    // приходится дроблением куска (3.760 против 3.663 по прежнему свипу) И перечитыванием Q.
    // Замерено: 0.78-0.81x, то есть плата ВЫШЕ выигрыша. Вывод бит в бит (расхождение 0.0e+00).
    // У int8 та же покупка обходится ДАРОМ -- байтовые K/V сами вдвое ужимают разделяемую, DC
    // трогать не надо, -- и там она даёт +2.6..11.6%. Это и есть настоящая ценность int8 на этом
    // ядре: не меньше инструкций и не меньше трафика, а ДОСТУПНОСТЬ ЗАНЯТОСТИ.
    // ЗАМЕРЕНО: для fp16 покупка занятости убыточна ОБОИМИ путями под 48 КБ, и это исчерпывающе.
    //   BQ=32/BK=64/DC=32 -> 45 КБ, 0.78-0.81x  (платим дроблением куска и перезаливкой Q)
    //   BQ=64/BK=32/DC=64 -> 34 КБ, 0.55-0.63x  (BK=32 ВДВОЕ увеличивает число плиток ключей, а с
    //                                            ним всё поплиточное: редукции, барьеры, перезаливка Q)
    // У int8 та же покупка ДАРОМ -- байтовые K/V сами ужимают разделяемую вдвое, резать ничего не
    // надо. По умолчанию для fp16 ВЫКЛЮЧЕНО; включается FA2SM70_FWD_OCC=1 как след замера.
    const char* oe = getenv("FA2SM70_FWD_OCC");
    const int occ2 = oe ? atoi(oe) : 0;
    constexpr int D = 256, PIPE = 3, REV = 1, PAD = 4;
    const int B = Q.size(0), Sq = Q.size(1), H = Q.size(2), d = Q.size(3);
    const int Hkv = K.size(1), Sk = K.size(2);
    TORCH_CHECK(d == D && K.size(3) == D, "volta fwd: head_dim must be 256");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && H % Hkv == 0, "volta fwd: fp16 / GQA");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "volta fwd: contiguous");
    TORCH_CHECK(Sk >= Sq, "volta fwd: bottom-right causal needs Sk >= Sq");
    auto opt = torch::dtype(torch::kFloat16).device(Q.device());
    auto O = torch::empty({B, Sq, H, D}, opt);
    const int lse_dim = ((Sq + 31) / 32) * 32;
    auto lse = torch::empty({B, H, lse_dim}, torch::dtype(torch::kFloat32).device(Q.device()));

    // Диспетчер по ДВУМ конфигурациям: BQ/BK/DC шаблонные, рантайм-значениями их сделать нельзя.
    // [ГЕОМЕТРИЯ ЗАНЯТОСТИ ДЛЯ fp16] Прежде я пробовал только BQ=32/BK=64, где под 48 КБ приходится
    // брать DC=32 -- и платил дроблением куска. Но самый КРУПНЫЙ буфер это sV = BK*(D+PADV), значит
    // резать надо BK, а не DC: при BK=32 разделяемая падает до 34 КБ И DC=64 сохраняется.
    // Сетка при этом обязана быть 4x2: SM = BQ/WM = 16 и SN = BK/WN = 16, оба кратны 16.
    constexpr int BQa = 64, BKa = 32, DCa = 64, MINBa = 2;
    constexpr int BQb = 64, BKb = 64, DCb = 64, MINBb = 1;   // прежняя, для сверки
    const int BQ = occ2 ? BQa : BQb;
    const size_t sh = occ2
        ? (size_t)(BQa * (DCa + PAD) + BQa * (BKa + PAD) + BKa * (DCa + PAD) + BKa * (D + 8)) * sizeof(__half)
        : (size_t)(BQb * (D + PAD)   + BQb * (BKb + PAD) + BKb * (DCb + PAD) + BKb * (D + 16)) * sizeof(__half);
    // [СЕТКА ВАРПОВ ПОД ЗАМЕР] Планировщик (tools/volta_block_planner.py) выдал фальсифицируемое
    // предсказание: замеренный свип в шапке volta_fwd_block.cuh варьировал ФОРМУ плитки варпа при
    // ВОСЬМИ варпах, и там 32x32 требовала 254 регистра и разливалась. При ЧЕТЫРЁХ варпах на нить
    // приходится вдвое больше регистров, и та же форма должна влезть без разлива, а загрузок на MMA
    // станет 1.00 вместо 1.50 -- то есть ограничитель уйдёт из разделяемой памяти в тензор.
    // Арифметика перед сборкой: накопитель O это BQ*D = 16384 значений; при 8 варпах на нить
    // приходится 64 регистра, при 4 -- 128, плюс S даёт 32. С прочим ~200 при потолке 255.
    // Переключатель нужен, чтобы предсказание можно было ОПРОВЕРГНУТЬ замером, а не принять на веру.
    // [ГЕОМЕТРИЯ -- НЕ РЫЧАГ] Сетка варпов оставлена переключаемой только как след замера: 4 варпа
    // ЗАМЕРЕНО ХУЖЕ (1.146 -> 1.427 на S=4096), 16 варпов -- в пределах шума. Предсказание
    // планировщика по геометрии ОПРОВЕРГНУТО; рычаг не в форме плиток, а в МАРШРУТАХ (ROUTE ниже).
    const char* we = getenv("FA2SM70_VOLTA_WARPS");
    const int wsel = we ? atoi(we) : 8;      // 8 = отгруженное (2x4); 4 = (2x2); 41 = (1x4)
    // [МАРШРУТЫ] бит 0 -- широкая запись V; бит 1 -- подсказка .cs; бит 2 -- предвыборка L2.
    const char* re = getenv("FA2SM70_FWD_ROUTE");
    const int rsel = re ? atoi(re) : 0;
    // Хвост из шести аргументов -- подача маски адаптивной разреженности (kmask/kcnt/mask_ld/
    // nqb/nheads/jobq, см. volta_fwd_block.cuh). Здесь путь ПЛОТНЫЙ (SPARSE=0, JOBQ=0), поэтому все
    // шесть -- нули: читающие их ветви константны по шаблону и снимаются компилятором. Проверено:
    // SASS плотного ядра совпадает по регистрам (254), кадру стека (16) и числу HMMA (1024) с тем,
    // что было до появления этих аргументов.
    void (*kern)(const __half*, const __half*, const __half*, __half*, float*,
                 int, int, float, int, int, long, long, long, long, long, long,
                 long, long, long, long, long,
                 const int32_t*, const int32_t*, int, int, int, const int32_t*) = nullptr;
    int nthreads = 0;
// ПРИ BQ=32 ЗАКОННА ТОЛЬКО СЕТКА 2x4: WM=4 даёт SM = 32/4 = 8, не кратно 16, и шаблон
    // разворачивается в массивы НУЛЕВОГО размера. Инстанцировать её нельзя даже в мёртвой ветке --
    // компилятор всё равно разворачивает шаблон. Поэтому для occ2 сетка зафиксирована.
#define FA2_PICK(WMx, WNx, Rx)                                                                      \
    do { kern = causal ? fa2_sm70::block_fwd::volta_fwd_block<BQb, BKb, D, WMx, WNx, true,  DCb, false, PIPE, false, MINBb, 0, REV, 0, 0, 0, Rx> \
                       : fa2_sm70::block_fwd::volta_fwd_block<BQb, BKb, D, WMx, WNx, false, DCb, false, PIPE, false, MINBb, 0, REV, 0, 0, 0, Rx>; \
         nthreads = WMx * WNx * 32; } while (0)
#define FA2_PICK_OCC(Rx)                                                                            \
    do { kern = causal ? fa2_sm70::block_fwd::volta_fwd_block<BQa, BKa, D, 4, 2, true,  DCa, false, PIPE, false, MINBa, 0, REV, 0, 0, 0, Rx, 1> \
                       : fa2_sm70::block_fwd::volta_fwd_block<BQa, BKa, D, 4, 2, false, DCa, false, PIPE, false, MINBa, 0, REV, 0, 0, 0, Rx, 1>; \
         nthreads = 8 * 32; } while (0)
#define FA2_ROUTES(WMx, WNx)                                                                        \
    do { switch (rsel & 7) {                                                                        \
      case 1: FA2_PICK(WMx, WNx, 1); break;   case 2: FA2_PICK(WMx, WNx, 2); break;                 \
      case 3: FA2_PICK(WMx, WNx, 3); break;   case 4: FA2_PICK(WMx, WNx, 4); break;                 \
      case 5: FA2_PICK(WMx, WNx, 5); break;   case 6: FA2_PICK(WMx, WNx, 6); break;                 \
      case 7: FA2_PICK(WMx, WNx, 7); break;   default: FA2_PICK(WMx, WNx, 0); } } while (0)
    if (occ2)                 FA2_PICK_OCC(0);
    else if (wsel == 4)  FA2_ROUTES(2, 2);
    else if (wsel == 41) FA2_ROUTES(1, 4);
    else if (wsel == 16) FA2_ROUTES(4, 4);
    else                 FA2_ROUTES(2, 4);
#undef FA2_PICK
#undef FA2_PICK_OCC
#undef FA2_ROUTES
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    // Масштаб СЛИТ с log2(e): софтмакс внутри ядра ведётся по основанию 2, а LSE возвращается в
    // натуральных единицах (эпилог умножает бегущий максимум на ln2). Передать сюда натуральный
    // масштаб = тихо получить неверный softmax при верном на вид LSE.
    const float sc = (float)scale * 1.4426950408889634f;
    kern<<<dim3((Sq + BQ - 1) / BQ, H, B), nthreads, sh, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()), lse.data_ptr<float>(),
        Sq, Sk, sc, H / Hkv, causal ? (Sk - Sq) : 0,
        /*q*/ (long)H * D, (long)D, (long)Sq * H * D,
        /*k*/ (long)D, (long)Sk * D, (long)Hkv * Sk * D,
        /*o*/ (long)H * D, (long)D, (long)Sq * H * D,
        /*lse*/ (long)lse_dim, (long)H * lse_dim,
        /*маска разреженности: путь плотный*/ nullptr, nullptr, 0, 0, H, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(O, lse.narrow(2, 0, Sq).contiguous());
}

// ==============================================================================================
// [ЯДРО НОВОГО ФУНДАМЕНТА] int8 K/V. Kq/Vq -- int8 [B,Hkv,Sk,D], Ks/Vs -- float [B,Hkv,Sk].
// Точка входа нужна ради ОДНОЙ методики замера: сравнивать её с fp16-путями из отдельной
// программы нельзя -- разные обвязки дают разный накладной расход, и разница читается как эффект.
// ==============================================================================================
// nosc: 0 -- как раньше (int8 с таблицей масштабов на позицию, либо e5m2 по пустым Ks/Vs);
//       1 -- int8 БЕЗ таблицы вовсе (масштаб столбцовый, свёрнут в Q и в эпилог вне ядра);
//       2 -- e5m2 без холостой машинерии масштабов (ДИАГНОСТИКА, не эталон);
//       6..10 -- карта форматов из volta_fwd_ws.cuh (смещённый байт и варианты таблицы);
//       +100 -- ТОТ ЖЕ формат, но перекладка у везущих ПО 16 Б на выдачу (§3b, кандидат A8).
//               Законность (кратность 16 у шага строки и указателей) проверяется ниже; при
//               нарушении -- молчаливый откат на узкую перекладку.
std::tuple<torch::Tensor, torch::Tensor> attn_fwd_volta_i8(
        torch::Tensor Q, torch::Tensor Kq, torch::Tensor Ks, torch::Tensor Vq, torch::Tensor Vs,
        double scale, bool causal, int64_t nosc) {
    const c10::cuda::CUDAGuard device_guard(Q.device());
    // СЕТКА 4x2, А НЕ 2x4: замерено 0.5889 против 0.6004 мс И на 7 регистров меньше. Причина считаемая:
    // SMB = (BQ/WM)/16 падает с 2 до 1, а состояние софтмакса растёт как SMB -- девять массивов по
    // SMB*2 значений. Загрузок на MMA при этом столько же (плитка варпа 16x32 против 32x16, обе 1.5).
    // [ЗАНЯТОСТЬ КУПЛЕНА ДРОБЛЕНИЕМ ПО ЗАПРОСАМ] BQ=32 + MINB=2: два резидентных блока, 125
    // регистров, кадр стека НОЛЬ и НОЛЬ обращений LDL/STL в SASS (проверено независимо от отчёта
    // компилятора). Замерено 0.5232 против 0.5878 мс -- x1.123. Дробление ПО ЗАПРОСАМ ничего не
    // пересчитывает: каждый блок считает свои строки, K/V перечитываются из L2 (DRAM 2-4%). Разрез
    // по ВЫХОДНОЙ размерности проигрывает, потому что заставляет считать S дважды (x1.5 работы).
    // Переключатель для СВЕРКИ В ТОЙ ЖЕ ОБВЯЗКЕ: автономный замер дал BQ=32/два блока быстрее
    // (0.5232 против 0.5878), а расширение показало обратное -- значит мерить надо здесь.
    const char* i8e = getenv("FA2SM70_I8_OCC");
    const int i8occ = i8e ? atoi(i8e) : 1;
    // [ЯДРО РОЛЕЙ] 12 варпов в ОДНОМ блоке: 4 везут, 8 считают, защёлка именованным барьером.
    // Замерено x1.22-1.46 к двухблочному варианту с максимумом на чанковых формах.
    const char* wse = getenv("FA2SM70_WS");
    const int usews = wse ? atoi(wse) : 1;
    constexpr int D = 256, REV = 1, PAD = 4, BK = 64, DC = 64;
    const int BQ = (usews || i8occ) ? 32 : 64;
    const int B = Q.size(0), Sq = Q.size(1), H = Q.size(2), d = Q.size(3);
    const int Hkv = Kq.size(1), Sk = Kq.size(2);
    TORCH_CHECK(d == D && Kq.size(3) == D, "volta fwd i8: head_dim must be 256");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && Kq.scalar_type() == torch::kChar, "volta fwd i8: fp16 Q, int8 K/V");
    // [e5m2 ОПОЗНАЁТСЯ ПО ОТСУТСТВИЮ МАСШТАБОВ, И ЭТО НЕ ХИТРОСТЬ, А СВОЙСТВО ФОРМАТА]
    // У e5m2 масштаб на позицию не нужен по построению: разворот даёт истинное значение (§4nn).
    // Пустой тензор масштабов -- поэтому единственно честный признак, а не флаг рядом с данными,
    // который можно забыть согласовать. Указатели передаются как есть: при KVFMT=1 чтение из них
    // снимается компилятором (ветвь константна по шаблону), поэтому nullptr безопасен.
    //
    // ВОССТАНОВЛЕНО 2026-07-30: этот блок ПРОПАЛ при переносе файла из worktree разреженности,
    // стоявшего на более старом коммите. Шим и tests/test_vllm_i8_gather.py зовут e5m2, то есть
    // три теста падали, а подъём с --kv-cache-dtype fp8_e5m2 упал бы. Класс ошибки тот же, что с
    // заголовками ядер скана в то же утро: проверялось НАЛИЧИЕ файла, а не его ВЕРСИЯ.
    // nosc >= 100 -- ТОТ ЖЕ формат (nosc-100), но с широкой перекладкой у везущих: признак формата
    // берётся по остатку, иначе 107 (смещённый байт без таблицы) опознался бы как e5m2 по пустым Ks.
    const int64_t fmt = nosc % 100;
    const bool e5m2 = (Ks.numel() == 0) && (fmt != 1) && (fmt != 7);
    TORCH_CHECK(e5m2 || Ks.numel() == 0 || Ks.scalar_type() == torch::kFloat32, "volta fwd i8: fp32 scales");
    TORCH_CHECK(!e5m2 || usews, "volta fwd e5m2: только ядро ролей (FA2SM70_WS=1)");
    TORCH_CHECK(H % Hkv == 0 && Sk >= Sq, "volta fwd i8: GQA and Sk >= Sq");
    auto O = torch::empty({B, Sq, H, D}, torch::dtype(torch::kFloat16).device(Q.device()));
    const int lse_dim = ((Sq + 31) / 32) * 32;
    auto lse = torch::empty({B, H, lse_dim}, torch::dtype(torch::kFloat32).device(Q.device()));
    constexpr int LDQ = DC + PAD, LDP = BK + PAD, LDK8 = DC + 16, LDV = D + 16;
    const char* wse0 = getenv("FA2SM70_WS");
    const int ws0 = wse0 ? atoi(wse0) : 1;
    // у ядра ролей: Q ЦЕЛИКОМ (шаг D+PAD), K ЦЕЛИКОМ на плитку, ДВОЙНОЙ буфер
    size_t sh = ws0 ? ((size_t)(32 * (D + PAD) + 32 * (BK + PAD)) * sizeof(__half)
                       + (size_t)2 * BK * ((D + 16) + (D + 16)))
                    : ((size_t)(BQ * LDQ + BQ * LDP) * sizeof(__half) + (size_t)BK * (LDK8 + LDV));
    // [СДВИГ БАНКОВОГО ОТПЕЧАТКА МЕЖДУ БЛОКАМИ]
    // Разделяемая память у блоков логически раздельная, но ФИЗИЧЕСКИ это одни и те же 32 банка в
    // общем блоке 128 КБ, а планировщиков на SM четыре -- их LDS идут в ОДИН банковый массив.
    // Второй резидентный блок получает базу со смещением, равным размеру аллокации. У нас это
    // 31232 Б, а 31232 % 128 == 0 -- то есть смещение КРАТНО полному циклу банков (32 банка по 4 Б),
    // и оба блока ложатся на банки ТОЖДЕСТВЕННО. Худший возможный случай: одинаковый код на
    // одинаковых банках.
    // Добиваем аллокацию до смещения в ПОЛЦИКЛА (64 Б = 16 банков) -- максимальное разведение.
    // Цена: 64 байта разделяемой памяти на блок.
    if (getenv("FA2SM70_I8_NOBANKSHIFT") == nullptr) sh += 64;
    void (*kern)(const __half*, const int8_t*, const float*, const int8_t*, const float*, __half*,
                 float*, int, int, float, int, int, long, long, long, long, long, long,
                 long, long, long, long, long, long, long, long long*, int*);
    int nthr;
    // [ФАЛЬСИФИКАТОР A/B, НЕ БОЕВОЙ ПУТЬ] FA2SM70_FAKEXOR=1 подменяет e5m2 на KVFMT=2: тот же e5m2,
    // но с ОДНИМ лишним LOP3 в цепи распаковки -- ровно тем, что отличает int8 в блоке счёта.
    // Ответ побитово тот же (маска нулевая), поэтому расхождение по ВРЕМЕНИ измеряет цену ровно
    // этой инструкции, а остаток разницы с int8 -- цену таблицы масштабов у везущих варпов.
    // FA2SM70_FAKEXOR=1 -- лишняя команда у СЧИТАЮЩИХ варпов; =2 -- та же команда у ВЕЗУЩИХ.
    // Пара 1/2 при равном числе команд отвечает на вопрос, что дороже: работа как таковая или роль.
    const char* fxe = getenv("FA2SM70_FAKEXOR");
    const int fakexor = (fxe && *fxe) ? atoi(fxe) : 0;
    // [ШИРОКАЯ ПЕРЕКЛАДКА У ВЕЗУЩИХ -- ЗАКОННОСТЬ РЕШАЕТСЯ ЗДЕСЬ, А НЕ В ЯДРЕ]
    // Ядро читает K/V шестнадцатибайтовыми словами, а это ОТКАЗ (misaligned address), если шаг
    // строки или сам указатель не кратны 16. Оба -- рантаймовые: шаг задаётся раскладкой кэша
    // (здесь k_sM = D = 256, но страничная раскладка дала бы иное), указатель -- аллокатором.
    // Проверяем ОБА и молча откатываемся на узкую перекладку: правильный ответ важнее выдач.
    const long k_sM_run = (long)D;
    const bool wide_ok = (k_sM_run % 16 == 0)
                      && ((reinterpret_cast<uintptr_t>(Kq.data_ptr<int8_t>()) & 15) == 0)
                      && ((reinterpret_cast<uintptr_t>(Vq.data_ptr<int8_t>()) & 15) == 0);
    // ВАРИАНТ = nosc/100: 1 -- широкая перекладка; 2 -- широкая загрузка + узкая укладка;
    // 3 -- узкая загрузка + широкая укладка; 4 -- перестановка кусков K (SWZ); 5 -- SWZ + широкая.
    const int64_t nosc_w = (nosc >= 100 && !wide_ok) ? (nosc % 100) : nosc;
    if (usews && nosc_w >= 100) {
        const int64_t f = nosc_w % 100, var = nosc_w / 100;
#define FA2_WS_VAR(F, W, S) (causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, F, W, S> \
                                    : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, F, W, S>)
#define FA2_WS_DIAG(F, G) (causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, F, 0, 0, G> \
                                  : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, F, 0, 0, G>)
#define FA2_WS_KSP(F, W) (causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, F, W, 0, 0, 0, 2> \
                                 : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, F, W, 0, 0, 0, 2>)
#define FA2_WS_PH(F, W) (causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, F, W, 0, 0, 1> \
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, F, W, 0, 0, 1>)
        if      (f == 0 && var == 1) kern = FA2_WS_VAR(0, 1, 0);
        else if (f == 7 && var == 1) kern = FA2_WS_VAR(7, 1, 0);
        else if (f == 8 && var == 1) kern = FA2_WS_VAR(8, 1, 0);
        else if (f == 8 && var == 4) kern = FA2_WS_VAR(8, 0, 1);
        else if (f == 8 && var == 30) kern = FA2_WS_KSP(8, 0);   // расщепление цепочки накопителя
        else if (f == 8 && var == 20) kern = FA2_WS_PH(8, 0);   // разведение фаз по группам строк
        else if (f == 8 && var == 11) kern = FA2_WS_DIAG(8, 1);   // ФАЛЬСИФИКАТОР: без софтмакса
        else if (f == 8 && var == 12) kern = FA2_WS_DIAG(8, 2);   // ФАЛЬСИФИКАТОР: без внутренних защёлок
        else if (f == 8 && var == 13) kern = FA2_WS_DIAG(8, 3);   // ФАЛЬСИФИКАТОР: без второго умножения
        else if (f == 8 && var == 14) kern = FA2_WS_DIAG(8, 4);   // ФАЛЬСИФИКАТОР: везущие не везут
        else if (f == 8 && var == 15) kern = FA2_WS_DIAG(8, 5);   // ФАЛЬСИФИКАТОР: без первого умножения
        else TORCH_CHECK(false, "volta fwd i8: нет варианта для nosc=", nosc);
#undef FA2_WS_KSP
#undef FA2_WS_PH
#undef FA2_WS_DIAG
#undef FA2_WS_VAR
        nthr = (4 + 8) * 32;
    } else if (usews) {   // роли: везущие + считающие, один блок
        const int64_t nosc = nosc_w;   // откат с широкой перекладки попадает сюда
        if (nosc == 1)
                  kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 4>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 4>;
        else if (nosc == 2)
                  kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 5>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 5>;
        else if (nosc == 6)   // смещённый байт + таблица масштабов на позицию
                  kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 6>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 6>;
        else if (nosc == 7)   // смещённый байт БЕЗ таблицы -- полный кандидат
                  kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 7>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 7>;
        else if (nosc == 8)   // смещённый байт + ЧЕРЕДУЮЩАЯСЯ таблица (Ks -- float2 [ks,vs] на позицию)
                  kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 8>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 8>;
        else if (nosc == 9)   // то же, но таблица в fp16 (half2): вдвое меньше её трафика
                  kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 9>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 9>;
        else if (nosc == 10)  // как 8, но обращение за масштабами уходит ПЕРВЫМ (сдвиг фазы)
                  kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 10>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 10>;
        else if (e5m2 && fakexor == 1)
                  kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 2>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 2>;
        else if (e5m2 && fakexor == 2)
                  kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 3>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 3>;
        else if (e5m2) kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D, 1>
                                : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D, 1>;
        else      kern = causal ? fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, true,  DC, 1, REV, D>
                      : fa2_sm70::fwd_ws::volta_fwd_ws<32, BK, D, 2, 4, false, DC, 1, REV, D>;
        nthr = (4 + 8) * 32;
    } else if (i8occ) {   // BQ=32: законна только сетка 2x4 (при WM=4 выходит SM=8, не кратно 16)
        kern = causal ? fa2_sm70::fwd_i8::volta_fwd_i8<32, BK, D, 2, 4, true,  DC, 2, REV, D>
                      : fa2_sm70::fwd_i8::volta_fwd_i8<32, BK, D, 2, 4, false, DC, 2, REV, D>;
        nthr = 8 * 32;
    } else {
        kern = causal ? fa2_sm70::fwd_i8::volta_fwd_i8<64, BK, D, 4, 2, true,  DC, 1, REV, D>
                      : fa2_sm70::fwd_i8::volta_fwd_i8<64, BK, D, 4, 2, false, DC, 1, REV, D>;
        nthr = 8 * 32;
    }
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    const float sc = (float)scale * 1.4426950408889634f;
    kern<<<dim3((Sq + BQ - 1) / BQ, H, B), nthr, sh, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        Kq.data_ptr<int8_t>(), Ks.numel() ? Ks.data_ptr<float>() : nullptr,
        Vq.data_ptr<int8_t>(), Vs.numel() ? Vs.data_ptr<float>() : nullptr,
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()), lse.data_ptr<float>(),
        Sq, Sk, sc, H / Hkv, causal ? (Sk - Sq) : 0,
        (long)H * D, (long)D, (long)Sq * H * D,
        (long)D, (long)Sk * D, (long)Hkv * Sk * D,
        (long)H * D, (long)D, (long)Sq * H * D,
        (long)lse_dim, (long)H * lse_dim,
        // ЧЕРЕДУЮЩАЯСЯ таблица -- две fp32 на позицию, поэтому шаги ВДВОЕ (ядро смещает Ks в
        // единицах float, а читает как float2). Прочие форматы -- как было.
        ((fmt == 8 || fmt == 10) ? (long)2 * Sk : (long)Sk),
        ((fmt == 8 || fmt == 10) ? (long)2 * Hkv * Sk : (long)Hkv * Sk), nullptr, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(O, lse.narrow(2, 0, Sq).contiguous());
}

std::tuple<torch::Tensor, torch::Tensor> attn_fwd_vsplit(torch::Tensor, torch::Tensor, torch::Tensor, double, bool);
std::tuple<torch::Tensor, torch::Tensor> attn_fwd_volta(torch::Tensor, torch::Tensor, torch::Tensor, double, bool);

// Переключатель разбирает ЗНАЧЕНИЕ, а не факт наличия. `getenv(..) == nullptr` считает пустую строку
// заданной, поэтому FA2SM70_NO_VOLTA_FWD="" молча включал бы откат -- в A/B это делает обе ветви
// одинаковыми и читается как «ускорения нет». Ровно эта ошибка уже стоила одного ложного замера
// сегодня (переключатель, не подключённый к измеряемому коду, даёт совпадение точнее шума).
static inline bool volta_fwd_disabled() {
    const char* e = getenv("FA2SM70_NO_VOLTA_FWD");
    return e && *e && *e != '0';
}

std::tuple<torch::Tensor, torch::Tensor> attn_fwd_qbshd(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                                        double scale, bool causal) {
    int Sq = Q.size(1), H = Q.size(2), d = Q.size(3), Hkv = K.size(1), Sk = K.size(2);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && H % Hkv == 0, "qbshd: fp16/GQA");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "qbshd: contiguous inputs");
    if (d <= 64)  return run_fmha_qbshd<64, 64, 64>(Q, K, V, scale, causal);
    // Same crossover as the packed path below (same kernel template, same tiles -- the wrapper only
    // changes layout handling): measured at 16384, not 8192. See the table there.
    if (d <= 128) return (Sk >= 16384) ? run_fmha_qbshd<64, 128, 128>(Q, K, V, scale, causal)
                                      : run_fmha_qbshd<32, 128, 128>(Q, K, V, scale, causal);
    if (d == 256 && Sk >= Sq && !volta_fwd_disabled()) {
        // РУКОПИСНЫЙ МЕЙНЛУП. Замер на серверной геометрии (V100, частоты зафиксированы, causal,
        // чанк 8192 запросов), отгруженный путь cutlass -> этот, полезные TFLOP/s:
        //   H12/2  8K 15.09->10.91 (1.38x)  16K 45.23->32.50 (1.39x)  32K 106.7->76.8 (1.39x)  64K 230.7->167.3 (1.38x)
        //   H24/4  8K 29.96->21.61 (1.39x)  16K 89.73->64.29 (1.40x)  32K 210.3->151.8 (1.39x)  64K 453.9->328.0 (1.38x)
        // то есть 27 -> 37-38 TFLOP/s из 93.6 пиковых. Расхождение с прежним путём 8e-05..1.2e-04 --
        // уровень округления fp16; против fp32-эталона максимум 3.3e-04 на семи формах, включая
        // GQA, рваные Sq/Sk, батч и отсутствие маски.
        return attn_fwd_volta(Q, K, V, scale, causal);
    }
    if (d <= 256) {
        // At d=256 with long keys, the value split wins: it keeps the output accumulator in registers
        // at the price of recomputing QK. Measured (H=12/Hkv=2, causal, ms, base -> split):
        //   2048x8192    9.09 ->  8.45  1.08     15680x65536  498.25 -> 471.13  1.06
        //   8192x16384  56.16 -> 53.96  1.04      8192x131072 583.67 -> 528.45  1.10
        //  15680x32768 216.92 ->207.24  1.05
        // Less than the 1.28x the d=128-vs-d=256 efficiency gap suggested, so the accumulator is only
        // part of that gap -- but it is a real win everywhere measured, growing with the key count.
        // Exact to relL2 4e-5..8e-5 against the single-pass path.
        if (d == 256 && Sk >= 8192 && Sq >= 128 && getenv("FA2SM70_NO_VSPLIT") == nullptr)
            return attn_fwd_vsplit(Q, K, V, scale, causal);
        if (Sk >= 8192 && Sq >= 128) return run_fmha_qbshd<128, 128, 256>(Q, K, V, scale, causal);
        return run_fmha_qbshd<32, 128, 256>(Q, K, V, scale, causal);
    }
    if (d <= 512) return run_fmha_qbshd<32, 128, 512>(Q, K, V, scale, causal);
    TORCH_CHECK(false, "qbshd: head_dim <= 512");
}

std::tuple<torch::Tensor, torch::Tensor> attn_fwd_cutlass(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale, bool causal, int64_t window, double softcap, torch::Tensor alibi_slopes, int64_t window_right) {
    int H = Q.size(1), d = Q.size(3), Hkv = K.size(1);
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && H % Hkv == 0, "fp16/GQA");
    auto Qc = Q.contiguous(); auto Kc = K.contiguous(); auto Vc = V.contiguous();
    // Native GQA/MQA: no repeat_interleave -- run_fmha maps query head -> KV head via kv_head_ratio.
    int Sk = K.size(2), Sq = Q.size(2);
    const float* al = nullptr; torch::Tensor als;
    if (alibi_slopes.defined() && alibi_slopes.numel() > 0) { TORCH_CHECK(alibi_slopes.scalar_type() == torch::kFloat32 && alibi_slopes.numel() == H, "alibi_slopes fp32 [H]");
        als = alibi_slopes.contiguous(); al = als.data_ptr<float>(); }
    // Configs are all bit-exact; tile geometry tuned by shape (measured on V100, test_sweep.py):
    if (d <= 64) return run_fmha<64, 64, 64>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
    if (d <= 128) {
        // [query-tile probe] The Sk>=8192 threshold below was never swept against the SHORTER sequences:
        // at S=4096 we run the 32-query tile, and ncu on exactly that shape shows ALU 272.8M against 273.7M
        // tensor instructions -- ~32 ALU ops per score element, i.e. per-tile bookkeeping competing with the
        // math one-for-one. A wider query tile amortises that bookkeeping over more queries. FA2SM70_FWD128
        // forces the tile (1=32, 2=64, 3=128) so the threshold can be re-derived instead of assumed.
        if (const char* e = getenv("FA2SM70_FWD128")) {
            switch (atoi(e)) {
                case 1: return run_fmha<32, 128, 128>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
                case 2: return run_fmha<64, 128, 128>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
                case 3: return run_fmha<128, 128, 128>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
                default: break;
            }
        }
        // [threshold re-derived 2026-07-26] It was Sk>=8192 and it fired one octave too early. Measured
        // (V100, causal fp16, median of 7x30 iters, tiles interleaved within each repeat so drift cannot
        // land on one variant; ms):
        //                        tile32   tile64    winner
        //   B1H32  S8192          14.586   14.761   32 by 1.20%
        //   B1H16  S8192           7.461    7.544   32 by 1.11%
        //   B2H16  S8192          14.633   14.773   32 by 0.96%
        //   B1H32  S16384         60.391   58.077   64 by 3.83%
        //   B1H8   S16384         15.507   14.957   64 by 3.55%
        // The sign is stable inside each group, so the crossover is between 8192 and 16384, not at 8192.
        // NB the WIDE-tile hypothesis is separately REFUTED at d<=128: a 128-query tile is ~30% WORSE at
        // every length (S1024..8192), and tile 64 loses to 32 below the crossover. Widening amortises the
        // per-tile bookkeeping (ncu: ALU 272.8M vs tensor 273.7M, ~32 ALU ops per score element) but costs
        // CTA count and registers, and below 16K that trade is negative. The "query tile is the lever"
        // result in the notes is a d=256/512 fact -- there the limiter is the gmem output accumulator --
        // and it does NOT carry over to d<=128. Probe with FA2SM70_FWD128=1|2|3.
        if (Sk >= 16384) return run_fmha<64, 128, 128>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
        return run_fmha<32, 128, 128>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
    }
    if (d <= 256) {
        // Chunked/prefix prefill (q_len << Sk) is a DIFFERENT shape from square prefill and the
        // 32-query tile that wins at q_len == Sk loses there: measured 0.76x of SDPA at Sk=64K and
        // 0.67x at Sk=147K, while it is 1.01-1.05x on square shapes. d<=128 already gates its tile
        // on Sk; d=256 never did. FA2SM70_FWD256 forces a config for the sweep that picks the gate.
        if (const char* e = getenv("FA2SM70_FWD256")) {
            switch (atoi(e)) {
                case 1: return run_fmha<32, 128, 256>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
                case 2: return run_fmha<64, 128, 256>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
                case 3: return run_fmha<32, 64, 256>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
                case 4: return run_fmha<64, 64, 256>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
                case 7: return run_fmha<128, 128, 256>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
                default: break;
            }
        }
        // Wider QUERY tile for long sequences. d=256 needs a gmem output accumulator
        // (kSingleValueIteration is `kMaxK <= kKeysPerBlock`, false here), and its round-trip count
        // scales with the NUMBER OF QUERY BLOCKS -- so each doubling of the tile halves that traffic,
        // and the win grows with the key count rather than the query count. Measured (V100,
        // H=24/Hkv=4, causal, ms), which is exactly that progression:
        //                    32x128   64x128   128x128   SDPA
        //     8192^2           37.9     37.7      34.5    37.6
        //     16384^2         183.2    159.0     144.0   193.2
        //     16384 x  65536 1818.9   1230.6    1082.0  1367.2
        //      8192 x 147456 2242.5   1447.2    1276.5  2008.8
        // i.e. 1.57x SDPA at 147K where the old fixed 32-tile was 0.67x. Dead end, recorded so it is
        // not retried: a 256-KEY tile would satisfy kSingleValueIteration and keep the accumulator in
        // registers outright, but neither <32,256,256> nor <64,256,256> instantiates on sm_70
        // ("zero-sized variable tb_frag_A" in the pipelined MMA). Narrow tiles are far worse
        // (32x64 / 64x64 measured 10-13 TF/s vs 19-23), confirming accumulator traffic as the limiter.
        // Gated at Sk >= 8192, the threshold d<=128 already uses, rather than made unconditional:
        // below it the tiles differ by ~1-5% in BOTH directions across runs on an unlocked-clock
        // V100, which is noise, and picking a winner from noise would risk shapes already tuned.
        if (Sk >= 8192 && Sq >= 128)
            return run_fmha<128, 128, 256>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
        if (Sk >= 8192 && Sq >= 64)
            return run_fmha<64, 128, 256>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
        return run_fmha<32, 128, 256>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);
    }
    if (d <= 512) return run_fmha<32, 128, 512>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);         // MLA-scale head_dim
    if (d <= 768) return run_fmha<32, 128, 768>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);         // >512 (output-accum, smem ~33KB)
    if (d <= 1024) return run_fmha<32, 128, 1024>(Qc, Kc, Vc, scale, causal, window, softcap, al, window_right);       // up to 1024
    TORCH_CHECK(false, "cutlass FMHA wrapper: d<=1024");
}
// ТОЧКА ВХОДА ДЛЯ e5m2: ни масштабов, ни свёртки смещения -- формат самомасштабирующийся (§4nn).
// Байты принимаются как uint8 (так их хранит пул vLLM) и трактуются побитово: fp16 == byte << 8.
std::tuple<torch::Tensor, torch::Tensor> attn_fwd_volta_e5m2(
        torch::Tensor Q, torch::Tensor Kb, torch::Tensor Vb, double scale, bool causal) {
    TORCH_CHECK(Kb.element_size() == 1 && Vb.element_size() == 1, "volta fwd e5m2: однобайтовые K/V");
    auto none = torch::empty({0}, torch::dtype(torch::kFloat32).device(Q.device()));
    return attn_fwd_volta_i8(Q, Kb.view(torch::kChar), none, Vb.view(torch::kChar), none, scale, causal, 0);
}
// [БЕЗ ВТОРОГО ПОТОКА ПОДАЧИ] int8 со СТОЛБЦОВЫМ масштабом: таблицы на позицию нет вовсе.
// Вызывающий обязан сам домножить Q на масштаб K по каналу и выход -- на масштаб V по каналу;
// обе свёртки ранга-1 и стоят O(Sq*d) против O(Sq*Sk*d) работы ядра.
std::tuple<torch::Tensor, torch::Tensor> attn_fwd_volta_i8_nosc(
        torch::Tensor Q, torch::Tensor Kq, torch::Tensor Vq, double scale, bool causal) {
    auto none = torch::empty({0}, torch::dtype(torch::kFloat32).device(Q.device()));
    return attn_fwd_volta_i8(Q, Kq, none, Vq, none, scale, causal, 1);
}
// ДИАГНОСТИКА (не эталон): e5m2 с той же снятой машинерией -- отделяет выигрыш от снятия
// ХОЛОСТОЙ единицы, которую платит и эталон, от выигрыша, специфичного для int8.
std::tuple<torch::Tensor, torch::Tensor> attn_fwd_volta_e5m2_nosc(
        torch::Tensor Q, torch::Tensor Kb, torch::Tensor Vb, double scale, bool causal) {
    auto none = torch::empty({0}, torch::dtype(torch::kFloat32).device(Q.device()));
    return attn_fwd_volta_i8(Q, Kb.view(torch::kChar), none, Vb.view(torch::kChar), none, scale, causal, 2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_fwd_vsplit", &attn_fwd_vsplit,
          "sm70 cutlass FMHA, d=256 via two head_dim_value=128 passes (output stays in registers)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("scale"), pybind11::arg("causal"));
    m.def("attn_fwd_volta", &attn_fwd_volta,
          "sm70 hand-written block mainloop FMHA forward (head_dim 256, BSHD in/out)");
    m.def("attn_fwd_volta_i8", &attn_fwd_volta_i8,
          "sm70 forward with int8 K/V (new-foundation kernel: transport law satisfied)",
          pybind11::arg("Q"), pybind11::arg("Kq"), pybind11::arg("Ks"),
          pybind11::arg("Vq"), pybind11::arg("Vs"), pybind11::arg("scale"), pybind11::arg("causal"),
          pybind11::arg("nosc") = 0);
    m.def("attn_fwd_volta_i8_nosc", &attn_fwd_volta_i8_nosc,
          "sm70 forward with int8 K/V and COLUMNWISE scale folded outside the kernel (no scale table)",
          pybind11::arg("Q"), pybind11::arg("Kq"), pybind11::arg("Vq"),
          pybind11::arg("scale"), pybind11::arg("causal"));
    m.def("attn_fwd_volta_e5m2_nosc", &attn_fwd_volta_e5m2_nosc,
          "DIAGNOSTIC ONLY: e5m2 with the idle scale machinery removed (not the reference)",
          pybind11::arg("Q"), pybind11::arg("Kb"), pybind11::arg("Vb"),
          pybind11::arg("scale"), pybind11::arg("causal"));

    m.def("attn_fwd_volta_e5m2", &attn_fwd_volta_e5m2,
          "sm70 forward with e5m2 K/V: byte transport, exact expansion (fp16 == byte<<8), no scales",
          pybind11::arg("Q"), pybind11::arg("Kb"), pybind11::arg("Vb"),
          pybind11::arg("scale"), pybind11::arg("causal"));
    m.def("attn_fwd_qbshd", &attn_fwd_qbshd,
          "sm70 cutlass FMHA forward, Q and O both BSHD [B,S,H,D] (chunked-prefill: no transposes)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("scale"), pybind11::arg("causal"));
    m.def("attn_fwd_cutlass", &attn_fwd_cutlass, "sm70 cutlass FMHA forward (optional sliding window / softcap / ALiBi)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("scale"),
          pybind11::arg("causal"), pybind11::arg("window") = 0, pybind11::arg("softcap") = 0.0,
          pybind11::arg("alibi_slopes"), pybind11::arg("window_right") = -1);
    m.def("attn_fwd_cutlass_varlen", &attn_fwd_cutlass_varlen, "sm70 cutlass FMHA forward, variable-length (cu_seqlens)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("cu_q"), pybind11::arg("cu_k"), pybind11::arg("max_q"), pybind11::arg("max_k"),
          pybind11::arg("scale"), pybind11::arg("causal"));
    m.def("attn_fwd_cutlass_dropout", &attn_fwd_cutlass_dropout, "sm70 cutlass FMHA forward with reproducible dropout",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("scale"), pybind11::arg("causal"),
          pybind11::arg("dropout_p"), pybind11::arg("seed"), pybind11::arg("offset"));
}
