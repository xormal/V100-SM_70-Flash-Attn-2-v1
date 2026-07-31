// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// ================================================================================================
// ЧАНКОВЫЙ СКАН GATED-DELTA-NET НА ТЕНЗОРНЫХ ЯДРАХ VOLTA -- ТОЧКА ВХОДА ИЗ PYTHON.
//
// Пять ядер скана (kkt / solve_tril / w,u / h / o) существовали как заголовки и автономные стенды,
// но НЕ БЫЛИ ПОДКЛЮЧЕНЫ: ни одной единицы трансляции их не включало, pybind у них не было, и в
// боевом пути их не стояло вообще -- `import fa2_sm70` давал НОЛЬ GDN-символов. Этот файл -- ровно
// то, чего не хватало: одна единица трансляции, которая их инстанцирует и выдаёт наружу.
//
// ЧТО ЗАМЕНЯЕТСЯ. chunk_gated_delta_rule_fwd из vllm/model_executor/layers/fla/ops/chunk.py:
//     g   = chunk_local_cumsum(g)          <- ОСТАЁТСЯ на Triton (0.195 мс, замена не окупается)
//     A   = chunk_scaled_dot_kkt_fwd(...)  <- volta_gdn_kkt          4.916 -> 0.222 мс
//     A   = solve_tril(A)                  <- volta_gdn_solve_tril   0.622 -> 0.178 мс
//     w,u = recompute_w_u_fwd(...)         <- volta_gdn_chunk_wu     1.133 -> 0.668 мс
//     h,v_new,ht = chunk_..._fwd_h(...)    <- volta_gdn_chunk_h      8.079 -> 1.359 мс
//     o   = chunk_fwd_o(...)               <- volta_gdn_chunk_o     10.193 -> 0.878 мс
// Причина разрыва одна и та же у всех пяти и измерена (volta_prims.h §4qq): Triton при цели sm_70
// НЕ ЭМИТИТ тензорных инструкций вообще -- в PTX mma.sync 0, wmma 0, hmma 0, а tl.dot
// разворачивается в цикл FP32-FMA на обычных ядрах с потолком 3.5 TFLOP/s против 93.6 у HMMA.
//
// ФОРМАТ. Модель идёт в bfloat16, а тензорные ядра Volta bf16 НЕ УМЕЮТ -- значит счёт обязан быть
// fp16, и конверсия здесь СТРУКТУРНА, а не небрежность. Она сделана НАЕЗДНИКОМ на уже существующих
// проходах, а не отдельными ядрами:
//   * выход  -- тип элемента у ядра o -- параметр шаблона (см. volta_gdn_o.cuh): эпилог упаковывает
//               пару из fp32-накопителя одной инструкцией и в fp16, и в bf16. Отдельный проход
//               .to(bfloat16) над [T][H][128] стоил бы 0.378 мс на слой; здесь он стоит НОЛЬ;
//   * вход q,k -- их и так переписывает l2norm, и он в шиме просит выход сразу в fp16;
//   * вход v,beta -- один слитый проход gdn_cast_in (ниже), который читает bf16 и пишет fp16
//               для ОБОИХ разом. Он памяти-ограничен и идёт на полосе копирования.
//
// ЗАЧЕМ ОДНА ФУНКЦИЯ НА ВЕСЬ СКАН, А НЕ ПЯТЬ. Промежуточные A, w, u, h, v_new наружу не нужны
// никому: их единственный потребитель -- следующее ядро той же цепочки. Держать их на стороне
// Python значило бы платить пятью выделениями кэш-аллокатора и пятью пересечениями границы
// расширения на КАЖДЫЙ из 48 слоёв каждого чанк-прохода (это 240 лишних вызовов на заявку).
// Пять отдельных точек входа тоже выведены -- но только для A/B и для стендов.
// ================================================================================================
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "volta_gdn_kkt.cuh"
#include "volta_gdn_tril.cuh"
#include "volta_gdn_wu.cuh"
#include "volta_gdn_h.cuh"
#include "volta_gdn_o.cuh"

namespace {

constexpr int BT = 64;      // чанк. Единственный живой размер у Qwen3.5; ядра на него рассчитаны.
constexpr int DK = 128;     // размерность головы ключа/значения у линейного слоя Qwen3.5
constexpr int DV = 128;

using fa2_sm70::gdn::kDiagRowSmem;

// ------------------------------------------------------------------------------------------------
// СЛИТАЯ КОНВЕРСИЯ ВХОДА. Читает v [T,H,DV] и beta [T,H] в bf16, пишет их же в fp16.
// Один запуск вместо двух: beta вчетверо с лишним меньше v и своим запуском платила бы почти
// целиком за накладной расход запуска (замерено по каталогу декода: 486 из 1448 ядер короче 3 мкс,
// и такие "мелочи" складываются в 1.686 мс зазоров).
// ------------------------------------------------------------------------------------------------
__global__ void gdn_cast_in(const __nv_bfloat16* __restrict__ v, __half* __restrict__ vo,
                            const __nv_bfloat16* __restrict__ b, __half* __restrict__ bo,
                            long nv, long nb) {
  // NB: этот проход СУЩЕСТВУЕТ ТОЛЬКО при bf16-развёртке. Сервер здесь поднимается с
  // --dtype float16, и тогда v/beta приходят уже в fp16 -- ядро не запускается ВООБЩЕ
  // (см. gdn_chunk_scan). Проверять надо развёртку, а не верить постановке: "модель идёт в
  // bfloat16" -- утверждение о конфигурации, и оно у этого сервера ЛОЖНО.
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long stride = (long)gridDim.x * blockDim.x;
  // v идёт ВЕКТОРАМИ ПО 8 (16 Б на конец): [T][H][128] всегда кратно 8 по последней оси.
  for (long j = i; j < nv / 8; j += stride) {
    const uint4 s = reinterpret_cast<const uint4*>(v)[j];
    const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&s);
    __half t[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) t[e] = __float2half(__bfloat162float(sb[e]));
    reinterpret_cast<uint4*>(vo)[j] = *reinterpret_cast<const uint4*>(t);
  }
  // beta -- скаляр на (позиция, голова), кратности нет; поэлементно.
  for (long j = i; j < nb; j += stride) bo[j] = __float2half(__bfloat162float(b[j]));
}

// Проверка: тензор непрерывен и того типа, что заявлено. Ошибка здесь стоит НЕ падения, а тихо
// неверного ответа (расширение читает чужую раскладку), поэтому проверяется каждый вход.
#define CHK(t, dt, name)                                                            \
  TORCH_CHECK((t).is_cuda() && (t).is_contiguous(), name " must be contiguous CUDA"); \
  TORCH_CHECK((t).scalar_type() == (dt), name " has wrong dtype")

// ------------------------------------------------------------------------------------------------
// Пять отдельных точек входа. Нужны НЕ для боевого пути, а чтобы:
//   * сверять каждый шаг с его трионовским образцом ПООТДЕЛЬНО (иначе расхождение в цепочке из
//     пяти ядер локализуется бисекцией по шагам, а это часы);
//   * мерить шаги по отдельности в одном процессе с чередованием.
// ------------------------------------------------------------------------------------------------

// A[t,h,j] = beta_t * exp(g_t - g_j) * (k_t . k_j), строго нижний треугольник.
torch::Tensor gdn_kkt(torch::Tensor k, torch::Tensor beta, torch::Tensor g,
                      torch::Tensor cu_seqlens, torch::Tensor chunk_indices, int64_t H) {
  const c10::cuda::CUDAGuard guard(k.device());
  CHK(k, torch::kHalf, "k"); CHK(beta, torch::kHalf, "beta"); CHK(g, torch::kFloat, "g");
  const int T = (int)k.size(-3), Hg = (int)k.size(-2);
  TORCH_CHECK(k.size(-1) == DK, "gdn: head_dim must be 128");
  const int NT = (int)chunk_indices.size(0);
  auto A = torch::empty({1, T, (long)H, BT}, torch::dtype(torch::kFloat).device(k.device()));
  const size_t sh = fa2_sm70::gdn::volta_gdn_kkt_smem<BT, DK>();
  // GQ = H/Hg -- параметр ШАБЛОНА: от него зависит размер __shared__ массивов beta/gate и
  // разворот эпилога. Инстанцируются только живые случаи, остальное -- явный отказ, а не
  // тихо неверный ответ.
  auto go = [&](auto tag) {
    constexpr int GQ = decltype(tag)::value;
    auto kern = fa2_sm70::gdn::volta_gdn_kkt<BT, DK, GQ, 2, 4>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    kern<<<dim3(NT, Hg), 256, sh, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)k.data_ptr(), (const __half*)beta.data_ptr(), g.data_ptr<float>(),
        A.data_ptr<float>(), cu_seqlens.data_ptr<int>(), chunk_indices.data_ptr<int>(),
        T, (int)H, Hg);
  };
  const int GQv = (int)H / Hg;
  if (GQv == 1) go(std::integral_constant<int, 1>{});
  else if (GQv == 2) go(std::integral_constant<int, 2>{});
  else if (GQv == 3) go(std::integral_constant<int, 3>{});
  else if (GQv == 4) go(std::integral_constant<int, 4>{});
  else if (GQv == 6) go(std::integral_constant<int, 6>{});
  else if (GQv == 8) go(std::integral_constant<int, 8>{});
  else TORCH_CHECK(false, "gdn_kkt: H/Hg=", GQv, " не инстанцирован");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return A;
}

// Ai = (I + A)^{-1}, унитреугольная нижняя, fp16.
torch::Tensor gdn_solve_tril(torch::Tensor A, torch::Tensor cu_seqlens,
                             torch::Tensor chunk_indices) {
  const c10::cuda::CUDAGuard guard(A.device());
  CHK(A, torch::kFloat, "A");
  const int T = (int)A.size(-3), H = (int)A.size(-2);
  const int NT = (int)chunk_indices.size(0);
  auto Ai = torch::empty({1, T, H, BT}, torch::dtype(torch::kHalf).device(A.device()));
  const size_t sh = fa2_sm70::gdn::volta_gdn_solve_tril_smem<BT, kDiagRowSmem>();
  auto kern = fa2_sm70::gdn::volta_gdn_solve_tril<BT, kDiagRowSmem>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  kern<<<dim3(NT, H), 128, sh, at::cuda::getCurrentCUDAStream()>>>(
      A.data_ptr<float>(), (__half*)Ai.data_ptr(), chunk_indices.data_ptr<int>(),
      cu_seqlens.data_ptr<int>(), T, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Ai;
}

// w = Ai (beta e^g k),  u = Ai (beta v).
std::tuple<torch::Tensor, torch::Tensor> gdn_wu(torch::Tensor k, torch::Tensor v,
                                                torch::Tensor beta, torch::Tensor g,
                                                torch::Tensor Ai, torch::Tensor cu_seqlens,
                                                torch::Tensor chunk_indices) {
  const c10::cuda::CUDAGuard guard(k.device());
  CHK(k, torch::kHalf, "k"); CHK(v, torch::kHalf, "v");
  CHK(beta, torch::kHalf, "beta"); CHK(g, torch::kFloat, "g"); CHK(Ai, torch::kHalf, "Ai");
  const int T = (int)k.size(-3), Hg = (int)k.size(-2), H = (int)v.size(-2);
  const int NT = (int)chunk_indices.size(0);
  auto w = torch::empty({1, T, H, DK}, torch::dtype(torch::kHalf).device(k.device()));
  auto u = torch::empty({1, T, H, DV}, torch::dtype(torch::kHalf).device(k.device()));
  // HPB -- голов значения на блок; обязано ДЕЛИТЬ H/Hg, иначе группа задела бы две головы ключа
  // и общая плитка k стала бы неверной. При GQA=3 это 3, при GQA=1 -- 1.
  const int GQv = H / Hg;
  const size_t sh = fa2_sm70::gdn::volta_gdn_wu_smem<BT, DK, DV>();
  auto go = [&](auto tag) {
    constexpr int HPB = decltype(tag)::value;
    auto kern = fa2_sm70::gdn::volta_gdn_chunk_wu<BT, DK, DV, 2, 4, HPB, 2, true, int>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    kern<<<dim3(NT, H / HPB), 256, sh, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)k.data_ptr(), (const __half*)v.data_ptr(), (const __half*)beta.data_ptr(),
        g.data_ptr<float>(), (const __half*)Ai.data_ptr(), (__half*)w.data_ptr(),
        (__half*)u.data_ptr(), cu_seqlens.data_ptr<int>(), chunk_indices.data_ptr<int>(),
        T, H, Hg);
  };
  if (GQv % 3 == 0) go(std::integral_constant<int, 3>{});
  else if (GQv % 2 == 0) go(std::integral_constant<int, 2>{});
  else go(std::integral_constant<int, 1>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, u};
}

// h [NT,H,DV,DK] -- состояние на НАЧАЛО каждого чанка; v_new; ht -- финальное состояние (fp32).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gdn_h(
    torch::Tensor k, torch::Tensor w, torch::Tensor u, torch::Tensor g,
    c10::optional<torch::Tensor> h0, torch::Tensor cu_seqlens, torch::Tensor chunk_offsets,
    int64_t NT, bool output_final_state) {
  const c10::cuda::CUDAGuard guard(k.device());
  CHK(k, torch::kHalf, "k"); CHK(w, torch::kHalf, "w"); CHK(u, torch::kHalf, "u");
  CHK(g, torch::kFloat, "g");
  // h0 читается ядром как const float*. Тип обязан проверяться ЗДЕСЬ: bf16-буфер, прочитанный
  // как float, даёт не отказ, а тихо неверный ответ -- худший из возможных исходов.
  if (h0.has_value()) { CHK(h0.value(), torch::kFloat, "initial_state"); }
  const int Hg = (int)k.size(-2), H = (int)u.size(-2);
  const int N = (int)cu_seqlens.size(0) - 1;
  auto h = torch::empty({1, (long)NT, H, DV, DK}, torch::dtype(torch::kHalf).device(k.device()));
  auto vn = torch::empty_like(u);
  auto ht = output_final_state
                ? torch::empty({N, H, DV, DK}, torch::dtype(torch::kFloat).device(k.device()))
                : torch::Tensor();
  constexpr int BV = 64;                        // замерено быстрейшим (стенд volta_gdn_h_test)
  const size_t sh = fa2_sm70::gdn::gdn_h_smem<BT, DK, BV>();
  auto kern = fa2_sm70::gdn::volta_gdn_chunk_h<BT, DK, DV, BV, 2, 4, 2, 4, true>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  kern<<<dim3(DV / BV, N * H), 256, sh, at::cuda::getCurrentCUDAStream()>>>(
      (const __half*)k.data_ptr(), (const __half*)w.data_ptr(), (const __half*)u.data_ptr(),
      g.data_ptr<float>(), h0.has_value() ? h0->data_ptr<float>() : nullptr,
      (__half*)h.data_ptr(), (__half*)vn.data_ptr(),
      output_final_state ? ht.data_ptr<float>() : nullptr,
      cu_seqlens.data_ptr<int>(), chunk_offsets.data_ptr<int>(), H, Hg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {h, vn, ht};
}

// o = scale * ( e^g (q h^T) + tril(e^{g_t-g_s} q k^T) v_new ).
torch::Tensor gdn_o(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor h,
                    torch::Tensor g, torch::Tensor cu_seqlens, torch::Tensor chunk_indices,
                    double scale, bool out_bf16) {
  const c10::cuda::CUDAGuard guard(q.device());
  CHK(q, torch::kHalf, "q"); CHK(k, torch::kHalf, "k"); CHK(v, torch::kHalf, "v");
  CHK(h, torch::kHalf, "h"); CHK(g, torch::kFloat, "g");
  const int T = (int)q.size(-3), Hg = (int)q.size(-2), H = (int)v.size(-2);
  const int NT = (int)chunk_indices.size(0);
  auto o = torch::empty({1, T, H, DV},
                        torch::dtype(out_bf16 ? torch::kBFloat16 : torch::kHalf).device(q.device()));
  const size_t sh = fa2_sm70::gdn::gdn_smem_bytes<BT, DK, DV>();
  auto st = at::cuda::getCurrentCUDAStream();
  if (out_bf16) {
    auto kern = fa2_sm70::gdn::volta_gdn_chunk_o<BT, DK, DV, 2, 4, __nv_bfloat16>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    kern<<<dim3(NT, H), 256, sh, st>>>(
        (const __half*)q.data_ptr(), (const __half*)k.data_ptr(), (const __half*)v.data_ptr(),
        (const __half*)h.data_ptr(), g.data_ptr<float>(), (__nv_bfloat16*)o.data_ptr(),
        chunk_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(), T, H, Hg, (float)scale);
  } else {
    auto kern = fa2_sm70::gdn::volta_gdn_chunk_o<BT, DK, DV, 2, 4, __half>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    kern<<<dim3(NT, H), 256, sh, st>>>(
        (const __half*)q.data_ptr(), (const __half*)k.data_ptr(), (const __half*)v.data_ptr(),
        (const __half*)h.data_ptr(), g.data_ptr<float>(), (__half*)o.data_ptr(),
        chunk_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(), T, H, Hg, (float)scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

// ------------------------------------------------------------------------------------------------
// БОЕВАЯ ТОЧКА ВХОДА: весь скан за один вызов. q,k -- УЖЕ fp16 (их пишет l2norm), v и beta -- bf16
// (конвертируются здесь слитым проходом), g -- fp32 (его пишет chunk_local_cumsum).
// ------------------------------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> gdn_chunk_scan(
    torch::Tensor q, torch::Tensor k, torch::Tensor v_in, torch::Tensor g, torch::Tensor beta_in,
    c10::optional<torch::Tensor> h0, torch::Tensor cu_seqlens, torch::Tensor chunk_indices,
    torch::Tensor chunk_offsets, double scale, bool out_bf16, bool output_final_state) {
  const c10::cuda::CUDAGuard guard(q.device());
  CHK(q, torch::kHalf, "q"); CHK(k, torch::kHalf, "k"); CHK(g, torch::kFloat, "g");
  TORCH_CHECK(v_in.is_cuda() && v_in.is_contiguous() && beta_in.is_contiguous(),
              "v/beta must be contiguous CUDA");
  const bool cast_in = (v_in.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(cast_in || v_in.scalar_type() == torch::kHalf, "v must be fp16 or bf16");
  TORCH_CHECK(beta_in.scalar_type() == v_in.scalar_type(), "beta must match v dtype");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt && chunk_indices.scalar_type() == torch::kInt &&
                  chunk_offsets.scalar_type() == torch::kInt,
              "cu_seqlens/chunk_indices/chunk_offsets must be int32");
  const int T = (int)q.size(-3), Hg = (int)q.size(-2), H = (int)v_in.size(-2);
  const int NT = (int)chunk_indices.size(0);
  auto st = at::cuda::getCurrentCUDAStream();

  // ЕСЛИ РАЗВЁРТКА УЖЕ fp16 -- НЕ КОПИРОВАТЬ. Безусловная конверсия "на всякий случай" стоила бы
  // здесь двух лишних проходов по 96 МБ на слой (0.6 мс) РОВНО НИ ЗА ЧТО. Ветка выбирается по
  // фактическому типу тензора, а не по флагу конфигурации.
  torch::Tensor v = v_in, beta = beta_in;
  if (cast_in) {
    v = torch::empty({1, T, H, DV}, torch::dtype(torch::kHalf).device(q.device()));
    beta = torch::empty({1, T, H}, torch::dtype(torch::kHalf).device(q.device()));
    const long nv = (long)T * H * DV, nb = (long)T * H;
    const int thr = 256;
    // Сетка ограничена сверху: ядро памяти-ограничено и от лишних блоков не ускоряется, а
    // хвостовой цикл по beta им бы только раздробился.
    int blocks = (int)((nv / 8 + thr - 1) / thr);
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;
    gdn_cast_in<<<blocks, thr, 0, st>>>((const __nv_bfloat16*)v_in.data_ptr(), (__half*)v.data_ptr(),
                                        (const __nv_bfloat16*)beta_in.data_ptr(),
                                        (__half*)beta.data_ptr(), nv, nb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto A = gdn_kkt(k, beta, g, cu_seqlens, chunk_indices, H);
  auto Ai = gdn_solve_tril(A, cu_seqlens, chunk_indices);
  auto wu = gdn_wu(k, v, beta, g, Ai, cu_seqlens, chunk_indices);
  // output_final_state проброшен, а не зашит в true: финальное состояние -- это N*H*DV*DK fp32,
  // то есть лишняя запись 6.3 МБ на слой там, где вызывающий его не просил. На 48 слоях это
  // 0.3 ГБ за чанк-проход -- мелочь, которую незачем платить, раз флаг и так есть в контракте.
  auto hv = gdn_h(k, std::get<0>(wu), std::get<1>(wu), g, h0, cu_seqlens, chunk_offsets, NT,
                  output_final_state);
  auto o = gdn_o(q, k, std::get<1>(hv), std::get<0>(hv), g, cu_seqlens, chunk_indices, scale,
                 out_bf16);
  return {o, std::get<2>(hv)};
}

// Явный признак присутствия: гейт врезки проверяет ИМЕННО его, а не факт импорта модуля.
int64_t gdn_scan_available() { return 1; }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_chunk_scan", &gdn_chunk_scan,
        "gated-delta-net chunk scan on Volta tensor cores (whole chain, sm_70)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("g"),
        pybind11::arg("beta"), pybind11::arg("initial_state"), pybind11::arg("cu_seqlens"),
        pybind11::arg("chunk_indices"), pybind11::arg("chunk_offsets"), pybind11::arg("scale"),
        pybind11::arg("out_bf16"), pybind11::arg("output_final_state") = true);
  m.def("gdn_kkt", &gdn_kkt, "beta*K*K^T внутри чанка (замена chunk_scaled_dot_kkt_fwd)");
  m.def("gdn_solve_tril", &gdn_solve_tril, "(I+A)^{-1} по чанкам (замена solve_tril)");
  m.def("gdn_wu", &gdn_wu, "w,u чанкового представления WY (замена recompute_w_u_fwd)");
  m.def("gdn_h", &gdn_h, "межчанковое состояние + v_new (замена chunk_gated_delta_rule_fwd_h)");
  m.def("gdn_o", &gdn_o, "выход скана (замена chunk_fwd_o)");
  m.def("gdn_scan_available", &gdn_scan_available, "признак присутствия для гейта врезки");
}
