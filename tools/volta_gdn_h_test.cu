// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Проверка и замер ядра межчанкового состояния gated-delta-net (volta_gdn_h.cuh) против ПРЯМОГО
// эталона на CPU. Эталон считается по определению в fp64, а НЕ вызовом другого ядра: иначе общая
// ошибка в понимании формулы прошла бы обе стороны сверки незамеченной.
//
// Эталон обязан воспроизводить ВСЕ ЧЕТЫРЕ округления до fp16 (состояние при записи h, оно же
// операндом GEMM1, v_new до гейта, vt после гейта). Если хоть одно пропустить, сверка будет мерить
// не то ядро: в этой рекуррентности округление состояния входит в САМ рекуррентный шаг.
//
// Сборка:
//   nvcc -arch=sm_70 -O3 -std=c++17 -I fa2_src/fmha_kernel/gemm -o gdnh tools/volta_gdn_h_test.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "../fa2_sm70/csrc/volta_gdn_h.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("ОТКАЗ %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static float h2f(__half h) { return __half2float(h); }

constexpr int BT = 64, DK = 128, DV = 128;

// Каталог для выгрузки входов И выходов: тот же набор чисел потом читает Triton, и обе стороны
// сверяются с ОДНИМ эталоном fp64. Без этого нельзя ответить на главный вопрос -- наша ошибка
// того же класса, что у образца, или хуже.
static const char* g_dump = nullptr;
template <typename T>
static void dump(const char* name, const std::vector<T>& v) {
  if (!g_dump) return;
  char path[512];
  snprintf(path, sizeof(path), "%s/%s", g_dump, name);
  FILE* f = fopen(path, "wb");
  if (!f) { printf("не открыть %s\n", path); exit(1); }
  fwrite(v.data(), sizeof(T), v.size(), f);
  fclose(f);
}

// ---- запуск одного варианта BV (шаблон, чтобы геометрия варпов считалась компилятором)
template <int BV, int WM1, int WN1, int WM2, int WN2, bool PIPE = true>
struct Runner {
  static constexpr int NTHR = WM1 * WN1 * 32;
  static void launch(const __half* k, const __half* w, const __half* u, const float* g,
                     const float* h0, __half* h, __half* vn, float* ht,
                     const int* cu, const int* coff, int N, int H, int HG, cudaStream_t s = 0) {
    auto kern = fa2_sm70::gdn::volta_gdn_chunk_h<BT, DK, DV, BV, WM1, WN1, WM2, WN2, PIPE>;
    static bool once = false;
    const size_t sh = fa2_sm70::gdn::gdn_h_smem<BT, DK, BV>();
    if (!once) {
      CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
      once = true;
    }
    dim3 grid(DV / BV, N * H);
    kern<<<grid, NTHR, sh, s>>>(k, w, u, g, h0, h, vn, ht, cu, coff, H, HG);
  }
  static size_t smem() { return fa2_sm70::gdn::gdn_h_smem<BT, DK, BV>(); }
};

// ---- ЭТАЛОН ПО ОПРЕДЕЛЕНИЮ (fp64 на CPU), с ЧЕТЫРЬМЯ округлениями до fp16 на своих местах
struct Ref {
  std::vector<double> h, vnew, ht;   // h: [NT][H][DV][DK], vnew: [T][H][DV], ht: [H][DV][DK]
};

static void reference(int T, int H, int HG, const std::vector<__half>& k,
                      const std::vector<__half>& w, const std::vector<__half>& u,
                      const std::vector<float>& g, const std::vector<float>& h0, Ref& out) {
  const int NT = (T + BT - 1) / BT;
  out.h.assign((size_t)NT * H * DV * DK, 0.0);
  out.vnew.assign((size_t)T * H * DV, 0.0);
  out.ht.assign((size_t)H * DV * DK, 0.0);
  std::vector<double> S((size_t)DV * DK);
  std::vector<double> Shat((size_t)DV * DK);
  std::vector<double> y((size_t)BT * DV);

  for (int hv = 0; hv < H; ++hv) {
    const int hk = hv / (H / HG);
    for (size_t i = 0; i < S.size(); ++i) S[i] = (double)h0[(size_t)hv * DV * DK + i];
    for (int c = 0; c < NT; ++c) {
      const int t0 = c * BT, L = std::min(BT, T - t0), last = t0 + L - 1;
      // (a) состояние на начало чанка -> fp16
      for (size_t i = 0; i < S.size(); ++i) {
        Shat[i] = (double)h2f(__float2half((float)S[i]));
        out.h[((size_t)c * H + hv) * DV * DK + i] = Shat[i];
      }
      // (b) y = u - w x Shat^T
      for (int i = 0; i < L; ++i)
        for (int v = 0; v < DV; ++v) {
          double acc = 0;
          for (int kk = 0; kk < DK; ++kk)
            acc += (double)h2f(w[((size_t)(t0 + i) * H + hv) * DK + kk]) * Shat[(size_t)v * DK + kk];
          y[(size_t)i * DV + v] = (double)h2f(u[((size_t)(t0 + i) * H + hv) * DV + v]) - acc;
        }
      // (c) v_new -- БЕЗ гейта
      for (int i = 0; i < L; ++i)
        for (int v = 0; v < DV; ++v)
          out.vnew[((size_t)(t0 + i) * H + hv) * DV + v] =
              (double)h2f(__float2half((float)y[(size_t)i * DV + v]));
      // (d) гейт и ВТОРОЕ округление;  (e) затухание и внешнее произведение
      const double gL = (double)g[(size_t)last * H + hv];
      for (size_t i = 0; i < S.size(); ++i) S[i] *= exp(gL);
      for (int i = 0; i < L; ++i) {
        const double gam = exp(gL - (double)g[(size_t)(t0 + i) * H + hv]);
        for (int v = 0; v < DV; ++v) {
          const double vt = (double)h2f(__float2half((float)(y[(size_t)i * DV + v] * gam)));
          if (vt == 0.0) continue;
          for (int kk = 0; kk < DK; ++kk)
            S[(size_t)v * DK + kk] +=
                (double)h2f(k[((size_t)(t0 + i) * HG + hk) * DK + kk]) * vt;
        }
      }
    }
    for (size_t i = 0; i < S.size(); ++i) out.ht[(size_t)hv * DV * DK + i] = S[i];
  }
}

// ДВЕ МЕТРИКИ, и обе нужны. relL2 -- та, которой меряет весь этот репозиторий (уровень fp16 = 2-3e-04);
// max/max -- худшая ячейка, нормированная на размах. Вторая по построению НЕ МОЖЕТ быть меньше одного
// fp16-ULP у самой крупной ячейки: у fp16 ULP относительно значения лежит в 4.9e-04..9.8e-04, поэтому
// РАСХОЖДЕНИЕ В ОДИН ШАГ ФОРМАТА на большой ячейке читается этой метрикой как ~1e-03. Это свойство
// метрики, а не ядра, и порог по ней брать нельзя -- порог берётся по relL2.
// Буферы выхода заполняются мусором ДО запуска, поэтому NaN считается ОТДЕЛЬНО: сравнение
// `e > emax` с NaN ложно, и непокрытая ячейка иначе прошла бы сверку незамеченной.
struct Err { double l2, mx; size_t nbad, nulp; };
static Err errstat(const double* ref, const double* got, size_t n) {
  double se = 0, sr = 0, emax = 0, rmax = 0; size_t nbad = 0, nulp = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!(got[i] == got[i])) { ++nbad; continue; }
    const double e = got[i] - ref[i];
    se += e * e; sr += ref[i] * ref[i];
    if (fabs(e) > emax) emax = fabs(e);
    if (fabs(ref[i]) > rmax) rmax = fabs(ref[i]);
  }
  // худшее расхождение В ШАГАХ ФОРМАТА, отсчитанных ОТ РАЗМАХА тензора: считать шаг от самой
  // ячейки нельзя -- у ячеек около нуля он вырождается и счётчик показывает тысячи
  const double ulp = ldexp(1.0, std::max(-24, (int)ilogb(rmax + 1e-30) - 10));
  nulp = (size_t)(emax / ulp + 0.5);
  return Err{nbad ? 1e30 : sqrt(se / (sr + 1e-30)), nbad ? 1e30 : emax / (rmax + 1e-30), nbad, nulp};
}
static Err report(const char* name, const std::vector<double>& ref, const std::vector<double>& got) {
  const Err e = errstat(ref.data(), got.data(), ref.size());
  printf("  %-8s relL2 %.2e   max/max %.2e   худшее %zu ULP fp16 (от размаха)   не записано %zu\n",
         name, e.l2, e.mx, e.nulp, e.nbad);
  return e;
}
template <typename T>
static std::vector<double> tod(const std::vector<T>& v);
template <>
std::vector<double> tod<__half>(const std::vector<__half>& v) {
  std::vector<double> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = (double)h2f(v[i]);
  return o;
}
template <>
std::vector<double> tod<float>(const std::vector<float>& v) {
  std::vector<double> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = (double)v[i];
  return o;
}

// ---------------------------------------------------------------------------------------------
struct Buffers {
  int T, H, HG, NT;
  std::vector<__half> k, w, u;
  std::vector<float> g, h0;
  __half *dk, *dw, *du, *dh, *dvn;
  float *dg, *dh0, *dht;
  int *dcu, *dcoff;
  void alloc(int T_, int H_, int HG_, unsigned seed) {
    T = T_; H = H_; HG = HG_; NT = (T + BT - 1) / BT;
    k.resize((size_t)T * HG * DK); w.resize((size_t)T * H * DK); u.resize((size_t)T * H * DV);
    g.resize((size_t)T * H); h0.resize((size_t)H * DV * DK);
    srand(seed);
    auto rnd = [] { return (float)rand() / RAND_MAX * 0.4f - 0.2f; };
    for (auto& x : k) x = __float2half(rnd());
    for (auto& x : w) x = __float2half(rnd());
    for (auto& x : u) x = __float2half(rnd());
    for (auto& x : h0) x = rnd();
    // g -- ЧАНКОВЫЙ накопленный логарифм гейта: внутри чанка убывает от 0, на границе чанка
    // обнуляется (ровно так его строит chunk_local_cumsum). Значит gL - g_i <= 0 всегда.
    for (int t = 0; t < T; ++t)
      for (int hh = 0; hh < H; ++hh)
        g[(size_t)t * H + hh] = -(0.004f + 0.0015f * hh) * (float)((t % BT) + 1);
    std::vector<int> cu = {0, T}, coff = {0, NT};
    CK(cudaMalloc(&dk, k.size() * 2));   CK(cudaMalloc(&dw, w.size() * 2));
    CK(cudaMalloc(&du, u.size() * 2));   CK(cudaMalloc(&dvn, u.size() * 2));
    CK(cudaMalloc(&dh, (size_t)NT * H * DV * DK * 2));
    CK(cudaMalloc(&dg, g.size() * 4));   CK(cudaMalloc(&dh0, h0.size() * 4));
    CK(cudaMalloc(&dht, h0.size() * 4));
    CK(cudaMalloc(&dcu, 8));             CK(cudaMalloc(&dcoff, 8));
    CK(cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(du, u.data(), u.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dh0, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcu, cu.data(), 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcoff, coff.data(), 8, cudaMemcpyHostToDevice));
  }
  void free_all() {
    cudaFree(dk); cudaFree(dw); cudaFree(du); cudaFree(dvn); cudaFree(dh);
    cudaFree(dg); cudaFree(dh0); cudaFree(dht); cudaFree(dcu); cudaFree(dcoff);
  }
};

template <int BV, int WM1, int WN1, int WM2, int WN2, bool PIPE = true>
static void verify(int T, int H, int HG) {
  Buffers b; b.alloc(T, H, HG, 7);
  CK(cudaMemset(b.dh, 0xff, (size_t)b.NT * H * DV * DK * 2));   // мусор: ловим непокрытые ячейки
  CK(cudaMemset(b.dvn, 0xff, b.u.size() * 2));
  Runner<BV, WM1, WN1, WM2, WN2, PIPE>::launch(b.dk, b.dw, b.du, b.dg, b.dh0, b.dh, b.dvn, b.dht,
                                         b.dcu, b.dcoff, 1, H, HG);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<__half> h((size_t)b.NT * H * DV * DK), vn(b.u.size());
  std::vector<float> ht(b.h0.size());
  CK(cudaMemcpy(h.data(), b.dh, h.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vn.data(), b.dvn, vn.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ht.data(), b.dht, ht.size() * 4, cudaMemcpyDeviceToHost));
  Ref r; reference(T, H, HG, b.k, b.w, b.u, b.g, b.h0, r);
  printf("T=%d H=%d HG=%d NT=%d BV=%d варпов %d  (разделяемой %.1f КБ)\n", T, H, HG, b.NT, BV,
         WM1 * WN1, Runner<BV, WM1, WN1, WM2, WN2, PIPE>::smem() / 1024.0);
  dump("k.bin", b.k); dump("w.bin", b.w); dump("u.bin", b.u);
  dump("g.bin", b.g); dump("h0.bin", b.h0);
  dump("h.bin", h); dump("vnew.bin", vn); dump("ht.bin", ht);
  const std::vector<double> gh = tod(h), gv = tod(vn), gt = tod(ht);
  const Err e1 = report("h", r.h, gh);
  const Err e2 = report("v_new", r.vnew, gv);
  const Err e3 = report("ht", r.ht, gt);
  // ПРОФИЛЬ ПО ЧАНКАМ И ПОЧЕМУ ГОДНОСТЬ ПРОВЕРЯЕТСЯ ИМЕННО ТАК.
  // Состояние округляется до fp16 НА КАЖДОМ чанке и тут же подаётся обратно в рекуррентность,
  // поэтому расхождение с эталоном fp64 РАСТЁТ с числом чанков у ЛЮБОЙ верной реализации -- в том
  // числе у трионовской (замерено: на этих же 8 чанках Triton даёт 2.5e-04 против нашего 3.1e-04
  // по одному и тому же эталону). Значит порог на итоговую relL2 -- величина, зависящая от длины,
  // и брать его константой значило бы подгонять число под ответ. Проверяются ДВА инварианта,
  // которые от длины НЕ зависят:
  //   1. чанк 0: h[0] == fp16(h0) ПО ПОСТРОЕНИЮ у обеих сторон -> расхождение обязано быть РОВНО 0;
  //   2. ПРИРОСТ relL2 за чанк не превосходит единичного округления fp16 (RMS 2^-11/sqrt(3) =
  //      2.82e-04): больше одного округления состояния за чанк в задаче нет, и превышение означало
  //      бы, что ошибку вносит НЕ формат, а дефект.
  // Дефект раскладки/формулы этих инвариантов не проходит: он даёт ошибку с ПЕРВОГО чанка.
  printf("  профиль h по чанкам (relL2): ");
  double prev = 0, dmax = 0, c0 = 0;
  for (int c = 0; c < b.NT; ++c) {
    const size_t off = (size_t)c * H * DV * DK, n = (size_t)H * DV * DK;
    const double v = errstat(r.h.data() + off, gh.data() + off, n).l2;
    if (c == 0) c0 = v;
    if (v - prev > dmax) dmax = v - prev;
    prev = v;
    if (c < 10) printf("%.1e ", v);
  }
  printf("\n  чанк 0 = %.1e (обязан быть 0)   макс. прирост за чанк = %.2e (предел fp16 2.82e-04)\n",
         c0, dmax);
  const bool ok = (c0 == 0.0) && (dmax < 2.82e-4) && e1.nbad == 0 && e2.nbad == 0 && e3.nbad == 0;
  printf("  ИТОГ %s\n\n", ok ? "ГОДЕН" : "НЕ ГОДЕН");
  b.free_all();
}

template <int BV, int WM1, int WN1, int WM2, int WN2, bool PIPE = true>
static void bench(Buffers& b, int reps) {
  cudaEvent_t a, c; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&c));
  auto go = [&] {
    Runner<BV, WM1, WN1, WM2, WN2, PIPE>::launch(b.dk, b.dw, b.du, b.dg, b.dh0, b.dh, b.dvn, b.dht,
                                           b.dcu, b.dcoff, 1, b.H, b.HG);
  };
  for (int i = 0; i < 3; ++i) go();
  CK(cudaDeviceSynchronize());
  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    CK(cudaEventRecord(a)); go(); CK(cudaEventRecord(c)); CK(cudaEventSynchronize(c));
    CK(cudaEventElapsedTime(&ms[i], a, c));
  }
  // усечённое среднее: отбрасываем по четверти с обоих концов
  for (int i = 0; i < reps; ++i)
    for (int j = i + 1; j < reps; ++j) if (ms[j] < ms[i]) { float t = ms[i]; ms[i] = ms[j]; ms[j] = t; }
  double s = 0; int n = 0;
  for (int i = reps / 4; i < reps - reps / 4; ++i) { s += ms[i]; ++n; }
  const double t = s / n;
  const double fl = 2.0 * 2.0 * (double)b.NT * b.H * BT * DK * DV;   // два умножения на пару
  printf("  BV=%-3d варпов %-2d конвейер %d блоков %-3d  %.3f мс   %.2f TFLOP/s   (разделяемой %.1f КБ)\n",
         BV, WM1 * WN1, (int)PIPE, (DV / BV) * b.H, t, fl / (t * 1e-3) / 1e12,
         fa2_sm70::gdn::gdn_h_smem<BT, DK, BV>() / 1024.0);
  CK(cudaGetLastError());
}

int main(int argc, char** argv) {
  const int Tv = argc > 1 ? atoi(argv[1]) : 512;
  const char* dd = argc > 2 ? argv[2] : nullptr;   // выгрузка для перекрёстной сверки с Triton

  printf("=== КОРРЕКТНОСТЬ (эталон fp64 на CPU по определению) ===\n");
  verify<64, 2, 4, 2, 4>(64, 6, 2);          // ОДИН чанк: h обязан совпасть побитово
  g_dump = dd;                               // выгружается ИМЕННО рабочий случай
  verify<64, 2, 4, 2, 4>(Tv, 6, 2);          // рабочая точка, полные чанки
  g_dump = nullptr;
  verify<64, 2, 4, 2, 4>(200, 6, 2);         // ХВОСТОВОЙ чанк: 200 = 3*64 + 8
  verify<32, 4, 2, 2, 4>(Tv, 6, 2);          // мелкий срез
  verify<64, 4, 4, 4, 4>(Tv, 6, 2);          // 16 варпов на тех же данных
  verify<64, 2, 4, 2, 4>(320, 3, 3);         // H == HG (отображение голов вырождается)

  printf("=== ЗАМЕР НА БОЕВОЙ ФОРМЕ (Qwen3.5, TP=2, на карту): T=15680 H=24 HG=8 ===\n");
  Buffers b; b.alloc(15680, 24, 8, 11);
  bench<64, 2, 4, 2, 4, false>(b, 20);      // A/B конвейера: LDG и STS СЛИТНО
  bench<64, 2, 4, 2, 4, true >(b, 20);      // ... и РАЗНЕСЕНЫ двумя умножениями
  bench<64, 4, 4, 4, 4, false>(b, 20);
  bench<64, 4, 4, 4, 4, true >(b, 20);
  bench<32, 4, 2, 2, 4, false>(b, 20);
  bench<32, 4, 2, 2, 4, true >(b, 20);
  b.free_all();

  // ЧЕМ ОГРАНИЧЕНЫ. Работа НА БЛОК от H не зависит, значит при 48 блоках на 80 SM время не должно
  // меняться при росте H до 40 (80 блоков) -- если предел в блоке. Если время растёт -- предел
  // общий, то есть полоса памяти. Один прогон отвечает на вопрос, ради которого иначе нужен ncu.
  printf("=== ЧЕМ ОГРАНИЧЕНЫ: та же работа на блок, разное число блоков (T=8192, BV=64) ===\n");
  for (int hh : {24, 40, 48}) {
    Buffers c; c.alloc(8192, hh, 8, 11);
    printf("  H=%-2d ", hh);
    bench<64, 2, 4, 2, 4, true>(c, 20);
    c.free_all();
  }
  return 0;
}
