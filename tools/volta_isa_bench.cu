// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// СТЕНД ХАРАКТЕРИЗАЦИИ ИНСТРУКЦИЙ sm_70: пропускная способность и задержка каждой,
// чтобы строить макросы из ЗАМЕРЕННЫХ кирпичей, а не из предположений.
//
// Меряются две разные величины, и путать их нельзя:
//   ПРОПУСКНАЯ  -- восемь НЕЗАВИСИМЫХ цепочек, задержка спрятана; показывает, сколько таких
//                  инструкций машина проглатывает в такт (потолок конвейера).
//   ЗАДЕРЖКА    -- ОДНА зависимая цепочка; показывает, сколько тактов ждать результата.
// Для планирования важны обе: пропускная задаёт бюджет выдачи, задержка -- длину критического пути.
//
// Единица вывода -- инструкций на такт на SM (IPC) и такты задержки. Опора: HMMA.884 занимает
// подраздел 2 такта, выдача 1 инструкция/такт/планировщик, поэтому бюджет нетензорной работы
// равен ОДНОЙ инструкции на HMMA.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

#define ITERS 4096
#define UNROLL 16

// ---- пропускная: восемь независимых цепочек ------------------------------------------------------
#define THROUGHPUT_KERNEL(NAME, TYPE, INIT, EXPR)                                                   \
__global__ void tp_##NAME(TYPE* out, int iters) {                                                   \
  TYPE a0=INIT, a1=INIT, a2=INIT, a3=INIT, a4=INIT, a5=INIT, a6=INIT, a7=INIT;                      \
  TYPE b = (TYPE)(threadIdx.x | 1);                                                                 \
  for (int i = 0; i < iters; ++i) {                                                                 \
    _Pragma("unroll")                                                                               \
    for (int u = 0; u < UNROLL; ++u) {                                                              \
      { TYPE& a = a0; a = (EXPR); } { TYPE& a = a1; a = (EXPR); }                                   \
      { TYPE& a = a2; a = (EXPR); } { TYPE& a = a3; a = (EXPR); }                                   \
      { TYPE& a = a4; a = (EXPR); } { TYPE& a = a5; a = (EXPR); }                                   \
      { TYPE& a = a6; a = (EXPR); } { TYPE& a = a7; a = (EXPR); }                                   \
    }                                                                                               \
  }                                                                                                 \
  out[threadIdx.x] = a0+a1+a2+a3+a4+a5+a6+a7;                                                       \
}
// ---- задержка: одна зависимая цепочка ------------------------------------------------------------
#define LATENCY_KERNEL(NAME, TYPE, INIT, EXPR)                                                      \
__global__ void lat_##NAME(TYPE* out, int iters) {                                                  \
  TYPE a = INIT; TYPE b = (TYPE)(threadIdx.x | 1);                                                  \
  for (int i = 0; i < iters; ++i) {                                                                 \
    _Pragma("unroll")                                                                               \
    for (int u = 0; u < UNROLL * 8; ++u) { a = (EXPR); }                                            \
  }                                                                                                 \
  out[threadIdx.x] = a;                                                                             \
}
#define BOTH(NAME, TYPE, INIT, EXPR) THROUGHPUT_KERNEL(NAME,TYPE,INIT,EXPR) LATENCY_KERNEL(NAME,TYPE,INIT,EXPR)

// --- целочисленные -------------------------------------------------------------------------------
// ЧЕРЕЗ asm, А НЕ ЧЕРЕЗ ВЫРАЖЕНИЕ. Записанные выражением, IADD/LOP3/SHF были свёрнуты компилятором
// (15576 / 2832 / 29.7 IPC -- физически невозможно, потолок 64). Инструкция, выданная asm, не
// сворачивается: компилятору нечего доказывать про её результат.
#define ASM1(NAME, TXT) \
__device__ __forceinline__ uint32_t op_##NAME(uint32_t a, uint32_t b, uint32_t c) { \
  uint32_t d; asm(TXT : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
ASM1(iaddA,  "add.u32 %0, %1, %2;")
ASM1(lop3A,  "lop3.b32 %0, %1, %2, %3, 0xE8;")
ASM1(shfA,   "shf.l.wrap.b32 %0, %1, %1, 3;")
ASM1(bmskA,  "bmsk.clamp.b32 %0, %1, %2;")
ASM1(fnsA,   "fns.b32 %0, %1, %2, 3;")
ASM1(bfeA,   "bfe.u32 %0, %1, 4, 8;")
ASM1(prmtrc8,"prmt.rc8.b32 %0, %1, %2, %3;")
ASM1(prmteclA,"prmt.ecl.b32 %0, %1, %2, %3;")
ASM1(szextA, "szext.clamp.s32 %0, %1, %2;")
ASM1(dp2aA,  "dp2a.lo.u32.u32 %0, %1, %2, %3;")
BOTH(iadd_asm,  uint32_t, 1u, op_iaddA(a, b, b))
BOTH(lop3_asm,  uint32_t, 1u, op_lop3A(a, b, 0x55u))
BOTH(shf_asm,   uint32_t, 1u, op_shfA(a, b, b))
BOTH(bmsk,      uint32_t, 1u, op_bmskA(a, b, b))
BOTH(fns,       uint32_t, 1u, op_fnsA(a | 0x10101u, b, b))
BOTH(bfe,       uint32_t, 1u, op_bfeA(a, b, b))
BOTH(prmt_rc8,  uint32_t, 1u, op_prmtrc8(a, b, b))
BOTH(prmt_ecl,  uint32_t, 1u, op_prmteclA(a, b, b))
BOTH(szext,     uint32_t, 1u, op_szextA(a, b, b))
BOTH(dp2a,      uint32_t, 1u, op_dp2aA(a, b, a))

// --- ВИДЕО-СЕМЕЙСТВО (SIMD по байтам/полусловам), всё через asm ---------------------------------
ASM1(vadd4A,   "vadd4.u32.u32.u32 %0, %1, %2, %3;")
ASM1(vsub4A,   "vsub4.u32.u32.u32 %0, %1, %2, %3;")
ASM1(vmin4A,   "vmin4.u32.u32.u32 %0, %1, %2, %3;")
ASM1(vmax4A,   "vmax4.u32.u32.u32 %0, %1, %2, %3;")
ASM1(vavrg4A,  "vavrg4.u32.u32.u32 %0, %1, %2, %3;")
ASM1(vabsd4A,  "vabsdiff4.u32.u32.u32 %0, %1, %2, %3;")
ASM1(vset4A,   "vset4.u32.u32.lt %0, %1, %2, %3;")
ASM1(vadd2A,   "vadd2.u32.u32.u32 %0, %1, %2, %3;")
ASM1(vmax2A,   "vmax2.u32.u32.u32 %0, %1, %2, %3;")
ASM1(vabsd2A,  "vabsdiff2.u32.u32.u32 %0, %1, %2, %3;")
ASM1(vsetA,    "vset.u32.u32.lt %0, %1, %2, %3;")
ASM1(vshlA,    "vshl.u32.u32.u32.clamp %0, %1, %2, %3;")
ASM1(vmadA,    "vmad.u32.u32.u32 %0, %1, %2, %3;")
BOTH(vadd4x,  uint32_t, 1u, op_vadd4A(a,b,b))
BOTH(vsub4x,  uint32_t, 1u, op_vsub4A(a,b,b))
BOTH(vmin4x,  uint32_t, 1u, op_vmin4A(a,b,b))
BOTH(vmax4x,  uint32_t, 1u, op_vmax4A(a,b,b))
BOTH(vavrg4x, uint32_t, 1u, op_vavrg4A(a,b,b))
BOTH(vabsd4x, uint32_t, 1u, op_vabsd4A(a,b,b))
BOTH(vset4x,  uint32_t, 1u, op_vset4A(a,b,b))
BOTH(vadd2x,  uint32_t, 1u, op_vadd2A(a,b,b))
BOTH(vmax2x,  uint32_t, 1u, op_vmax2A(a,b,b))
BOTH(vabsd2x, uint32_t, 1u, op_vabsd2A(a,b,b))
BOTH(vsetx,   uint32_t, 1u, op_vsetA(a,b,b))
BOTH(vshlx,   uint32_t, 1u, op_vshlA(a,b,b))
BOTH(vmadx,   uint32_t, 1u, op_vmadA(a,b,b))
BOTH(iadd,   uint32_t, 1u,  a + b)
BOTH(imad,   uint32_t, 1u,  a * b + 3u)
BOTH(lop3,   uint32_t, 1u,  (a ^ b) | 0x55u)
BOTH(shf,    uint32_t, 1u,  (a << 3) | (a >> 29))
BOTH(prmt,   uint32_t, 1u,  __byte_perm(a, b, 0x4140))
BOTH(popc,   uint32_t, 1u,  __popc(a) + b)
BOTH(brev,   uint32_t, 1u,  __brev(a) ^ b)
BOTH(flo,    uint32_t, 1u,  (uint32_t)__clz(a | 1u) + b)
BOTH(vadd4,  uint32_t, 1u,  __vadd4(a, b))
BOTH(vmax4,  uint32_t, 1u,  __vmaxu4(a, b))
BOTH(vabsd4, uint32_t, 1u,  __vabsdiffu4(a, b))
BOTH(dp4a,   uint32_t, 1u,  (uint32_t)__dp4a((int)a, (int)b, (int)a))
// --- одинарная точность --------------------------------------------------------------------------
BOTH(fadd,   float, 1.f, a + 1.000001f)
BOTH(ffma,   float, 1.f, fmaf(a, 1.000001f, 0.5f))
BOTH(fmul,   float, 1.f, a * 1.000001f)
BOTH(frcp,   float, 1.f, __frcp_rn(a + 1.f))
BOTH(fexp2,  float, 1.f, exp2f(a * 1e-6f))
BOTH(i2f,    float, 1.f, __int2float_rn((int)(a * 2.f)))
// --- половинная точность (парная) ----------------------------------------------------------------
__global__ void tp_hfma2(__half2* out, int iters) {
  __half2 a0=__float2half2_rn(1.f),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
  __half2 b=__float2half2_rn(1.0001f);
  for (int i=0;i<iters;++i){
#pragma unroll
    for (int u=0;u<UNROLL;++u){ a0=__hfma2(a0,b,b); a1=__hfma2(a1,b,b); a2=__hfma2(a2,b,b); a3=__hfma2(a3,b,b);
                                a4=__hfma2(a4,b,b); a5=__hfma2(a5,b,b); a6=__hfma2(a6,b,b); a7=__hfma2(a7,b,b); } }
  out[threadIdx.x]=__hadd2(__hadd2(__hadd2(a0,a1),__hadd2(a2,a3)),__hadd2(__hadd2(a4,a5),__hadd2(a6,a7)));
}
__global__ void lat_hfma2(__half2* out, int iters) {
  __half2 a=__float2half2_rn(1.f), b=__float2half2_rn(1.0001f);
  for (int i=0;i<iters;++i){
#pragma unroll
    for (int u=0;u<UNROLL*8;++u) a=__hfma2(a,b,b); }
  out[threadIdx.x]=a;
}
// --- обмен между полосами (кроссбар регистров, МИМО MIO) -----------------------------------------
BOTH(shfl,   uint32_t, 1u,  __shfl_sync(0xffffffffu, a, (int)((a + b) & 31)))
BOTH(shflxor,uint32_t, 1u,  __shfl_xor_sync(0xffffffffu, a, 4) + b)
BOTH(ballot, uint32_t, 1u,  __ballot_sync(0xffffffffu, (a & 1) != 0) ^ b)
// --- разделяемая память ---------------------------------------------------------------------------
__shared__ uint32_t smem_pool[4096];
#define SMEM_KERNEL(NAME, LOAD)                                                                     \
__global__ void tp_##NAME(uint32_t* out, int iters) {                                               \
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) smem_pool[i] = i;                            \
  __syncthreads();                                                                                  \
  uint32_t acc = 0; const int base = (threadIdx.x * 4) & 4095;                                      \
  for (int i = 0; i < iters; ++i) {                                                                 \
    _Pragma("unroll")                                                                               \
    for (int u = 0; u < UNROLL; ++u) { LOAD; }                                                      \
  }                                                                                                 \
  out[threadIdx.x] = acc;                                                                           \
}
SMEM_KERNEL(lds32,  acc += smem_pool[(base + u * 32) & 4095])
SMEM_KERNEL(lds64,  { uint2 v = *reinterpret_cast<uint2*>(&smem_pool[(base + u * 32) & 4088]); acc += v.x + v.y; })
SMEM_KERNEL(lds128, { uint4 v = *reinterpret_cast<uint4*>(&smem_pool[(base + u * 32) & 4080]); acc += v.x + v.w; })

static const char* PIPE(const char* n){ return n; }
float run(void(*k)(uint32_t*,int), uint32_t* d, int iters, int blocks, int threads) {
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  k<<<blocks,threads>>>(d,8); cudaDeviceSynchronize();
  cudaEventRecord(a); k<<<blocks,threads>>>(d,iters); cudaEventRecord(b);
  cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b); return ms;
}
int main(){
  cudaSetDevice(0);
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  const int SM = p.multiProcessorCount, TH = 256, BL = SM * 4;
  const double clk = p.clockRate * 1e3;                     // Гц
  uint32_t* d; cudaMalloc(&d, 4096*4);
  printf("V100: %d SM, %.0f МГц. Замер: %d блоков x %d нитей.\n", SM, clk/1e6, BL, TH);
  printf("%-10s %12s %12s   %s\n", "инстр", "IPC на SM", "задержка", "примечание");
#define REPORT(NAME, NOTE)                                                                           \
  { double n = (double)ITERS*UNROLL*8*TH*BL;                                                         \
    float t = run((void(*)(uint32_t*,int))tp_##NAME, d, ITERS, BL, TH);                              \
    float l = run((void(*)(uint32_t*,int))lat_##NAME, d, ITERS, BL, TH);                             \
    double ipc = n / (t*1e-3) / clk / SM;                                                            \
    double lat = (l*1e-3) * clk / ((double)ITERS*UNROLL*8);                                          \
    printf("%-10s %12.2f %12.1f   %s\n", #NAME, ipc, lat, NOTE); }
  printf("  -- ВИДЕО-СЕМЕЙСТВО (4 байта или 2 полуслова за инструкцию) --\n");
  REPORT(vadd4x,  "сложение 4 байт")
  REPORT(vsub4x,  "вычитание 4 байт")
  REPORT(vmin4x,  "минимум 4 байт")
  REPORT(vmax4x,  "максимум 4 байт")
  REPORT(vavrg4x, "среднее 4 байт")
  REPORT(vabsd4x, "|a-b| 4 байт")
  REPORT(vset4x,  "СРАВНЕНИЕ 4 байт -> маска")
  REPORT(vadd2x,  "сложение 2 полуслов")
  REPORT(vmax2x,  "максимум 2 полуслов")
  REPORT(vabsd2x, "|a-b| 2 полуслов")
  REPORT(vsetx,   "сравнение 32 бит -> маска")
  REPORT(vshlx,   "сдвиг с зажимом")
  REPORT(vmadx,   "умножение-сложение видео")
  printf("  -- через asm (компилятор не сворачивает) --\n");
  REPORT(iadd_asm, "СЛОЖЕНИЕ, выданное asm")
  REPORT(lop3_asm, "LOP3 с ЛУТом: ЛЮБАЯ функция трёх входов")
  REPORT(shf_asm,  "воронкообразный сдвиг")
  REPORT(bmsk,     "битовая маска из (база,длина) -- нет обёртки в CUDA C")
  REPORT(fns,      "N-Й УСТАНОВЛЕННЫЙ БИТ -- компакция без префикс-суммы")
  REPORT(bfe,      "извлечение битового поля")
  REPORT(prmt_rc8, "prmt.rc8: повтор байта восемь раз")
  REPORT(prmt_ecl, "prmt.ecl: зажим краёв (репликация знака)")
  REPORT(szext,    "расширение знака")
  REPORT(dp2a,     "2-путевое скалярное произведение")
  printf("  -- через выражение (часть свёрнута, помечено) --\n");
  REPORT(iadd,   "целочисленное сложение (СВЁРНУТО)")
  REPORT(imad,   "целочисленное умножение-сложение (адресная арифметика!)")
  REPORT(lop3,   "логика (XOR/OR/AND сливаются в одну)")
  REPORT(shf,    "сдвиг")
  REPORT(prmt,   "ПЕРЕСТАНОВКА БАЙТ -- наш разворот int8")
  REPORT(popc,   "подсчёт единиц")
  REPORT(brev,   "разворот бит")
  REPORT(flo,    "поиск старшего бита")
  REPORT(vadd4,  "SIMD: четыре байта за инструкцию")
  REPORT(vmax4,  "SIMD: максимум по четырём байтам")
  REPORT(vabsd4, "SIMD: |a-b| по четырём байтам")
  REPORT(dp4a,   "int8 скалярное произведение на ALU")
  REPORT(fadd,   "fp32 сложение")
  REPORT(ffma,   "fp32 умножение-сложение")
  REPORT(fmul,   "fp32 умножение")
  REPORT(frcp,   "обратное (SFU)")
  REPORT(fexp2,  "экспонента по основанию 2 (SFU) -- софтмакс!")
  REPORT(i2f,    "конверсия целое<->fp32 (F2I+I2F)")
  REPORT(shfl,   "обмен полос: ПРОИЗВОЛЬНЫЙ индекс")
  REPORT(shflxor,"обмен полос: XOR-маска (дешевле?)")
  REPORT(ballot, "голосование по варпу")
#undef REPORT
  { double n=(double)ITERS*UNROLL*8*TH*BL;
    float t=run((void(*)(uint32_t*,int))(void*)tp_hfma2,d,ITERS,BL,TH);
    float l=run((void(*)(uint32_t*,int))(void*)lat_hfma2,d,ITERS,BL,TH);
    printf("%-10s %12.2f %12.1f   %s\n","hfma2", n/(t*1e-3)/clk/SM, (l*1e-3)*clk/((double)ITERS*UNROLL*8),
           "fp16 ПАРНОЕ умножение-сложение (2 значения за инструкцию)"); }
  { double n=(double)ITERS*UNROLL*TH*BL;
    for (int i=0;i<3;++i){
      void(*k[3])(uint32_t*,int)={tp_lds32,tp_lds64,tp_lds128};
      const char* nm[3]={"lds32","lds64","lds128"};
      const char* nt[3]={"4 байта","8 байт","16 байт -- та же цена?"};
      float t=run(k[i],d,ITERS,BL,TH);
      printf("%-10s %12.2f %12s   %s\n",nm[i], n/(t*1e-3)/clk/SM, "-", nt[i]); } }
  return 0;
}
