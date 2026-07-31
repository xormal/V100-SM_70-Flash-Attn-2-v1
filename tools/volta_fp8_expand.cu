// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// МОЖНО ЛИ ЗАВЕЗТИ e4m3 И e5m2 В БАЙТОВЫЙ ПУТЬ ЗА НОЛЬ ЦЕНЫ.
//
// Мерим не "скорость конверсии" саму по себе, а то единственное, что решает: ВЛЕЗАЕТ ЛИ разворот в
// свободные слоты выдачи рядом с HMMA. По §4p на одну HMMA свободны ОДНА fp32 и ОДНА int32; по §4bb
// у отгруженного байтового ядра уже занято PRMT 0.512 + LOP3 0.447 + IADD3 0.488 + IMAD 1.381 на
// HMMA. Поэтому вопрос ставится так: сколько инструкций на значение стоит каждый разворот, и что
// делает с тензорной полосой добавление его В ЦЕПЬ (а не в свободный слот -- §4r показал, что это
// разные вещи).
//
// ТРИ РАЗВОРОТА, все на 4 байта из одного 32-битного слова в два регистра по два fp16:
//
//   int8 (наш нынешний): значение кодируется ТОЧНО в мантиссу fp16, 0x6400|u == 1024+u.
//       Одна prmt на два значения: берём байт данных и константный 0x64. Смещение 1024 снимается
//       свёрткой в ранг-1 в эпилоге, то есть ВНЕ цепи.
//
//   e5m2: смещение порядка 15 -- ТО ЖЕ, ЧТО У fp16, и поля идут в том же порядке. Значит e5m2 это
//       РОВНО СТАРШИЙ БАЙТ fp16: fp16_bits == byte << 8, побитово, включая поддённормальные
//       (E=0: M/4*2^-14 в обоих) и Inf/NaN (E=31 в обоих). Разворот -- одна prmt, и НИКАКОЙ
//       свёртки смещения не нужно вовсе.
//
//   e4m3: смещение порядка 7 против 15, поэтому нужен пересчёт порядка. Побитово точный приём:
//       (b & 0x7F) << 7, прочитанное как fp16, даёт РОВНО значение/256 -- и для нормальных
//       (2^(E-15) против 2^(E-7)), и для поддённормальных (2^-14*M/4 против 2^-6*M/8): отношение
//       в обоих случаях ровно 256. Множитель 256 -- КОНСТАНТА, поэтому сворачивается в уже
//       существующее умножение на масштаб софтмакса и стоит НОЛЬ. Остаётся собственно перекладка
//       бит, и она НЕ сводится к одной prmt: знак едет на 8 позиций, а остальные семь бит -- на 7,
//       то есть между полями образуется РАЗРЫВ, которого перестановка байт не делает.
//       Минимум: prmt (byte<<8) + сдвиг вправо на 1 (byte<<7) + lop3, забирающий бит знака из
//       первого, остальное из второго. Три инструкции на два значения.
//
// Сборка: nvcc -arch=sm_70 -O3 -o fp8x tools/volta_fp8_expand.cu && ./fp8x
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>

#define NIT 4096
#define MMA(D, A, B) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "        \
  "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"                  \
  : "+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
  : "r"(A[0]),"r"(A[1]),"r"(B[0]),"r"(B[1]))

// --- РАЗВОРОТЫ. Каждый: 32 бита (4 байта) -> два регистра по два fp16. ---------------------------
__device__ __forceinline__ void exp_i8(uint32_t x, uint32_t& lo, uint32_t& hi) {
    // байт данных в младший байт полуслова, константа 0x64 -- в старший
    asm volatile("prmt.b32 %0, %1, 0x64646464, 0x5140;" : "=r"(lo) : "r"(x));
    asm volatile("prmt.b32 %0, %1, 0x64646464, 0x7362;" : "=r"(hi) : "r"(x));
}
__device__ __forceinline__ void exp_e5m2(uint32_t x, uint32_t& lo, uint32_t& hi) {
    // fp16 = byte << 8: байт данных в СТАРШИЙ байт полуслова, ноль -- в младший
    asm volatile("prmt.b32 %0, %1, 0, 0x1404;" : "=r"(lo) : "r"(x));
    asm volatile("prmt.b32 %0, %1, 0, 0x3626;" : "=r"(hi) : "r"(x));
}
// ЧЕТЫРЕ инструкции на два значения, и меньше НЕ ВЫХОДИТ -- вот почему. После сдвига вправо на 1
// бит знака садится в позицию 14, а это СТАРШИЙ БИТ ПОРЯДКА fp16, который обязан быть нулём (поле
// порядка должно равняться четырёхбитному E, дополненному нулём до пяти). Обнулить его одной lop3
// вместе со сборкой нельзя: lop3 применяет ОДНУ функцию ко всем разрядам, различая их только по
// биту константы, а разряду 14 нужно поведение, отличное и от разряда 15 (взять знак), и от
// разрядов 0-13 (взять данные) -- третьего варианта одна константа не даёт. Отсюда отдельная маска.
__device__ __forceinline__ void exp_e4m3(uint32_t x, uint32_t& lo, uint32_t& hi) {
    uint32_t p8, p7;
    asm volatile("prmt.b32 %0, %1, 0, 0x1404;" : "=r"(p8) : "r"(x));         // {b1<<8, b0<<8}
    asm volatile("shr.b32 %0, %1, 1;" : "=r"(p7) : "r"(p8));                 // {b1<<7, b0<<7}
    asm volatile("and.b32 %0, %0, 0x3FFF3FFF;" : "+r"(p7));                  // снять знак с 14-го
    asm volatile("lop3.b32 %0, %1, %2, 0x80008000, 0xE4;" : "=r"(lo) : "r"(p8), "r"(p7));
    asm volatile("prmt.b32 %0, %1, 0, 0x3626;" : "=r"(p8) : "r"(x));
    asm volatile("shr.b32 %0, %1, 1;" : "=r"(p7) : "r"(p8));
    asm volatile("and.b32 %0, %0, 0x3FFF3FFF;" : "+r"(p7));
    asm volatile("lop3.b32 %0, %1, %2, 0x80008000, 0xE4;" : "=r"(hi) : "r"(p8), "r"(p7));
}

// ПРОВЕРКА ЗАПИСАННОГО В §4k: константы на ОДНОРОДНОМ адресе почти бесплатны (13.27 против 7.27 у
// LDS, в смеси +1.5%). Соблазн очевиден: таблица 256 полуслов = 512 Б влезает в банк констант,
// передаётся ПАРАМЕТРОМ ЯДРА и разворачивает ЛЮБОЙ байтовый формат одним чтением. Но §4k тут же
// оговаривает: на РАСХОДЯЩЕМСЯ адресе путь рушится в 16 раз. Индекс здесь -- САМ БАЙТ ДАННЫХ, то
// есть расходящийся по построению. Меряем, а не вспоминаем.
struct LutParam { __half t[256]; };
__device__ __forceinline__ void exp_lut(const LutParam lut, uint32_t x, uint32_t& lo, uint32_t& hi) {
    const uint16_t* t = reinterpret_cast<const uint16_t*>(lut.t);
    lo = (uint32_t)t[x & 0xFF] | ((uint32_t)t[(x >> 8) & 0xFF] << 16);
    hi = (uint32_t)t[(x >> 16) & 0xFF] | ((uint32_t)t[(x >> 24) & 0xFF] << 16);
}

// --- ТЕСТ ПОЛОСЫ: разворот В ЦЕПИ подачи операнда, как в настоящем мейнлупе -----------------------
template <int MODE>
__global__ void k_chain(float* o, const uint32_t* src, LutParam lut) {
    float d[8];
    uint32_t A[2], B[2], x = src[threadIdx.x & 31];
    #pragma unroll
    for (int j = 0; j < 8; ++j) d[j] = (float)j;
    A[0] = A[1] = x;
    #pragma unroll 1
    for (int i = 0; i < NIT; ++i) {
        if (MODE == 0)      { B[0] = x; B[1] = x ^ i; }             // без разворота -- база
        else if (MODE == 1) exp_i8(x ^ i, B[0], B[1]);
        else if (MODE == 2) exp_e5m2(x ^ i, B[0], B[1]);
        else if (MODE == 3) exp_e4m3(x ^ i, B[0], B[1]);
        else                exp_lut(lut, x ^ i, B[0], B[1]);
        MMA(d, A, B);
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += d[j];
    o[threadIdx.x] = s;
}

// --- ПРОВЕРКА ТОЧНОСТИ: разворот обязан совпасть с эталонной конверсией ПОБИТОВО -----------------
__global__ void k_check(const uint32_t* src, uint32_t* out_i8, uint32_t* out_e5, uint32_t* out_e4) {
    uint32_t x = src[threadIdx.x], a, b;
    exp_i8(x, a, b);   out_i8[2 * threadIdx.x] = a; out_i8[2 * threadIdx.x + 1] = b;
    exp_e5m2(x, a, b); out_e5[2 * threadIdx.x] = a; out_e5[2 * threadIdx.x + 1] = b;
    exp_e4m3(x, a, b); out_e4[2 * threadIdx.x] = a; out_e4[2 * threadIdx.x + 1] = b;
}

static LutParam g_lut;
static float run(void (*k)(float*, const uint32_t*, LutParam), float* o, const uint32_t* s) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    k<<<80 * 4, 128>>>(o, s, g_lut); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 20; ++r) k<<<80 * 4, 128>>>(o, s, g_lut);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / 20.0f;
}

// эталон: e4m3 -> float по определению формата (знак, порядок со смещением 7, 3 бита мантиссы)
static float ref_e4m3(uint8_t u) {
    int s = (u >> 7) & 1, e = (u >> 3) & 15, m = u & 7;
    if (e == 15 && m == 7) return s ? -480.0f : 480.0f;  // e4m3fn NaN -> КОНЕЧНОЕ +-480, см. отчёт
    float v = e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf(m / 8.0f, -6);
    return s ? -v : v;
}
static float ref_e5m2(uint8_t u) {
    int s = (u >> 7) & 1, e = (u >> 2) & 31, m = u & 3;
    if (e == 31) return m ? 0.0f / 0.0f : (s ? -1.0f : 1.0f) / 0.0f;
    float v = e ? ldexpf(1.0f + m / 4.0f, e - 15) : ldexpf(m / 4.0f, -14);
    return s ? -v : v;
}

int main() {
    uint32_t h[256];
    for (int i = 0; i < 256; ++i)                                // все 256 значений байта
        h[i] = (uint32_t)i | ((uint32_t)i << 8) | ((uint32_t)i << 16) | ((uint32_t)i << 24);
    uint32_t *ds, *o8, *o5, *o4; float* o;
    cudaMalloc(&ds, sizeof h); cudaMemcpy(ds, h, sizeof h, cudaMemcpyHostToDevice);
    cudaMalloc(&o8, 2048 * 4); cudaMalloc(&o5, 2048 * 4); cudaMalloc(&o4, 2048 * 4);
    cudaMalloc(&o, 80 * 4 * 128 * 4);

    // 1) ТОЧНОСТЬ
    k_check<<<1, 256>>>(ds, o8, o5, o4);
    uint32_t r5[512], r4[512];
    cudaMemcpy(r5, o5, sizeof r5, cudaMemcpyDeviceToHost);
    cudaMemcpy(r4, o4, sizeof r4, cudaMemcpyDeviceToHost);
    int bad5 = 0, bad4 = 0;
    for (int i = 0; i < 256; ++i) {
        __half g5, g4;
        uint16_t b5 = (uint16_t)(r5[2 * i] & 0xFFFF), b4 = (uint16_t)(r4[2 * i] & 0xFFFF);
        memcpy(&g5, &b5, 2); memcpy(&g4, &b4, 2);
        float e5 = ref_e5m2((uint8_t)i), e4 = ref_e4m3((uint8_t)i) / 256.0f;   // e4m3: множитель 256 в масштабе
        float v5 = __half2float(g5), v4 = __half2float(g4);
        bool n5 = (e5 != e5), n4 = (e4 != e4);
        if (!((n5 && v5 != v5) || v5 == e5)) { if (bad5 < 4) printf("  e5m2 байт %3d: %g против %g\n", i, v5, e5); ++bad5; }
        if (!((n4 && v4 != v4) || v4 == e4)) { if (bad4 < 4) printf("  e4m3 байт %3d: %g против %g\n", i, v4, e4); ++bad4; }
    }
    printf("ТОЧНОСТЬ на всех 256 значениях байта: e5m2 расхождений %d, e4m3 (с множителем 256) %d\n\n",
           bad5, bad4);

    // 2) ПОЛОСА: разворот в цепи операнда
    for (int i = 0; i < 256; ++i) { uint16_t b = (uint16_t)((i & 0x7F) << 7); memcpy(&g_lut.t[i], &b, 2); }
    float t0 = run(k_chain<0>, o, ds), t1 = run(k_chain<1>, o, ds),
          t2 = run(k_chain<2>, o, ds), t3 = run(k_chain<3>, o, ds), t4 = run(k_chain<4>, o, ds);
    printf("%-28s %9s %9s\n", "разворот в цепи подачи", "мс", "к базе");
    printf("%-28s %9.3f %9s\n", "без разворота (база)", t0, "1.000");
    printf("%-28s %9.3f %9.3f\n", "int8 (наш нынешний)", t1, t1 / t0);
    printf("%-28s %9.3f %9.3f\n", "e5m2", t2, t2 / t0);
    printf("%-28s %9.3f %9.3f\n", "e4m3", t3, t3 / t0);
    printf("%-28s %9.3f %9.3f\n", "ЛЮБОЙ формат через ЛУТ", t4, t4 / t0);
    printf("\nинструкций на 4 байта: int8 2, e5m2 2, e4m3 8\n");
    printf("ОГОВОРКА: единственный NaN у e4m3fn (0x7F/0xFF) приём\n"
           "переводит в конечное +-480. Для KV-кэша это не отказ (NaN там уже означает сбой),\n"
           "но различие зафиксировано, а не спрятано в допуске.\n");
    return 0;
}
