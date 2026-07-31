// Что из механизмов сокрытия задержки РЕАЛЬНО существует на sm_70: компилируется ли, и во что.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

__global__ void k_prefetch(const float* g, float* o) {          // предвыборка в L2 без регистра-приёмника
    asm volatile("prefetch.global.L2 [%0];" :: "l"(g + threadIdx.x * 32));
    o[threadIdx.x] = g[threadIdx.x];
}
__global__ void k_prefetch_l1(const float* g, float* o) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(g + threadIdx.x * 32));
    o[threadIdx.x] = g[threadIdx.x];
}
__global__ void k_named_bar(float* o) {                          // именованный барьер на ПОДМНОЖЕСТВО варпов
    const int w = threadIdx.x >> 5;
    if (w < 4) { o[threadIdx.x] = 1.f; asm volatile("bar.sync 1, 128;"); }
    else       { asm volatile("bar.sync 2, 384;"); o[threadIdx.x] = 2.f; }
}
__global__ void k_split_bar(float* o, const float* g) {          // arrive + отдельное ожидание
    o[threadIdx.x] = g[threadIdx.x];
    asm volatile("bar.arrive 0, 512;");
    float x = g[threadIdx.x + 512] * 2.f;                        // независимая работа между arrive и sync
    asm volatile("bar.sync 0, 512;");
    o[threadIdx.x + 512] = x;
}
__global__ void k_ldgsts(const float* g, float* o) {             // есть ли cp.async на Volta (ожидаемо нет)
    __shared__ float s[64];
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"((unsigned)__cvta_generic_to_shared(&s[threadIdx.x])), "l"(g));
#else
    s[threadIdx.x] = g[threadIdx.x];
#endif
    __syncthreads(); o[threadIdx.x] = s[threadIdx.x];
}
int main() { printf("сборка прошла\n"); return 0; }
