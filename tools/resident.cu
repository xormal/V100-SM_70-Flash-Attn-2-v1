// Резидент-предвыборщик: загрузка, результат которой НЕ нужен, но которая обязана дойти до L2.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s: %s\n",#x,cudaGetErrorString(e));exit(1);} }while(0)

// Вариант 1: наивно — результат не используется.
__global__ void k_naive(const float* g, float* o, int n) {
    float s = 0.f;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) s += g[i];
    (void)s;                                   // компилятор вправе выкинуть весь цикл
}
// Вариант 2: сторож — значение «используется», но запись никогда не происходит.
__global__ void k_guard(const float* g, float* o, int n) {
    float s = 0.f;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) s += g[i];
    if (s == 1234.5678f) o[0] = s;
}
int main() {
    CK(cudaSetDevice(0));
    const int n = 1 << 22;
    float *g, *o; CK(cudaMalloc(&g, n * 4)); CK(cudaMalloc(&o, 4)); CK(cudaMemset(g, 0, n * 4));
    for (int k = 0; k < 2; ++k) {
        auto f = (k == 0) ? k_naive : k_guard;
        f<<<80, 256>>>(g, o, n); CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
        for (int i = 0; i < 20; ++i) f<<<80, 256>>>(g, o, n);
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b); ms /= 20;
        printf("  %-8s %7.4f мс -> %6.1f ГБ/с  %s\n", k ? "сторож" : "наивно", ms,
               (double)n * 4 / (ms * 1e-3) / 1e9, ms < 0.005 ? "<- цикл ВЫБРОШЕН" : "<- загрузки реальны");
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return 0;
}
