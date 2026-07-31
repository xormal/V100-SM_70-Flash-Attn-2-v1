// Сколько именованных барьеров даёт Volta и во что обходится ожидание.
#include <cstdio>
#include <cstdint>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s: %s\n",#x,cudaGetErrorString(e));exit(1);} }while(0)

template <int ID>
__global__ void k_bar(float* o) {
    asm volatile("bar.sync %0, 128;" :: "n"(ID));
    o[threadIdx.x] = ID;
}
// Ожидание на именованном барьере: жжёт ли оно слоты выдачи? Сравниваем блок, где половина варпов
// ждёт на барьере, с блоком, где та же половина крутит опрос разделяемой памяти.
__global__ void k_wait_bar(float* o, int iters) {
    __shared__ int flag;
    if (threadIdx.x == 0) flag = 0;
    __syncthreads();
    float acc = 0.f;
    if ((threadIdx.x >> 5) < 2) {                       // «счётные» варпы
        for (int i = 0; i < iters; ++i) acc += __sinf(acc + i);
        if (threadIdx.x == 0) { flag = 1; __threadfence_block(); }
        asm volatile("bar.arrive 1, 128;");
    } else {                                            // «ждущие» варпы
        asm volatile("bar.sync 1, 128;");
    }
    if (acc == 1234.5678f) o[0] = acc;
}
__global__ void k_wait_poll(float* o, int iters) {
    __shared__ volatile int flag;
    if (threadIdx.x == 0) flag = 0;
    __syncthreads();
    float acc = 0.f;
    if ((threadIdx.x >> 5) < 2) {
        for (int i = 0; i < iters; ++i) acc += __sinf(acc + i);
        if (threadIdx.x == 0) { __threadfence_block(); flag = 1; }
    } else {
        while (flag == 0) { }                           // ОПРОС: жжёт слоты выдачи
    }
    if (acc == 1234.5678f) o[0] = acc;
}
int main() {
    CK(cudaSetDevice(0));
    float* o; CK(cudaMalloc(&o, 4096));
    k_bar<0><<<1,128>>>(o);  CK(cudaDeviceSynchronize());
    k_bar<15><<<1,128>>>(o); CK(cudaGetLastError() == cudaSuccess ? cudaSuccess : cudaSuccess);
    printf("bar.sync id=15: %s\n", cudaDeviceSynchronize() == cudaSuccess ? "работает" : "ОШИБКА");
    const int it = 20000;
    for (int k = 0; k < 2; ++k) {
        auto f = k ? k_wait_poll : k_wait_bar;
        f<<<80,128>>>(o, it); CK(cudaDeviceSynchronize());
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
        for (int i=0;i<20;++i) f<<<80,128>>>(o, it);
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms,a,b);
        printf("  ожидание %-12s %8.4f мс\n", k ? "ОПРОСОМ" : "барьером", ms/20);
    }
    return 0;
}
