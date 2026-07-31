// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// БАРЬЕР КАК ПЕРЕДАЧА, А НЕ КАК ОСТАНОВКА: bar.arrive (сигналю, не жду) против bar.sync (жду).
// И специализация варпов: одни грузят, другие считают, барьер -- защёлка между буферами.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define NIT 1024
#define MMA(D,A,B) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"          \
  :"+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
  :"r"(A[0]),"r"(A[1]),"r"(B[0]),"r"(B[1]))
// --- цена половинок барьера
__global__ void k_none(float* o){ float a=threadIdx.x;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ asm volatile("":::"memory"); a+=1.0f; } o[threadIdx.x]=a; }
__global__ void k_sync(float* o){ float a=threadIdx.x;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ asm volatile("bar.sync 0;"); a+=1.0f; } o[threadIdx.x]=a; }
__global__ void k_arrive(float* o){ float a=threadIdx.x;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ asm volatile("bar.arrive 0, 256;"); a+=1.0f; } o[threadIdx.x]=a; }
// --- МОНОЛИТ: каждый варп сам грузит и сам считает
__global__ void k_mono(const float* g, float* o){ __shared__ float sm[2][2048];
  float d[8]; uint32_t A[2],B[2];
  #pragma unroll
  for(int j=0;j<8;j++) d[j]=0.f;
  A[0]=A[1]=B[0]=B[1]=0x3C003C00u;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    int buf=i&1;
    for(int k=threadIdx.x;k<2048;k+=blockDim.x) sm[buf][k]=g[(k+i*37)&65535];   // загрузка
    __syncthreads();
    float v=sm[buf][threadIdx.x&2047]; A[0]=__float_as_uint(v)|0x3C003C00u;
    MMA(d,A,B); MMA(d,A,B); MMA(d,A,B); MMA(d,A,B);                              // счёт
    __syncthreads(); }
  o[threadIdx.x]=d[0]+d[1]; }
// --- СПЕЦИАЛИЗАЦИЯ: варпы 0-1 грузят (bar.arrive -- не ждут), варпы 2-7 считают (bar.sync -- ждут)
__global__ void k_spec(const float* g, float* o){ __shared__ float sm[2][2048];
  float d[8]; uint32_t A[2],B[2];
  #pragma unroll
  for(int j=0;j<8;j++) d[j]=0.f;
  A[0]=A[1]=B[0]=B[1]=0x3C003C00u;
  const int w = threadIdx.x>>5;
  if(w < 2){                                   // ПРОИЗВОДИТЕЛЬ
    #pragma unroll 1
    for(int i=0;i<NIT;i++){ int buf=i&1;
      for(int k=(threadIdx.x&63);k<2048;k+=64) sm[buf][k]=g[(k+i*37)&65535];
      asm volatile("bar.arrive 1, 256;");      // ЩЁЛКНУЛ: буфер готов, НЕ ЖДУ
      asm volatile("bar.sync 2, 256;"); }      // жду, пока потребитель освободит второй буфер
  } else {                                     // ПОТРЕБИТЕЛЬ
    #pragma unroll 1
    for(int i=0;i<NIT;i++){ int buf=i&1;
      asm volatile("bar.sync 1, 256;");        // жду защёлку
      float v=sm[buf][threadIdx.x&2047]; A[0]=__float_as_uint(v)|0x3C003C00u;
      MMA(d,A,B); MMA(d,A,B); MMA(d,A,B); MMA(d,A,B);
      asm volatile("bar.arrive 2, 256;"); }    // отпустил буфер, НЕ ЖДУ
  }
  o[threadIdx.x]=d[0]+d[1]; }
int main(){ cudaSetDevice(0); float* o; cudaMalloc(&o,8192*4);
  float* g; cudaMalloc(&g,65536*4); cudaMemset(g,0,65536*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms; double cyc;
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cyc=(double)ms/5.0*1e-3*1.380e9; }
  printf("=== ПОЛОВИНКИ БАРЬЕРА: сигнал против ожидания ===\n");
  T5((k_none<<<B,T>>>(o)))   double base=cyc/NIT; printf("  без барьера (база)      %7.2f тактов/итер\n",base);
  T5((k_sync<<<B,T>>>(o)))   printf("  bar.sync   (ЖДУ)        %7.2f  -> чистая цена %6.2f такта\n",cyc/NIT,cyc/NIT-base);
  T5((k_arrive<<<B,T>>>(o))) printf("  bar.arrive (СИГНАЛ)     %7.2f  -> чистая цена %6.2f такта\n",cyc/NIT,cyc/NIT-base);
  printf("\n=== КОНВЕЙЕР: монолит против специализации варпов ===\n");
  T5((k_mono<<<B,T>>>(g,o))) double m=ms/5.0; printf("  монолит (все грузят и считают)   %7.3f мс\n",m);
  T5((k_spec<<<B,T>>>(g,o))) printf("  специализация 2 грузят / 6 считают %7.3f мс  (x%.3f)\n",ms/5.0,m/(ms/5.0));
  return 0; }
