// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// ГРАФИК ЦЕЛИКОМ: Tensor(HMMA плитка N) + INT32(разворот int8 + адреса N+1) +
// FP32(доводка плитки N-1) + LSU(загрузка плитки N+1). Тезис: на int8 сходятся все четыре.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define NIT 1024
#define MMA(D,A,B) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"          \
  :"+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
  :"r"(A[0]),"r"(A[1]),"r"(B[0]),"r"(B[1]))
// наш примитив: 4 байта -> 4 значения fp16 за ТРИ инструкции (INT32-конвейер)
#define EXP(w,f01,f23) { uint32_t _w=(w)^0x80808080u;                       \
  asm volatile("prmt.b32 %0,%1,%2,0x4140;":"=r"(f01):"r"(_w),"r"(0x64646464u)); \
  asm volatile("prmt.b32 %0,%1,%2,0x4342;":"=r"(f23):"r"(_w),"r"(0x64646464u)); }
#define INIT float d[8]; uint32_t A[2],B[2]; \
  _Pragma("unroll") for(int j=0;j<8;j++) d[j]=0.f; A[0]=A[1]=B[0]=B[1]=0x3C003C00u;
#define M4 MMA(d,A,B); MMA(d,A,B); MMA(d,A,B); MMA(d,A,B);
// (1) только тензор -- потолок
__global__ void k_t(float* o){ INIT
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4 M4 } o[threadIdx.x]=d[0]+d[1]; }
// (2) тензор + разворот int8 (INT32)
__global__ void k_ti(float* o, uint32_t w){ INIT uint32_t f0,f1,f2,f3;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4 EXP(w,f0,f1) M4 EXP(w+i,f2,f3) }
  o[threadIdx.x]=d[0]+d[1]+(float)(f0+f1+f2+f3); }
// (3) + доводка предыдущей плитки (FP32)
__global__ void k_tif(float* o, uint32_t w, float s){ INIT uint32_t f0,f1,f2,f3; float x0=s,x1=s;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4 EXP(w,f0,f1)
    asm volatile("fma.rn.f32 %0,%0,%1,%1;":"+f"(x0):"f"(s));
    M4 EXP(w+i,f2,f3)
    asm volatile("fma.rn.f32 %0,%0,%1,%1;":"+f"(x1):"f"(s)); }
  o[threadIdx.x]=d[0]+d[1]+(float)(f0+f1+f2+f3)+x0+x1; }
// (4) + загрузка следующей плитки (LSU), int8 = 16 БАЙТ на полосу одной инструкцией
__global__ void k_all(const uint4* g, float* o, uint32_t w, float s){ INIT
  __shared__ uint4 sh[512];
  for(int k=threadIdx.x;k<512;k+=blockDim.x) sh[k]=g[k];
  __syncthreads();
  uint32_t f0,f1,f2,f3; float x0=s,x1=s; uint4 t; uint32_t b=threadIdx.x&511;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4 EXP(w,f0,f1)
    asm volatile("fma.rn.f32 %0,%0,%1,%1;":"+f"(x0):"f"(s));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(t.x),"=r"(t.y),"=r"(t.z),"=r"(t.w)
      :"r"((unsigned)__cvta_generic_to_shared(&sh[b])));
    M4 EXP(w+t.x,f2,f3)
    asm volatile("fma.rn.f32 %0,%0,%1,%1;":"+f"(x1):"f"(s));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(t.x),"=r"(t.y),"=r"(t.z),"=r"(t.w)
      :"r"((unsigned)__cvta_generic_to_shared(&sh[(b+7)&511])));
    b=(b+13)&511; }
  o[threadIdx.x]=d[0]+d[1]+(float)(f0+f1+f2+f3+t.x)+x0+x1; }
// (5) для сравнения: то же, но данные fp16 -- LSU везёт ВДВОЕ больше, разворот не нужен
__global__ void k_f16(const uint4* g, float* o, float s){ INIT
  __shared__ uint4 sh[512];
  for(int k=threadIdx.x;k<512;k+=blockDim.x) sh[k]=g[k];
  __syncthreads();
  float x0=s,x1=s; uint4 t,t2; uint32_t b=threadIdx.x&511;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4
    asm volatile("fma.rn.f32 %0,%0,%1,%1;":"+f"(x0):"f"(s));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(t.x),"=r"(t.y),"=r"(t.z),"=r"(t.w)
      :"r"((unsigned)__cvta_generic_to_shared(&sh[b])));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(t2.x),"=r"(t2.y),"=r"(t2.z),"=r"(t2.w)
      :"r"((unsigned)__cvta_generic_to_shared(&sh[(b+3)&511])));
    M4
    asm volatile("fma.rn.f32 %0,%0,%1,%1;":"+f"(x1):"f"(s));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(t.x),"=r"(t.y),"=r"(t.z),"=r"(t.w)
      :"r"((unsigned)__cvta_generic_to_shared(&sh[(b+7)&511])));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(t2.x),"=r"(t2.y),"=r"(t2.z),"=r"(t2.w)
      :"r"((unsigned)__cvta_generic_to_shared(&sh[(b+11)&511])));
    b=(b+13)&511; }
  o[threadIdx.x]=d[0]+d[1]+(float)(t.x+t2.x)+x0+x1; }
int main(){ cudaSetDevice(0); float* o; cudaMalloc(&o,8192*4);
  uint4* g; cudaMalloc(&g,512*16); cudaMemset(g,1,512*16);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms; double cyc;
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cyc=(double)ms/5.0*1e-3*1.380e9; }
  double nh=(double)B*T*NIT*8/32.0; double base;
  printf("=== ГРАФИК ЦЕЛИКОМ: сколько тензорной полосы остаётся при полной загрузке всех блоков ===\n");
  printf("%-52s %10s %8s\n","состав","HMMA/такт","от пика");
  T5((k_t<<<B,T>>>(o)))              base=nh/cyc/80.0; printf("%-52s %10.3f %8.3f\n","Tensor (потолок)",base,1.0);
  T5((k_ti<<<B,T>>>(o,0x01020304u))) printf("%-52s %10.3f %8.3f\n","+ INT32: разворот int8 перестановкой байт",nh/cyc/80.0,(nh/cyc/80.0)/base);
  T5((k_tif<<<B,T>>>(o,0x01020304u,1.0001f))) printf("%-52s %10.3f %8.3f\n","+ FP32: доводка плитки N-1",nh/cyc/80.0,(nh/cyc/80.0)/base);
  T5((k_all<<<B,T>>>(g,o,0x01020304u,1.0001f))) printf("%-52s %10.3f %8.3f  <- ВСЕ ЧЕТЫРЕ\n","+ LSU: загрузка плитки N+1 (int8, 16 Б/полоса)",nh/cyc/80.0,(nh/cyc/80.0)/base);
  T5((k_f16<<<B,T>>>(g,o,1.0001f)))  printf("%-52s %10.3f %8.3f  <- fp16: вдвое больше загрузок\n","СРАВНЕНИЕ fp16: LSU x2, разворот не нужен",nh/cyc/80.0,(nh/cyc/80.0)/base);
  return 0; }
