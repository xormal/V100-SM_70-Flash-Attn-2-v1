// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// ЧЕТЫРЕ МАРШРУТА ДЛЯ ОДНОЙ И ТОЙ ЖЕ int8-РАБОТЫ, в ОДНИХ единицах: умножений-с-накоплением на такт.
// Затем -- совместная работа всех маршрутов, чтобы найти суммарный потолок метапроцессора.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define NIT 1024
#define MMA(D,A,B) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"          \
  :"+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
  :"r"(A[0]),"r"(A[1]),"r"(B[0]),"r"(B[1]))
#define INIT float d[8]; uint32_t A[2],B[2]; \
  _Pragma("unroll") for(int j=0;j<8;j++) d[j]=0.f; A[0]=A[1]=B[0]=B[1]=0x3C003C00u;
#define M4 MMA(d,A,B); MMA(d,A,B); MMA(d,A,B); MMA(d,A,B);
// маршрут 1: ТЕНЗОР. mma.sync.m8n8k4 = 8*8*4 = 256 умножений-с-накоплением на инструкцию варпа
__global__ void k_tensor(float* o){ INIT
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4 M4 } o[threadIdx.x]=d[0]; }
// маршрут 2: dp4a на INT32. 4 int8-умножения-с-накоплением на инструкцию НИТИ
__global__ void k_dp4a(int* o, uint32_t a, uint32_t b){ int y[8];
  #pragma unroll
  for(int j=0;j<8;j++) y[j]=j;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    #pragma unroll
    for(int j=0;j<8;j++) asm volatile("dp4a.s32.s32 %0,%1,%2,%0;":"+r"(y[j]):"r"(a),"r"(b));
    #pragma unroll
    for(int j=0;j<8;j++) asm volatile("dp4a.s32.s32 %0,%1,%2,%0;":"+r"(y[j]):"r"(a),"r"(b)); }
  int s=0;
  #pragma unroll
  for(int j=0;j<8;j++) s+=y[j]; o[threadIdx.x]=s; }
// маршрут 3: FP32 упакованными полями -- 2 накопления на инструкцию НИТИ
__global__ void k_fp32(float* o, float p){ float z[8];
  #pragma unroll
  for(int j=0;j<8;j++) z[j]=p+j;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    #pragma unroll
    for(int j=0;j<8;j++) asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(z[j]):"f"(p),"f"(p));
    #pragma unroll
    for(int j=0;j<8;j++) asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(z[j]):"f"(p),"f"(p)); }
  float s=0;
  #pragma unroll
  for(int j=0;j<8;j++) s+=z[j]; o[threadIdx.x]=s; }
// ВСЕ ТРИ ВМЕСТЕ: тензор + dp4a + fp32-поля
__global__ void k_all(float* o, int* q, uint32_t a, uint32_t b, float p){ INIT
  int y[4]; float z[4];
  #pragma unroll
  for(int j=0;j<4;j++){ y[j]=j; z[j]=p+j; }
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    M4
    #pragma unroll
    for(int j=0;j<4;j++){ asm volatile("dp4a.s32.s32 %0,%1,%2,%0;":"+r"(y[j]):"r"(a),"r"(b));
                          asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(z[j]):"f"(p),"f"(p)); }
    M4
    #pragma unroll
    for(int j=0;j<4;j++){ asm volatile("dp4a.s32.s32 %0,%1,%2,%0;":"+r"(y[j]):"r"(a),"r"(b));
                          asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(z[j]):"f"(p),"f"(p)); } }
  int s=0; float t=0;
  #pragma unroll
  for(int j=0;j<4;j++){ s+=y[j]; t+=z[j]; }
  o[threadIdx.x]=d[0]+t; q[threadIdx.x]=s; }
int main(){ cudaSetDevice(0); float* o; int* q; cudaMalloc(&o,8192*4); cudaMalloc(&q,8192*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms; double cyc;
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cyc=(double)ms/5.0*1e-3*1.380e9; }
  double warpI=(double)B*T*NIT*8/32.0;      // инструкций ВАРПА (тензор)
  double thrI =(double)B*T*NIT*16;          // инструкций НИТИ (dp4a/fp32)
  double tMAC=0,dMAC=0,fMAC=0;
  printf("=== МАРШРУТЫ ДЛЯ int8-РАБОТЫ, в умножениях-с-накоплением на такт на SM ===\n");
  T5((k_tensor<<<B,T>>>(o)))            tMAC=warpI*256.0/cyc/80.0;
  printf("  тензор  mma.m8n8k4 (256 МАС/инстр.варпа) %10.0f МАС/такт/SM\n",tMAC);
  T5((k_dp4a<<<B,T>>>(q,0x01020304u,0x04030201u))) dMAC=thrI*4.0/cyc/80.0;
  printf("  INT32   dp4a       (4 МАС/инстр.нити)   %10.0f МАС/такт/SM\n",dMAC);
  T5((k_fp32<<<B,T>>>(o,1.0001f)))      fMAC=thrI*2.0/cyc/80.0;
  printf("  FP32    поля 2x12  (2 МАС/инстр.нити)   %10.0f МАС/такт/SM\n",fMAC);
  double sum=tMAC+dMAC+fMAC;
  printf("\n  сумма поодиночке (недостижимый идеал)   %10.0f МАС/такт/SM\n",sum);
  T5((k_all<<<B,T>>>(o,q,0x01020304u,0x04030201u,1.0001f)))
  double aT=warpI*256.0/cyc/80.0, aD=(double)B*T*NIT*8*4.0/cyc/80.0, aF=(double)B*T*NIT*8*2.0/cyc/80.0;
  printf("  ВСЕ ТРИ ВМЕСТЕ: тензор %.0f + dp4a %.0f + fp32 %.0f = %.0f  (%.1f%% от суммы)\n",
     aT,aD,aF,aT+aD+aF,100*(aT+aD+aF)/sum);
  printf("\n=== ОПТИМАЛЬНАЯ РАЗВЁСКА: доля работы на каждый маршрут ===\n");
  double tot=aT+aD+aF;
  printf("  тензор %5.1f%%   INT32/dp4a %5.1f%%   FP32/поля %5.1f%%\n",100*aT/tot,100*aD/tot,100*aF/tot);
  printf("  прирост над ОДНИМ тензором: x%.3f  (%.0f -> %.0f МАС/такт/SM)\n",tot/tMAC,tMAC,tot);
  return 0; }
