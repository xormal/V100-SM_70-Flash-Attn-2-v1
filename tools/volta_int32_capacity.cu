// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// ЁМКОСТЬ INT32 ПОД ИДУЩЕЙ HMMA: сколько целочисленных инструкций уезжает ДАРОМ на 4 HMMA.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define NIT 1024
#define MMA(D,A,B) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"          \
  :"+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
  :"r"(A[0]),"r"(A[1]),"r"(B[0]),"r"(B[1]))
#define P1(d,a) asm volatile("prmt.b32 %0,%1,%2,0x4140;":"=r"(d):"r"(a),"r"(0x64646464u))
#define INIT float d[8]; uint32_t A[2],B[2]; \
  _Pragma("unroll") for(int j=0;j<8;j++) d[j]=0.f; A[0]=A[1]=B[0]=B[1]=0x3C003C00u;
#define M4 MMA(d,A,B); MMA(d,A,B); MMA(d,A,B); MMA(d,A,B);
// N НЕЗАВИСИМЫХ целочисленных инструкций на каждые 4 HMMA
#define KERN(N) __global__ void k_##N(float* o, uint32_t w){ INIT                       \
  uint32_t y[16]; _Pragma("unroll") for(int j=0;j<16;j++) y[j]=w+j;                     \
  _Pragma("unroll 1") for(int i=0;i<NIT;i++){                                           \
    M4  _Pragma("unroll") for(int j=0;j<N;j++) P1(y[j],y[j]);                            \
    M4  _Pragma("unroll") for(int j=0;j<N;j++) P1(y[j],y[j]); }                           \
  uint32_t s=0; _Pragma("unroll") for(int j=0;j<16;j++) s+=y[j];                        \
  o[threadIdx.x]=d[0]+d[1]+(float)s; }
KERN(0) KERN(2) KERN(4) KERN(6) KERN(8) KERN(12) KERN(16)
int main(){ cudaSetDevice(0); float* o; cudaMalloc(&o,8192*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms; double cyc;
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cyc=(double)ms/5.0*1e-3*1.380e9; }
  double nh=(double)B*T*NIT*8/32.0; double base;
  printf("=== ЁМКОСТЬ INT32 ПОД HMMA: сколько перестановок байт уезжает ДАРОМ ===\n");
  printf("%-28s %10s %9s %14s %12s\n","int32 инстр. на 4 HMMA","HMMA/такт","от пика","int32/такт","на 1 HMMA");
  #define ROW(N) { T5((k_##N<<<B,T>>>(o,0x01020304u))) double h=nh/cyc/80.0;             \
    double ii=(double)B*T*NIT*2*(N)/32.0/cyc/80.0;                                       \
    printf("%-28d %10.3f %9.3f %14.3f %12.2f%s\n",N,h,h/base,ii,(N)?ii/h:0.0,            \
      (h/base>0.97)?"  <- ДАРОМ":((h/base>0.90)?"  <- почти":"")); }
  T5((k_0<<<B,T>>>(o,0x01020304u))) base=nh/cyc/80.0;
  printf("%-28d %10.3f %9.3f %14s %12s\n",0,base,1.0,"--","--");
  ROW(2) ROW(4) ROW(6) ROW(8) ROW(12) ROW(16)
  printf("\n  'на 1 HMMA' -- сколько целочисленных инструкций приходится на одну HMMA при этой загрузке\n");
  return 0; }
