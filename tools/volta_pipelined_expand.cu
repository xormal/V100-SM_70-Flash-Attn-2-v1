// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// РАЗВОРОТ В ЦЕПИ ПРОТИВ РАЗВОРОТА, СДВИНУТОГО НА ТАКТ ВПЕРЁД.
// Тезис: ёмкость INT32 есть, теряет её ЗАВИСИМОСТЬ, а не занятость.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define NIT 1024
#define MMA(D,A,B) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"          \
  :"+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
  :"r"(A[0]),"r"(A[1]),"r"(B[0]),"r"(B[1]))
#define EXP(w,f0,f1) { uint32_t _w=(w)^0x80808080u;                                    \
  asm volatile("prmt.b32 %0,%1,%2,0x4140;":"=r"(f0):"r"(_w),"r"(0x64646464u));         \
  asm volatile("prmt.b32 %0,%1,%2,0x4342;":"=r"(f1):"r"(_w),"r"(0x64646464u)); }
#define INIT float d[8]; _Pragma("unroll") for(int j=0;j<8;j++) d[j]=0.f;
// (1) потолок: операнды готовы, разворота нет
__global__ void k_peak(float* o){ INIT uint32_t A[2],B[2]; A[0]=A[1]=B[0]=B[1]=0x3C003C00u;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ MMA(d,A,B);MMA(d,A,B);MMA(d,A,B);MMA(d,A,B);
                          MMA(d,A,B);MMA(d,A,B);MMA(d,A,B);MMA(d,A,B); } o[threadIdx.x]=d[0]+d[1]; }
// (2) В ЦЕПИ: развернули -> тут же скормили той же MMA
__global__ void k_chain(float* o, const uint32_t* g){ INIT uint32_t A[2],B[2]; B[0]=B[1]=0x3C003C00u;
  uint32_t w=g[threadIdx.x&255];
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    EXP(w+i, A[0], A[1]) MMA(d,A,B);MMA(d,A,B);MMA(d,A,B);MMA(d,A,B);
    EXP(w+i+1, A[0], A[1]) MMA(d,A,B);MMA(d,A,B);MMA(d,A,B);MMA(d,A,B); }
  o[threadIdx.x]=d[0]+d[1]; }
// (3) СДВИНУТО НА ТАКТ: считаем плиткой N, разворачиваем N+1 -- перестановки НЕЗАВИСИМЫ от текущей MMA
__global__ void k_pipe(float* o, const uint32_t* g){ INIT
  uint32_t Acur[2], Anxt[2], B[2]; B[0]=B[1]=0x3C003C00u;
  uint32_t w=g[threadIdx.x&255];
  EXP(w, Acur[0], Acur[1])                              // пролог
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    MMA(d,Acur,B); EXP(w+i+1, Anxt[0], Anxt[1])         // разворот N+1 ВПЕРЕМЕШКУ с MMA плитки N
    MMA(d,Acur,B); MMA(d,Acur,B); MMA(d,Acur,B);
    Acur[0]=Anxt[0]; Acur[1]=Anxt[1];
    MMA(d,Acur,B); EXP(w+i+2, Anxt[0], Anxt[1])
    MMA(d,Acur,B); MMA(d,Acur,B); MMA(d,Acur,B);
    Acur[0]=Anxt[0]; Acur[1]=Anxt[1]; }
  o[threadIdx.x]=d[0]+d[1]; }
int main(){ cudaSetDevice(0); float* o; uint32_t* g; cudaMalloc(&o,8192*4); cudaMalloc(&g,256*4);
  uint32_t h[256]; for(int i=0;i<256;i++) h[i]=0x01020304u+i; cudaMemcpy(g,h,256*4,cudaMemcpyHostToDevice);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms; double cyc;
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cyc=(double)ms/5.0*1e-3*1.380e9; }
  double nh=(double)B*T*NIT*8/32.0; double base;
  printf("=== РАЗВОРОТ int8: В ЦЕПИ ПРОТИВ СДВИНУТОГО НА ТАКТ ===\n");
  T5((k_peak<<<B,T>>>(o)))    base=nh/cyc/80.0;
  printf("  %-46s %8.3f HMMA/такт  %6.3f от пика\n","потолок (операнды готовы, разворота нет)",base,1.0);
  T5((k_chain<<<B,T>>>(o,g))) double c=nh/cyc/80.0;
  printf("  %-46s %8.3f HMMA/такт  %6.3f от пика\n","разворот В ЦЕПИ с той же MMA",c,c/base);
  T5((k_pipe<<<B,T>>>(o,g)))  double p=nh/cyc/80.0;
  printf("  %-46s %8.3f HMMA/такт  %6.3f от пика\n","разворот СДВИНУТ на плитку вперёд",p,p/base);
  printf("\n  сдвиг фазы дал x%.3f к цепи; потеря от пика упала с %.1f%% до %.1f%%\n",
     p/c, 100*(1-c/base), 100*(1-p/base));
  return 0; }
