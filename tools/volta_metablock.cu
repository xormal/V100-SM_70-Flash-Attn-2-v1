// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// МЕТАБЛОК ЦЕЛИКОМ: Tensor(HMMA) + FP32 + INT32 + FP64 одновременно.
// Вопрос: сколько работы других конвейеров уезжает ПОД идущей HMMA бесплатно.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define NIT 1024
#define MMA(D,A,B) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"          \
  :"+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
  :"r"(A[0]),"r"(A[1]),"r"(B[0]),"r"(B[1]))
#define DECL float d0[8],d1[8],d2[8],d3[8]; uint32_t a[2],b[2];                  \
  _Pragma("unroll") for(int j=0;j<8;j++){ d0[j]=0.f; d1[j]=0.f; d2[j]=0.f; d3[j]=0.f; } \
  a[0]=a[1]=0x3C003C00u; b[0]=b[1]=0x3C003C00u;
#define OUT o[threadIdx.x]=d0[0]+d1[1]+d2[2]+d3[3];
#define M4 MMA(d0,a,b); MMA(d1,a,b); MMA(d2,a,b); MMA(d3,a,b);
__global__ void k_t(float* o){ DECL
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4 M4 } OUT }
__global__ void k_tf(float* o, float s){ DECL float x0=s,x1=s,x2=s,x3=s;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4
    asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(s)); asm volatile("add.f32 %0,%0,%1;":"+f"(x1):"f"(s));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x2):"f"(s)); asm volatile("add.f32 %0,%0,%1;":"+f"(x3):"f"(s));
    M4
    asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(s)); asm volatile("add.f32 %0,%0,%1;":"+f"(x1):"f"(s));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x2):"f"(s)); asm volatile("add.f32 %0,%0,%1;":"+f"(x3):"f"(s)); }
  o[threadIdx.x]=d0[0]+d1[1]+d2[2]+d3[3]+x0+x1+x2+x3; }
__global__ void k_ti(float* o, uint32_t u){ DECL uint32_t y0=u,y1=u,y2=u,y3=u;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4
    asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y0):"r"(u)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y1):"r"(u));
    asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y2):"r"(u)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y3):"r"(u));
    M4
    asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y0):"r"(u)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y1):"r"(u));
    asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y2):"r"(u)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y3):"r"(u)); }
  o[threadIdx.x]=d0[0]+d1[1]+d2[2]+d3[3]+(float)(y0+y1+y2+y3); }
__global__ void k_tfi(float* o, float s, uint32_t u){ DECL float x0=s,x1=s; uint32_t y0=u,y1=u;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4
    asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(s)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y0):"r"(u));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x1):"f"(s)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y1):"r"(u));
    M4
    asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(s)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y0):"r"(u));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x1):"f"(s)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y1):"r"(u)); }
  o[threadIdx.x]=d0[0]+d1[1]+d2[2]+d3[3]+x0+x1+(float)(y0+y1); }
__global__ void k_tfid(float* o, float s, uint32_t u, double w){ DECL float x0=s,x1=s; uint32_t y0=u,y1=u; double z0=w,z1=w;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4
    asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(s)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y0):"r"(u));
    asm volatile("add.f64 %0,%0,%1;":"+d"(z0):"d"(w)); asm volatile("add.f32 %0,%0,%1;":"+f"(x1):"f"(s));
    M4
    asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y1):"r"(u)); asm volatile("add.f64 %0,%0,%1;":"+d"(z1):"d"(w));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(s)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y0):"r"(u)); }
  o[threadIdx.x]=d0[0]+d1[1]+d2[2]+d3[3]+x0+x1+(float)(y0+y1)+(float)(z0+z1); }
int main(){ cudaSetDevice(0); float* o; cudaMalloc(&o,8192*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms; double cyc;
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cyc=(double)ms/5.0*1e-3*1.380e9; }
  double nh=(double)B*T*NIT*8/32.0;          // инструкций HMMA на варп
  double n8=(double)B*T*NIT*8/32.0;          // прочих инструкций на варп (по 8 за итерацию)
  double base=0;
  printf("=== МЕТАБЛОК: Tensor + FP32 + INT32 + FP64 (V100, 80x256) ===\n");
  printf("%-42s %10s %10s %8s\n","состав","HMMA/такт","прочих/такт","HMMA от базы");
  T5((k_t<<<B,T>>>(o)))          base=nh/cyc/80.0;
  printf("%-42s %10.3f %10s %8s\n","только HMMA",base,"--","1.000");
  T5((k_tf<<<B,T>>>(o,1.0f)))    printf("%-42s %10.3f %10.3f %8.3f\n","HMMA + FP32",nh/cyc/80.0,n8/cyc/80.0,(nh/cyc/80.0)/base);
  T5((k_ti<<<B,T>>>(o,3u)))      printf("%-42s %10.3f %10.3f %8.3f\n","HMMA + INT32",nh/cyc/80.0,n8/cyc/80.0,(nh/cyc/80.0)/base);
  T5((k_tfi<<<B,T>>>(o,1.0f,3u)))printf("%-42s %10.3f %10.3f %8.3f\n","HMMA + FP32 + INT32",nh/cyc/80.0,n8/cyc/80.0,(nh/cyc/80.0)/base);
  T5((k_tfid<<<B,T>>>(o,1.0f,3u,1.5))) printf("%-42s %10.3f %10.3f %8.3f\n","HMMA + FP32 + INT32 + FP64",nh/cyc/80.0,n8/cyc/80.0,(nh/cyc/80.0)/base);
  printf("\n  'HMMA от базы' = 1.000 означает, что прочая работа уехала ПОЛНОСТЬЮ БЕСПЛАТНО\n");
  return 0; }
