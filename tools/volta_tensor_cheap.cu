// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// ЧТО ДЕШЕВЛЕ ДЕЛАТЬ В ТЕНЗОРАХ: не умножение матриц, а операции со СТРУКТУРНОЙ константой.
// diag(a)*O -- перемасштабирование строк; P*1 -- сумма по строке. Обе сейчас идут на FP32/сдвигах.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define NIT 2048
#define MMA(D,A,B) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"          \
  :"+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
  :"r"(A[0]),"r"(A[1]),"r"(B[0]),"r"(B[1]))
// --- ПЕРЕМАСШТАБИРОВАНИЕ 8 накопителей: штатно = 8 умножений fp32
__global__ void k_scale_fp32(float* o, float a){ float d[8];
  #pragma unroll
  for(int j=0;j<8;j++) d[j]=(float)j;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    #pragma unroll
    for(int j=0;j<8;j++) asm volatile("mul.f32 %0,%0,%1;":"+f"(d[j]):"f"(a)); }
  float s=0;
  #pragma unroll
  for(int j=0;j<8;j++) s+=d[j]; o[threadIdx.x]=s; }
// --- то же ЧЕРЕЗ ТЕНЗОР: diag(a) как операнд A, накопитель как B -> ОДНА инструкция на 8 значений
__global__ void k_scale_mma(float* o, uint32_t a){ float d[8]; uint32_t A[2],B[2];
  #pragma unroll
  for(int j=0;j<8;j++) d[j]=(float)j;
  A[0]=A[1]=a; B[0]=B[1]=a;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ MMA(d,A,B); }
  float s=0;
  #pragma unroll
  for(int j=0;j<8;j++) s+=d[j]; o[threadIdx.x]=s; }
// --- СУММА ПО СТРОКЕ штатно: 8 сложений + 5 сдвигов (редукция по варпу)
__global__ void k_sum_shfl(float* o, float a){ float d[8];
  #pragma unroll
  for(int j=0;j<8;j++) d[j]=a+j;
  float s=0;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    s=0;
    #pragma unroll
    for(int j=0;j<8;j++) asm volatile("add.f32 %0,%0,%1;":"+f"(s):"f"(d[j]));
    #pragma unroll
    for(int k=1;k<32;k*=2){ float t; asm volatile("shfl.sync.bfly.b32 %0,%1,%2,0x1f,0xffffffff;"
      :"=f"(t):"f"(s),"r"(k)); asm volatile("add.f32 %0,%0,%1;":"+f"(s):"f"(t)); }
    d[0]+=s*1e-9f; }
  o[threadIdx.x]=d[0]+s; }
// --- СУММА ЧЕРЕЗ ТЕНЗОР: умножение на вектор единиц = ОДНА инструкция
__global__ void k_sum_mma(float* o, uint32_t one){ float d[8]; uint32_t A[2],B[2];
  #pragma unroll
  for(int j=0;j<8;j++) d[j]=0.f;
  A[0]=A[1]=one; B[0]=B[1]=one;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ MMA(d,A,B); }
  float s=0;
  #pragma unroll
  for(int j=0;j<8;j++) s+=d[j]; o[threadIdx.x]=s; }
int main(){ cudaSetDevice(0); float* o; cudaMalloc(&o,8192*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms; double cyc;
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cyc=(double)ms/5.0*1e-3*1.380e9; }
  double n=(double)B*T*NIT;
  printf("=== ЧТО ДЕШЕВЛЕ ДЕЛАТЬ В ТЕНЗОРАХ (тактов на ОДНУ операцию над 8 накопителями) ===\n");
  double a,b;
  T5((k_scale_fp32<<<B,T>>>(o,1.0001f))) a=cyc/(n/32.0);
  printf("  перемасштабирование: 8 x mul.f32 (штатно)      %8.3f такта\n",a);
  T5((k_scale_mma<<<B,T>>>(o,0x3C003C00u))) b=cyc/(n/32.0);
  printf("  перемасштабирование: diag через ОДНУ mma       %8.3f такта   x%.2f\n",b,a/b);
  T5((k_sum_shfl<<<B,T>>>(o,1.0f))) a=cyc/(n/32.0);
  printf("  сумма по строке: 8 сложений + 5 бабочек        %8.3f такта\n",a);
  T5((k_sum_mma<<<B,T>>>(o,0x3C003C00u))) b=cyc/(n/32.0);
  printf("  сумма по строке: умножение на единицы, ОДНА mma %8.3f такта   x%.2f\n",b,a/b);
  return 0; }
