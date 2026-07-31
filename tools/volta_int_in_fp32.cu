// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// ЦЕЛЫЕ В fp32: мантисса 24 бита -> два точных поля по 12 бит, одно add.f32 = ДВА целых сложения.
// Главное: это занимает слот FP32, а не INT32 -> под HMMA доступны ОБА свободных слота под целые.
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
// --- точность: два поля по 12 бит в мантиссе fp32
__global__ void k_exact(float* o, uint32_t* chk){
  uint32_t v = threadIdx.x & 255;
  float a = (float)(v + (v<<12));            // поля 0 и 1
  float acc = 0.f;
  #pragma unroll 1
  for(int i=0;i<16;i++) acc += a;            // 255*16 = 4080 < 4096, переноса нет
  o[threadIdx.x]=acc;
  uint32_t u=(uint32_t)acc;
  chk[threadIdx.x] = (u & 4095u) | (((u>>12)&4095u)<<16); }
// --- ТОЛЬКО INT32 под HMMA: 8 перестановок на 4 HMMA (= 2/HMMA, рабочая точка §4r)
__global__ void k_i_only(float* o, uint32_t w){ INIT uint32_t y[8];
  #pragma unroll
  for(int j=0;j<8;j++) y[j]=w+j;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    M4 
    #pragma unroll
    for(int j=0;j<8;j++) asm volatile("prmt.b32 %0,%1,%2,0x4140;":"=r"(y[j]):"r"(y[j]),"r"(0x64646464u));
    M4
    #pragma unroll
    for(int j=0;j<8;j++) asm volatile("prmt.b32 %0,%1,%2,0x4140;":"=r"(y[j]):"r"(y[j]),"r"(0x64646464u)); }
  uint32_t s=0;
  #pragma unroll
  for(int j=0;j<8;j++) s+=y[j]; o[threadIdx.x]=d[0]+(float)s; }
// --- ТА ЖЕ ЦЕЛОЧИСЛЕННАЯ РАБОТА, РАЗНЕСЁННАЯ: половина на INT32, половина на FP32 (упакованные поля)
__global__ void k_split(float* o, uint32_t w, float pk){ INIT uint32_t y[4]; float z[4];
  #pragma unroll
  for(int j=0;j<4;j++){ y[j]=w+j; z[j]=pk+j; }
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    M4
    #pragma unroll
    for(int j=0;j<4;j++){ asm volatile("prmt.b32 %0,%1,%2,0x4140;":"=r"(y[j]):"r"(y[j]),"r"(0x64646464u));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(z[j]):"f"(pk)); }   // 2 целых за инстр.
    M4
    #pragma unroll
    for(int j=0;j<4;j++){ asm volatile("prmt.b32 %0,%1,%2,0x4140;":"=r"(y[j]):"r"(y[j]),"r"(0x64646464u));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(z[j]):"f"(pk)); } }
  uint32_t s=0; float t=0;
  #pragma unroll
  for(int j=0;j<4;j++){ s+=y[j]; t+=z[j]; } o[threadIdx.x]=d[0]+(float)s+t; }
// --- ВДВОЕ БОЛЬШЕ целочисленной работы через разнесение (8 int32 + 8 fp32-упакованных на 4 HMMA)
__global__ void k_both(float* o, uint32_t w, float pk){ INIT uint32_t y[8]; float z[8];
  #pragma unroll
  for(int j=0;j<8;j++){ y[j]=w+j; z[j]=pk+j; }
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    M4
    #pragma unroll
    for(int j=0;j<8;j++){ asm volatile("prmt.b32 %0,%1,%2,0x4140;":"=r"(y[j]):"r"(y[j]),"r"(0x64646464u));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(z[j]):"f"(pk)); }
    M4
    #pragma unroll
    for(int j=0;j<8;j++){ asm volatile("prmt.b32 %0,%1,%2,0x4140;":"=r"(y[j]):"r"(y[j]),"r"(0x64646464u));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(z[j]):"f"(pk)); } }
  uint32_t s=0; float t=0;
  #pragma unroll
  for(int j=0;j<8;j++){ s+=y[j]; t+=z[j]; } o[threadIdx.x]=d[0]+(float)s+t; }
__global__ void k_peak(float* o){ INIT
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ M4 M4 } o[threadIdx.x]=d[0]; }
int main(){ cudaSetDevice(0); float* o; uint32_t* c; cudaMalloc(&o,8192*4); cudaMalloc(&c,8192*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms; double cyc;
  k_exact<<<1,256>>>(o,c); cudaDeviceSynchronize();
  float hf; uint32_t hc; cudaMemcpy(&hf,o+7,4,cudaMemcpyDeviceToHost); cudaMemcpy(&hc,c+7,4,cudaMemcpyDeviceToHost);
  uint32_t u=(uint32_t)hf;
  printf("=== ТОЧНОСТЬ: два поля по 12 бит в мантиссе fp32 ===\n");
  printf("  нить 7, значение 7, 16 сложений: поле0=%u поле1=%u (ожидаем 112 и 112) -- %s\n",
     u&4095u,(u>>12)&4095u, ((u&4095u)==112 && ((u>>12)&4095u)==112)?"ТОЧНО":"НЕТ");
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cyc=(double)ms/5.0*1e-3*1.380e9; }
  double nh=(double)B*T*NIT*8/32.0; double base; double ni=(double)B*T*NIT/32.0;
  printf("\n=== РАЗНЕСЕНИЕ ЦЕЛОЧИСЛЕННОЙ РАБОТЫ ПО ДВУМ КОНВЕЙЕРАМ ПОД HMMA ===\n");
  printf("%-52s %9s %8s %14s\n","состав","HMMA/такт","от пика","целых/такт/SM");
  T5((k_peak<<<B,T>>>(o))) base=nh/cyc/80.0;
  printf("%-52s %9.3f %8.3f %14s\n","потолок (целых нет)",base,1.0,"--");
  T5((k_i_only<<<B,T>>>(o,0x01020304u)))
  printf("%-52s %9.3f %8.3f %14.3f\n","16 int32 на 8 HMMA (2/HMMA, рабочая точка)",nh/cyc/80.0,(nh/cyc/80.0)/base,ni*16/cyc/80.0);
  T5((k_split<<<B,T>>>(o,0x01020304u,1.0f)))
  printf("%-52s %9.3f %8.3f %14.3f\n","та же работа РАЗНЕСЕНА: 8 int32 + 8 fp32x2",nh/cyc/80.0,(nh/cyc/80.0)/base,ni*(8+16)/cyc/80.0);
  T5((k_both<<<B,T>>>(o,0x01020304u,1.0f)))
  printf("%-52s %9.3f %8.3f %14.3f\n","ВДВОЕ больше: 16 int32 + 16 fp32x2",nh/cyc/80.0,(nh/cyc/80.0)/base,ni*(16+32)/cyc/80.0);
  printf("\n  'целых/такт/SM' считает fp32-поля как ДВА целых сложения на инструкцию\n");
  return 0; }
