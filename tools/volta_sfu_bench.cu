// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// SFU: темп/задержка всего семейства MUFU, конкуренция за выдачу с FP32, и обходной маршрут --
// экспонента битовым трюком (то же семейство приёмов, что OR-разворот int8).
#include <cstdio>
#include <cuda_runtime.h>
#define NIT 2048
#define THR8(nm, op, T, C, init) __global__ void k_##nm(T* o, T a){                         \
  T x0=init,x1=init,x2=init,x3=init,x4=init,x5=init,x6=init,x7=init;                    \
  _Pragma("unroll 1") for(int i=0;i<NIT;i++){                                           \
    asm volatile(#op " %0,%0;":"+" C(x0)); asm volatile(#op " %0,%0;":"+" C(x1));       \
    asm volatile(#op " %0,%0;":"+" C(x2)); asm volatile(#op " %0,%0;":"+" C(x3));       \
    asm volatile(#op " %0,%0;":"+" C(x4)); asm volatile(#op " %0,%0;":"+" C(x5));       \
    asm volatile(#op " %0,%0;":"+" C(x6)); asm volatile(#op " %0,%0;":"+" C(x7)); }     \
  o[threadIdx.x]=x0+x1+x2+x3+x4+x5+x6+x7; }
#define LAT1(nm, op, T, C, init) __global__ void kl_##nm(T* o, T a){ T x=init;              \
  _Pragma("unroll 1") for(int i=0;i<NIT*8;i++) asm volatile(#op " %0,%0;":"+" C(x));    \
  o[threadIdx.x]=x; }
#define SFU(nm, op) THR8(nm,op,float,"f",0.5f) LAT1(nm,op,float,"f",0.5f)
SFU(ex2, ex2.approx.f32) SFU(rcp, rcp.approx.f32) SFU(rsq, rsqrt.approx.f32)
SFU(lg2, lg2.approx.f32) SFU(sinf, sin.approx.f32)  SFU(sqrtf, sqrt.approx.f32)
// СМЕСЬ fp32 + SFU: если SFU -- отдельная мощность, суммарный темп ВЫШЕ чистого fp32.
__global__ void k_mix(float* o, float a){
  float f0=a,f1=a+1,f2=a+2,f3=a+3, s0=0.5f,s1=0.6f,s2=0.7f,s3=0.8f;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    asm volatile("add.f32 %0,%0,%1;":"+f"(f0):"f"(a)); asm volatile("ex2.approx.f32 %0,%0;":"+f"(s0));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f1):"f"(a)); asm volatile("ex2.approx.f32 %0,%0;":"+f"(s1));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f2):"f"(a)); asm volatile("ex2.approx.f32 %0,%0;":"+f"(s2));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f3):"f"(a)); asm volatile("ex2.approx.f32 %0,%0;":"+f"(s3)); }
  o[threadIdx.x]=f0+f1+f2+f3+s0+s1+s2+s3; }
__global__ void k_f32(float* o, float a){ float x0=a,x1=a+1,x2=a+2,x3=a+3,x4=a+4,x5=a+5,x6=a+6,x7=a+7;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x1):"f"(a));
   asm volatile("add.f32 %0,%0,%1;":"+f"(x2):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x3):"f"(a));
   asm volatile("add.f32 %0,%0,%1;":"+f"(x4):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x5):"f"(a));
   asm volatile("add.f32 %0,%0,%1;":"+f"(x6):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x7):"f"(a)); }
  o[threadIdx.x]=x0+x1+x2+x3+x4+x5+x6+x7; }
// ОБХОД: 2^x через поле экспоненты. Одно FFMA + одно переистолкование бит (наш приём §1).
__device__ __forceinline__ float ex2_bits(float x){
  float t = fmaf(x, 8388608.0f, 1064866805.0f);            // 2^23 * (x + 126.94269504)
  int i = (int)t; return __int_as_float(i); }
__global__ void k_bits(float* o, float a){ float x0=0.5f,x1=0.6f,x2=0.7f,x3=0.8f,x4=0.9f,x5=1.1f,x6=1.2f,x7=1.3f;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ x0=ex2_bits(x0*0.5f); x1=ex2_bits(x1*0.5f); x2=ex2_bits(x2*0.5f); x3=ex2_bits(x3*0.5f);
    x4=ex2_bits(x4*0.5f); x5=ex2_bits(x5*0.5f); x6=ex2_bits(x6*0.5f); x7=ex2_bits(x7*0.5f); }
  o[threadIdx.x]=x0+x1+x2+x3+x4+x5+x6+x7; }
__global__ void k_acc(float* o){ // точность обхода против MUFU на диапазоне софтмакса [-16,0]
  float x = -16.0f + (float)threadIdx.x * (16.0f/255.0f);
  float r; asm volatile("ex2.approx.f32 %0,%1;":"=f"(r):"f"(x));
  float b = ex2_bits(x); o[threadIdx.x] = (r!=0.0f)? fabsf(b-r)/r : 0.0f; }
int main(){ cudaSetDevice(0); float* o; cudaMalloc(&o,4096*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms;
  double ops=(double)B*T*NIT*8;
  #define RUN(nm,call,n) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); \
    double cyc=(double)ms/5.0*1e-3*1.380e9; printf("  %-30s %7.2f IPC/SM  %7.1f тактов\n",nm,(n)/cyc/80.0,cyc/(NIT*8.0)); }
  printf("=== SFU (V100, 80 блоков x 256 нитей) ===\n");
  #define P(op,nm) RUN(nm " темп",(k_##op<<<B,T>>>(o,1.0f)),ops) RUN(nm " задержка",(kl_##op<<<B,T>>>(o,1.0f)),ops)
  P(ex2,"ex2  ") P(rcp,"rcp  ") P(rsq,"rsqrt")
  P(lg2,"lg2  ") P(sinf,"sin  ") P(sqrtf,"sqrt ")
  printf("\n  --- конкурирует ли SFU за выдачу с FP32 ---\n");
  RUN("чистый add.f32",(k_f32<<<B,T>>>(o,1.0f)),ops)
  RUN("смесь add.f32+ex2 (половина)",(k_mix<<<B,T>>>(o,1.0f)),ops/2)
  printf("\n  --- ОБХОД: 2^x битовым трюком вместо MUFU ---\n");
  RUN("ex2_bits (ffma + переистолк.)",(k_bits<<<B,T>>>(o,1.0f)),ops)
  k_acc<<<1,256>>>(o); cudaDeviceSynchronize(); float h[256]; cudaMemcpy(h,o,256*4,cudaMemcpyDeviceToHost);
  float mx=0,sum=0; for(int i=0;i<256;i++){ mx=fmaxf(mx,h[i]); sum+=h[i]; }
  printf("     точность на [-16,0] (диапазон софтмакса): макс отн.ошибка %.4f, средняя %.4f\n",mx,sum/256);
  return 0; }
