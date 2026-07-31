// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// 1) КЭШ ПОВТОРНОГО ИСПОЛЬЗОВАНИЯ ОПЕРАНДОВ: один регистр в ТОЙ ЖЕ позиции у соседних инструкций
//    -> флаг .reuse, чтение из регистрового файла не выполняется.
// 2) БАНКИ РЕГИСТРОВОГО ФАЙЛА: операнды одной инструкции из одного банка стоят лишний такт.
// 3) УПРАВЛЕНИЕ ШИНОЙ: подсказки кэша .ca/.cg/.cs/.lu на потоковых данных.
#include <cstdio>
#include <cuda_runtime.h>
#define NIT 4096
// ПОВТОР: a и b одни и те же во ВСЕХ восьми инструкциях, позиции операндов совпадают
__global__ void k_reuse(float* o, float a, float b){
  float x0=a,x1=a,x2=a,x3=a,x4=a,x5=a,x6=a,x7=a;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x0):"f"(a),"f"(b));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x1):"f"(a),"f"(b));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x2):"f"(a),"f"(b));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x3):"f"(a),"f"(b));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x4):"f"(a),"f"(b));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x5):"f"(a),"f"(b));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x6):"f"(a),"f"(b));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x7):"f"(a),"f"(b)); }
  o[threadIdx.x]=x0+x1+x2+x3+x4+x5+x6+x7; }
// РАЗНЫЕ: у каждой инструкции свои источники -> повторно использовать нечего
__global__ void k_distinct(float* o, float a, float b){
  float a0=a,a1=a+1,a2=a+2,a3=a+3,a4=a+4,a5=a+5,a6=a+6,a7=a+7;
  float b0=b,b1=b+1,b2=b+2,b3=b+3,b4=b+4,b5=b+5,b6=b+6,b7=b+7;
  float x0=a,x1=a,x2=a,x3=a,x4=a,x5=a,x6=a,x7=a;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x0):"f"(a0),"f"(b0));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x1):"f"(a1),"f"(b1));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x2):"f"(a2),"f"(b2));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x3):"f"(a3),"f"(b3));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x4):"f"(a4),"f"(b4));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x5):"f"(a5),"f"(b5));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x6):"f"(a6),"f"(b6));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x7):"f"(a7),"f"(b7)); }
  o[threadIdx.x]=x0+x1+x2+x3+x4+x5+x6+x7; }
// ЧАСТИЧНЫЙ повтор: общий только ОДИН операнд (как в мейнлупе -- общий A, разные B)
__global__ void k_half(float* o, float a, float b){
  float b0=b,b1=b+1,b2=b+2,b3=b+3,b4=b+4,b5=b+5,b6=b+6,b7=b+7;
  float x0=a,x1=a,x2=a,x3=a,x4=a,x5=a,x6=a,x7=a;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x0):"f"(a),"f"(b0));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x1):"f"(a),"f"(b1));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x2):"f"(a),"f"(b2));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x3):"f"(a),"f"(b3));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x4):"f"(a),"f"(b4));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x5):"f"(a),"f"(b5));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x6):"f"(a),"f"(b6));
    asm volatile("fma.rn.f32 %0,%1,%2,%0;":"+f"(x7):"f"(a),"f"(b7)); }
  o[threadIdx.x]=x0+x1+x2+x3+x4+x5+x6+x7; }
// ШИНА: потоковое чтение массива, больше L2 (6 МБ) -- подсказки .ca / .cg / .cs / .lu
#define STREAM(nm, hint) __global__ void ks_##nm(const float* g, float* o, size_t n){    \
  float a=0; size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x; \
  for(; i<n; i+=st){ float v; asm volatile("ld.global" hint ".f32 %0,[%1];":"=f"(v):"l"(g+i)); a+=v; } \
  o[threadIdx.x&255]=a; }
STREAM(ca,".ca") STREAM(cg,".cg") STREAM(cs,".cs") STREAM(lu,".lu") STREAM(cv,".cv")
int main(){ cudaSetDevice(0); float* o; cudaMalloc(&o,4096*4);
  size_t N = 64ull*1024*1024; float* g; cudaMalloc(&g,N*4); cudaMemset(g,0,N*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms;
  double ops=(double)B*T*NIT*8;
  #define RUN(nm,call,fmt,val) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); \
    double cyc=(double)ms/5.0*1e-3*1.380e9; printf(fmt,nm,(val)); }
  printf("=== КЭШ ПОВТОРНОГО ИСПОЛЬЗОВАНИЯ ОПЕРАНДОВ (8 независимых FFMA) ===\n");
  #define R(nm,call) RUN(nm,call,"  %-38s %7.2f IPC/SM\n",ops/cyc/80.0)
  R("оба операнда ОБЩИЕ (.reuse x2)",(k_reuse<<<B,T>>>(o,1.0f,1.001f)))
  R("общий ОДИН операнд (.reuse x1)",(k_half<<<B,T>>>(o,1.0f,1.001f)))
  R("все операнды РАЗНЫЕ (без .reuse)",(k_distinct<<<B,T>>>(o,1.0f,1.001f)))
  printf("\n=== УПРАВЛЕНИЕ ШИНОЙ: потоковое чтение 256 МБ (больше L2 = 6 МБ) ===\n");
  double bytes=(double)N*4;
  #define RB(nm,call) RUN(nm,call,"  %-38s %7.1f ГБ/с\n",bytes/(ms/5.0*1e-3)/1e9)
  RB("ld.global.ca (кэшировать всё)",(ks_ca<<<B*4,T>>>(g,o,N)))
  RB("ld.global.cg (мимо L1, в L2)",(ks_cg<<<B*4,T>>>(g,o,N)))
  RB("ld.global.cs (потоковый, вытеснять)",(ks_cs<<<B*4,T>>>(g,o,N)))
  RB("ld.global.lu (последнее использ.)",(ks_lu<<<B*4,T>>>(g,o,N)))
  RB("ld.global.cv (не кэшировать)",(ks_cv<<<B*4,T>>>(g,o,N)))
  return 0; }
