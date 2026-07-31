// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// ПРОБЕЛЫ ОБХОДА: (1) именованные барьеры, (2) цена барьера, (3) ширина LDS,
// (4) программная предвыборка, (5) модель выдачи -- сколько инструкций за такт на планировщик.
#include <cstdio>
#include <cuda_runtime.h>
#define NIT 2048
// --- (2) цена барьера: общий на весь блок
__global__ void k_bar_full(float* o){ float a=threadIdx.x;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ asm volatile("bar.sync 0;"); a+=1.0f; }
  o[threadIdx.x]=a; }
// --- (1) именованный барьер: два независимых полублока по 4 варпа
__global__ void k_bar_named(float* o){ float a=threadIdx.x;
  int grp = (threadIdx.x < 128) ? 1 : 2;   // барьеры 1 и 2, по 128 нитей каждый
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ if(grp==1) asm volatile("bar.sync 1, 128;"); else asm volatile("bar.sync 2, 128;"); a+=1.0f; }
  o[threadIdx.x]=a; }
// --- база без барьера
__global__ void k_bar_none(float* o){ float a=threadIdx.x;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ asm volatile("":::"memory"); a+=1.0f; }
  o[threadIdx.x]=a; }
// --- (3) ширина обращения к разделяемой памяти: 32 / 64 / 128 бит
#define LDSW(nm, instr, TY, decl, acc)                                                  \
__global__ void k_lds_##nm(float* o){ __shared__ float sm[4096];                         \
  for(int i=threadIdx.x;i<4096;i+=blockDim.x) sm[i]=(float)i; __syncthreads();           \
  float s=0; unsigned b=(threadIdx.x*4u)&4032u;                                          \
  _Pragma("unroll 1") for(int i=0;i<NIT;i++){ decl;                                      \
    asm volatile(instr :: "r"((unsigned)__cvta_generic_to_shared(&sm[b])));               \
    acc; b=(b+128u)&4032u; } o[threadIdx.x]=s; }
__global__ void k_lds32(float* o){ __shared__ float sm[4096];
  for(int i=threadIdx.x;i<4096;i+=blockDim.x) sm[i]=(float)i; __syncthreads();
  float s=0; unsigned b=(threadIdx.x*4u)&4032u;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ float v0,v1,v2,v3;
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v0):"r"((unsigned)__cvta_generic_to_shared(&sm[b])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v1):"r"((unsigned)__cvta_generic_to_shared(&sm[b+1])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v2):"r"((unsigned)__cvta_generic_to_shared(&sm[b+2])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v3):"r"((unsigned)__cvta_generic_to_shared(&sm[b+3])));
    s+=v0+v1+v2+v3; b=(b+128u)&4032u; } o[threadIdx.x]=s; }
__global__ void k_lds64(float* o){ __shared__ float sm[4096];
  for(int i=threadIdx.x;i<4096;i+=blockDim.x) sm[i]=(float)i; __syncthreads();
  float s=0; unsigned b=(threadIdx.x*4u)&4032u;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ float2 a0,a1;
    asm volatile("ld.shared.v2.f32 {%0,%1},[%2];":"=f"(a0.x),"=f"(a0.y):"r"((unsigned)__cvta_generic_to_shared(&sm[b])));
    asm volatile("ld.shared.v2.f32 {%0,%1},[%2];":"=f"(a1.x),"=f"(a1.y):"r"((unsigned)__cvta_generic_to_shared(&sm[b+2])));
    s+=a0.x+a0.y+a1.x+a1.y; b=(b+128u)&4032u; } o[threadIdx.x]=s; }
__global__ void k_lds128(float* o){ __shared__ float sm[4096];
  for(int i=threadIdx.x;i<4096;i+=blockDim.x) sm[i]=(float)i; __syncthreads();
  float s=0; unsigned b=(threadIdx.x*4u)&4032u;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ float4 a0;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(a0.x),"=f"(a0.y),"=f"(a0.z),"=f"(a0.w)
      :"r"((unsigned)__cvta_generic_to_shared(&sm[b])));
    s+=a0.x+a0.y+a0.z+a0.w; b=(b+128u)&4032u; } o[threadIdx.x]=s; }
// --- (4) программная предвыборка на потоковом чтении
__global__ void k_pf(const float* g, float* o, size_t n, int use){
  float a=0; size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=st){ if(use){ size_t j=i+st*8; if(j<n) asm volatile("prefetch.global.L2 [%0];"::"l"(g+j)); }
    float v; asm volatile("ld.global.f32 %0,[%1];":"=f"(v):"l"(g+i)); a+=v; }
  o[threadIdx.x&255]=a; }
// --- (5) модель выдачи: fp32 и целочисленный конвейер вместе
__global__ void k_iss_f(float* o, float a){ float x0=a,x1=a,x2=a,x3=a;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x1):"f"(a));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x2):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x3):"f"(a)); }
  o[threadIdx.x]=x0+x1+x2+x3; }
__global__ void k_iss_i(unsigned* o, unsigned a){ unsigned y0=a,y1=a,y2=a,y3=a;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y0):"r"(a)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y1):"r"(a));
    asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y2):"r"(a)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y3):"r"(a)); }
  o[threadIdx.x]=y0+y1+y2+y3; }
__global__ void k_iss_mix(float* o, unsigned* p, float a, unsigned b){
  float x0=a,x1=a; unsigned y0=b,y1=b;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(a)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y0):"r"(b));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x1):"f"(a)); asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y1):"r"(b)); }
  o[threadIdx.x]=x0+x1; p[threadIdx.x]=y0+y1; }
int main(){ cudaSetDevice(0); float* o; unsigned* p; cudaMalloc(&o,4096*4); cudaMalloc(&p,4096*4);
  size_t N=64ull*1024*1024; float* g; cudaMalloc(&g,N*4); cudaMemset(g,0,N*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms;
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); }
  double cyc;
  #define SHOW(nm,val,u) printf("  %-40s %8.2f %s\n",nm,val,u)
  printf("=== (2) ЦЕНА БАРЬЕРА и (1) ИМЕНОВАННЫЕ БАРЬЕРЫ (8 варпов в блоке) ===\n");
  T5((k_bar_none<<<B,T>>>(o))) cyc=ms/5.0*1e-3*1.380e9; double base=cyc/NIT;
  SHOW("без барьера (база)",base,"тактов/итер");
  T5((k_bar_full<<<B,T>>>(o))) cyc=ms/5.0*1e-3*1.380e9; double bf=cyc/NIT;
  SHOW("bar.sync 0 (весь блок)",bf,"тактов/итер"); SHOW("  -> ЧИСТАЯ ЦЕНА БАРЬЕРА",bf-base,"тактов");
  T5((k_bar_named<<<B,T>>>(o))) cyc=ms/5.0*1e-3*1.380e9; double bn=cyc/NIT;
  SHOW("bar.sync 1/2, 128 (полублоки)",bn,"тактов/итер"); SHOW("  -> ЧИСТАЯ ЦЕНА",bn-base,"тактов");
  printf("  -> именованный ДЕШЕВЛЕ общего в %.2f раза\n",(bf-base)/((bn-base)>0?(bn-base):1e-9));
  printf("\n=== (3) ШИРИНА ОБРАЩЕНИЯ К РАЗДЕЛЯЕМОЙ ПАМЯТИ ===\n");
  double nb=(double)B*T*NIT*16.0;
  T5((k_lds32<<<B,T>>>(o))) cyc=ms/5.0*1e-3*1.380e9;
  printf("  4 x ld.shared.f32  (4 инстр./16 Б)  %7.2f Б/такт/SM, %6.2f инстр/такт/SM\n",nb/cyc/80.0,(double)B*T*NIT*4/cyc/80.0);
  T5((k_lds64<<<B,T>>>(o))) cyc=ms/5.0*1e-3*1.380e9;
  printf("  2 x ld.shared.v2   (2 инстр./16 Б)  %7.2f Б/такт/SM, %6.2f инстр/такт/SM\n",nb/cyc/80.0,(double)B*T*NIT*2/cyc/80.0);
  T5((k_lds128<<<B,T>>>(o))) cyc=ms/5.0*1e-3*1.380e9;
  printf("  1 x ld.shared.v4   (1 инстр./16 Б)  %7.2f Б/такт/SM, %6.2f инстр/такт/SM\n",nb/cyc/80.0,(double)B*T*NIT*1/cyc/80.0);
  printf("\n=== (4) ПРОГРАММНАЯ ПРЕДВЫБОРКА ===\n");
  double by=(double)N*4;
  T5((k_pf<<<B*4,T>>>(g,o,N,0))) SHOW("без предвыборки",by/(ms/5.0*1e-3)/1e9,"ГБ/с");
  T5((k_pf<<<B*4,T>>>(g,o,N,1))) SHOW("prefetch.global.L2 на 8 шагов вперёд",by/(ms/5.0*1e-3)/1e9,"ГБ/с");
  printf("\n=== (5) МОДЕЛЬ ВЫДАЧИ: сколько ИНСТРУКЦИЙ ВАРПА за такт на SM ===\n");
  double ops4=(double)B*T*NIT*4;
  T5((k_iss_f<<<B,T>>>(o,1.0f))) cyc=ms/5.0*1e-3*1.380e9;
  printf("  только add.f32                    %6.2f инстр.варпа/такт/SM\n",ops4/32.0/cyc/80.0);
  T5((k_iss_i<<<B,T>>>(p,3u))) cyc=ms/5.0*1e-3*1.380e9;
  printf("  только lop3.b32                   %6.2f инстр.варпа/такт/SM\n",ops4/32.0/cyc/80.0);
  T5((k_iss_mix<<<B,T>>>(o,p,1.0f,3u))) cyc=ms/5.0*1e-3*1.380e9;
  printf("  смесь fp32+lop3 (по 2)            %6.2f инстр.варпа/такт/SM  <- предел планировщиков\n",ops4/32.0/cyc/80.0);
  printf("  (4 планировщика x 1 выдача = 4.00 -- если смесь упирается в 4, модель подтверждена)\n");
  return 0; }
