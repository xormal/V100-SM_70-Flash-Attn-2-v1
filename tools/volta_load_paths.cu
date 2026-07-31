// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// ПУТИ ЗАГРУЗКИ: разделяемая память, константы (LDC), текстура, глобальная .nc.
// Вопрос не в темпе каждого, а в СМЕСИ: даёт ли альтернативный путь мощность СВЕРХ MIO.
#include <cstdio>
#include <cuda_runtime.h>
#define NIT 1024
__constant__ float cbuf[2048];
__global__ void k_lds(float* o){ __shared__ float sm[2048];
  for(int i=threadIdx.x;i<2048;i+=blockDim.x) sm[i]=(float)i; __syncthreads();
  float a=0; int b=(threadIdx.x*4)&2047;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ float v0,v1,v2,v3,v4,v5,v6,v7;
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v0):"r"((unsigned)__cvta_generic_to_shared(&sm[(b   )&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v1):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+ 64)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v2):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+128)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v3):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+192)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v4):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+256)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v5):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+320)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v6):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+384)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v7):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+448)&2047])));
    a+=v0+v1+v2+v3+v4+v5+v6+v7; b=(b+1)&2047; }
  o[threadIdx.x]=a; }
// LDC однородный: адрес ОДИН на весь варп -> широковещание из кэша констант
__global__ void k_ldc_u(float* o){ float a=0; int b=0;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ a+=cbuf[b]+cbuf[b+64]+cbuf[b+128]+cbuf[b+192]
                           +cbuf[b+256]+cbuf[b+320]+cbuf[b+384]+cbuf[b+448]; b=(b+1)&511; }
  o[threadIdx.x]=a; }
// LDC расходящийся: адрес зависит от полосы -> сериализация (знаем, что плохо; меряем НАСКОЛЬКО)
__global__ void k_ldc_d(float* o){ float a=0; int b=threadIdx.x&511;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ a+=cbuf[b]+cbuf[(b+64)&511]+cbuf[(b+128)&511]+cbuf[(b+192)&511]
      +cbuf[(b+256)&511]+cbuf[(b+320)&511]+cbuf[(b+384)&511]+cbuf[(b+448)&511]; b=(b+1)&511; }
  o[threadIdx.x]=a; }
__global__ void k_ldg(const float* __restrict__ g, float* o){ float a=0; int b=(threadIdx.x*4)&2047;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ a+=__ldg(g+b)+__ldg(g+((b+64)&2047))+__ldg(g+((b+128)&2047))+__ldg(g+((b+192)&2047))
      +__ldg(g+((b+256)&2047))+__ldg(g+((b+320)&2047))+__ldg(g+((b+384)&2047))+__ldg(g+((b+448)&2047)); b=(b+1)&2047; }
  o[threadIdx.x]=a; }
__global__ void k_tex(cudaTextureObject_t t, float* o){ float a=0; int b=(threadIdx.x*4)&2047;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ a+=tex1Dfetch<float>(t,b)+tex1Dfetch<float>(t,(b+64)&2047)
    +tex1Dfetch<float>(t,(b+128)&2047)+tex1Dfetch<float>(t,(b+192)&2047)+tex1Dfetch<float>(t,(b+256)&2047)
    +tex1Dfetch<float>(t,(b+320)&2047)+tex1Dfetch<float>(t,(b+384)&2047)+tex1Dfetch<float>(t,(b+448)&2047); b=(b+1)&2047; }
  o[threadIdx.x]=a; }
// СМЕСИ: половина обращений уходит по альтернативному пути
__global__ void k_mix_ldc(float* o){ __shared__ float sm[2048];
  for(int i=threadIdx.x;i<2048;i+=blockDim.x) sm[i]=(float)i; __syncthreads();
  float a=0; int b=(threadIdx.x*4)&2047; int c=0;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ float v0,v1,v2,v3;
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v0):"r"((unsigned)__cvta_generic_to_shared(&sm[(b   )&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v1):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+ 64)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v2):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+128)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v3):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+192)&2047])));
    a+=v0+v1+v2+v3+cbuf[c]+cbuf[c+64]+cbuf[c+128]+cbuf[c+192]; b=(b+1)&2047; c=(c+1)&511; }
  o[threadIdx.x]=a; }
__global__ void k_mix_tex(cudaTextureObject_t t, float* o){ __shared__ float sm[2048];
  for(int i=threadIdx.x;i<2048;i+=blockDim.x) sm[i]=(float)i; __syncthreads();
  float a=0; int b=(threadIdx.x*4)&2047;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){ float v0,v1,v2,v3;
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v0):"r"((unsigned)__cvta_generic_to_shared(&sm[(b   )&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v1):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+ 64)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v2):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+128)&2047])));
    asm volatile("ld.shared.f32 %0,[%1];":"=f"(v3):"r"((unsigned)__cvta_generic_to_shared(&sm[(b+192)&2047])));
    a+=v0+v1+v2+v3+tex1Dfetch<float>(t,b)+tex1Dfetch<float>(t,(b+64)&2047)
      +tex1Dfetch<float>(t,(b+128)&2047)+tex1Dfetch<float>(t,(b+192)&2047); b=(b+1)&2047; }
  o[threadIdx.x]=a; }
int main(){ cudaSetDevice(0); float* o; float* g; cudaMalloc(&o,4096*4); cudaMalloc(&g,2048*4);
  float h[2048]; for(int i=0;i<2048;i++) h[i]=(float)i;
  cudaMemcpy(g,h,2048*4,cudaMemcpyHostToDevice); cudaMemcpyToSymbol(cbuf,h,2048*4);
  cudaResourceDesc rd{}; rd.resType=cudaResourceTypeLinear; rd.res.linear.devPtr=g;
  rd.res.linear.desc=cudaCreateChannelDesc<float>(); rd.res.linear.sizeInBytes=2048*4;
  cudaTextureDesc td{}; td.readMode=cudaReadModeElementType;
  cudaTextureObject_t tex=0; cudaCreateTextureObject(&tex,&rd,&td,nullptr);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms;
  double ops=(double)B*T*NIT*8;
  #define RUN(nm,call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); { cudaError_t er=cudaGetLastError(); if(er!=cudaSuccess) printf("  ОШИБКА %s: %s\n",nm,cudaGetErrorString(er)); } \
    double cyc=(double)ms/5.0*1e-3*1.380e9; printf("  %-34s %7.2f загр/такт/SM  (%.3f мс)\n",nm,cyc>0?ops/cyc/80.0:0.0,ms/5.0); }
  printf("=== ПУТИ ЗАГРУЗКИ (V100, 80 блоков x 256 нитей, 8 независимых обращений) ===\n");
  RUN("разделяемая память (LDS)",(k_lds<<<B,T>>>(o)))
  RUN("константы, ОДНОРОДНЫЙ адрес",(k_ldc_u<<<B,T>>>(o)))
  RUN("константы, расходящийся адрес",(k_ldc_d<<<B,T>>>(o)))
  RUN("глобальная .nc (__ldg)",(k_ldg<<<B,T>>>(g,o)))
  RUN("текстура (tex1Dfetch)",(k_tex<<<B,T>>>(tex,o)))
  printf("\n  --- СМЕСЬ: половина обращений по альтернативному пути ---\n");
  RUN("LDS + константы (по 4+4)",(k_mix_ldc<<<B,T>>>(o)))
  RUN("LDS + текстура (по 4+4)",(k_mix_tex<<<B,T>>>(tex,o)))
  printf("\n  (смесь считается по ВСЕМ 8 обращениям: выше базовой LDS = путь ДОБАВЛЯЕТ мощность)\n");
  return 0; }
