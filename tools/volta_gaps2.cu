// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// ОСТАТОК СПИСКА: (6-перемер) модель выдачи ДЛИННЫМ телом, (5) раздел L1/разделяемая,
// (1б) именованные барьеры при РАЗНОЙ нагрузке, расхождение полос, red против atom, округление.
#include <cstdio>
#include <cuda_runtime.h>
#define NIT 512
#define X8(I)  I I I I I I I I
#define X32(I) X8(I) X8(I) X8(I) X8(I)
// --- (6) 32 независимые цепи, тело развёрнуто: накладные цикла ~2 из 34
__global__ void k_f32(float* o, float a){ float x[8]; 
  #pragma unroll
  for(int j=0;j<8;j++) x[j]=a+j;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    #pragma unroll
    for(int j=0;j<8;j++){ asm volatile("add.f32 %0,%0,%1;":"+f"(x[j]):"f"(a));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(x[j]):"f"(a));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(x[j]):"f"(a));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(x[j]):"f"(a)); } }
  float s=0; 
  #pragma unroll
  for(int j=0;j<8;j++) s+=x[j]; o[threadIdx.x]=s; }
__global__ void k_int(unsigned* o, unsigned a){ unsigned y[8];
  #pragma unroll
  for(int j=0;j<8;j++) y[j]=a+j;
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    #pragma unroll
    for(int j=0;j<8;j++){ asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y[j]):"r"(a));
                          asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y[j]):"r"(a));
                          asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y[j]):"r"(a));
                          asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y[j]):"r"(a)); } }
  unsigned s=0;
  #pragma unroll
  for(int j=0;j<8;j++) s+=y[j]; o[threadIdx.x]=s; }
__global__ void k_mix(float* o, unsigned* p, float a, unsigned b){ float x[4]; unsigned y[4];
  #pragma unroll
  for(int j=0;j<4;j++){ x[j]=a+j; y[j]=b+j; }
  #pragma unroll 1
  for(int i=0;i<NIT;i++){
    #pragma unroll
    for(int j=0;j<4;j++){ asm volatile("add.f32 %0,%0,%1;":"+f"(x[j]):"f"(a));
                          asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y[j]):"r"(b));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(x[j]):"f"(a));
                          asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y[j]):"r"(b));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(x[j]):"f"(a));
                          asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y[j]):"r"(b));
                          asm volatile("add.f32 %0,%0,%1;":"+f"(x[j]):"f"(a));
                          asm volatile("lop3.b32 %0,%0,%1,%1,0x96;":"+r"(y[j]):"r"(b)); } }
  float s=0; unsigned t=0;
  #pragma unroll
  for(int j=0;j<4;j++){ s+=x[j]; t+=y[j]; } o[threadIdx.x]=s; p[threadIdx.x]=t; }
// --- (5) потоковое чтение, чтобы крутить раздел L1/разделяемая
__global__ void k_stream(const float* g, float* o, size_t n){ extern __shared__ float sh[];
  if(threadIdx.x==1024) sh[0]=1.0f;  // формально задействуем разделяемую
  float a=0; size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=st){ float v; asm volatile("ld.global.f32 %0,[%1];":"=f"(v):"l"(g+i)); a+=v; }
  o[threadIdx.x&255]=a; }
// --- расхождение полос
__global__ void k_div_uni(float* o){ float a=0;
  #pragma unroll 1
  for(int i=0;i<NIT*8;i++){ if(blockIdx.x>=0) a+=1.0f; else a+=2.0f; } o[threadIdx.x]=a; }
__global__ void k_div_lane(float* o){ float a=0; int l=threadIdx.x&31;
  #pragma unroll 1
  for(int i=0;i<NIT*8;i++){ if(l&1) a+=1.0f; else a+=2.0f; } o[threadIdx.x]=a; }
__global__ void k_div_32(float* o){ float a=0; int l=threadIdx.x&31;
  #pragma unroll 1
  for(int i=0;i<NIT*8;i++){ switch(l&7){ case 0:a+=1;break; case 1:a+=2;break; case 2:a+=3;break;
    case 3:a+=4;break; case 4:a+=5;break; case 5:a+=6;break; case 6:a+=7;break; default:a+=8; } }
  o[threadIdx.x]=a; }
// --- red против atom
__global__ void k_red(float* g){ 
  #pragma unroll 1
  for(int i=0;i<NIT;i++) asm volatile("red.global.add.f32 [%0],%1;"::"l"(g+((threadIdx.x*17+i)&1023)),"f"(1.0f)); }
__global__ void k_atom(float* g){ float r;
  #pragma unroll 1
  for(int i=0;i<NIT;i++) asm volatile("atom.global.add.f32 %0,[%1],%2;":"=f"(r):"l"(g+((threadIdx.x*17+i)&1023)),"f"(1.0f)); }
// --- режимы округления
#define ROUND(nm, md) __global__ void k_r_##nm(float* o, float a, float b){ float x[8];         \
  _Pragma("unroll") for(int j=0;j<8;j++) x[j]=a+j;                                              \
  _Pragma("unroll 1") for(int i=0;i<NIT;i++){ _Pragma("unroll") for(int j=0;j<8;j++){           \
    asm volatile("fma." #md ".f32 %0,%0,%1,%2;":"+f"(x[j]):"f"(a),"f"(b));                      \
    asm volatile("fma." #md ".f32 %0,%0,%1,%2;":"+f"(x[j]):"f"(a),"f"(b));                      \
    asm volatile("fma." #md ".f32 %0,%0,%1,%2;":"+f"(x[j]):"f"(a),"f"(b));                      \
    asm volatile("fma." #md ".f32 %0,%0,%1,%2;":"+f"(x[j]):"f"(a),"f"(b)); } }                  \
  float s=0; _Pragma("unroll") for(int j=0;j<8;j++) s+=x[j]; o[threadIdx.x]=s; }
ROUND(rn,rn) ROUND(rz,rz) ROUND(rm,rm)
// --- (1б) именованные барьеры при РАЗНОЙ нагрузке групп
__global__ void k_bar_asym_full(float* o){ float a=threadIdx.x; int slow=(threadIdx.x<128);
  #pragma unroll 1
  for(int i=0;i<256;i++){ int w = slow?64:8; for(int k=0;k<w;k++) a=fmaf(a,1.0001f,1.0f);
    asm volatile("bar.sync 0;"); } o[threadIdx.x]=a; }
__global__ void k_bar_asym_named(float* o){ float a=threadIdx.x; int slow=(threadIdx.x<128);
  #pragma unroll 1
  for(int i=0;i<256;i++){ int w = slow?64:8; for(int k=0;k<w;k++) a=fmaf(a,1.0001f,1.0f);
    if(slow) asm volatile("bar.sync 1, 128;"); else asm volatile("bar.sync 2, 128;"); } o[threadIdx.x]=a; }
int main(){ cudaSetDevice(0); float* o; unsigned* p; cudaMalloc(&o,8192*4); cudaMalloc(&p,8192*4);
  size_t N=64ull*1024*1024; float* g; cudaMalloc(&g,N*4); cudaMemset(g,0,N*4);
  int B=80,T=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms; double cyc;
  #define T5(call) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cyc=(double)ms/5.0*1e-3*1.380e9; }
  double ni=(double)B*T*NIT*32;
  printf("=== (6) МОДЕЛЬ ВЫДАЧИ, перемер: тело 32 инструкции, накладные цикла ~6%% ===\n");
  T5((k_f32<<<B,T>>>(o,1.0f)))   printf("  только add.f32       %6.2f инстр.варпа/такт/SM  (%5.1f оп/такт/SM)\n",ni/32.0/cyc/80.0,ni/cyc/80.0);
  T5((k_int<<<B,T>>>(p,3u)))     printf("  только lop3.b32      %6.2f инстр.варпа/такт/SM  (%5.1f оп/такт/SM)\n",ni/32.0/cyc/80.0,ni/cyc/80.0);
  T5((k_mix<<<B,T>>>(o,p,1.0f,3u))) printf("  СМЕСЬ fp32+lop3      %6.2f инстр.варпа/такт/SM  (%5.1f оп/такт/SM)\n",ni/32.0/cyc/80.0,ni/cyc/80.0);
  printf("  (если смесь ВЫШЕ каждого по отдельности -- конвейеры складываются; если нет -- выдача одна)\n");
  printf("\n=== (5) РАЗДЕЛ 128 КБ МЕЖДУ L1 И РАЗДЕЛЯЕМОЙ ПАМЯТЬЮ ===\n");
  double by=(double)N*4;
  for(int cv : {0, 25, 50, 75, 100}) {
    cudaFuncSetAttribute(k_stream, cudaFuncAttributePreferredSharedMemoryCarveout, cv);
    T5((k_stream<<<B*4,T,1024>>>(g,o,N)))
    printf("  доля разделяемой %3d%% (L1 = %3d%%)   %7.1f ГБ/с\n",cv,100-cv,by/(ms/5.0*1e-3)/1e9); }
  printf("\n=== РАСХОЖДЕНИЕ ПОЛОС ===\n");
  double nd=(double)B*T*NIT*8;
  T5((k_div_uni<<<B,T>>>(o)))  double du=cyc/(NIT*8.0); printf("  однородная ветвь           %7.2f тактов/итер\n",du);
  T5((k_div_lane<<<B,T>>>(o))) printf("  расхождение надвое         %7.2f тактов/итер  (x%.2f)\n",cyc/(NIT*8.0),cyc/(NIT*8.0)/du);
  T5((k_div_32<<<B,T>>>(o)))   printf("  расхождение на 8 путей     %7.2f тактов/итер  (x%.2f)\n",cyc/(NIT*8.0),cyc/(NIT*8.0)/du);
  printf("\n=== red БЕЗ ВОЗВРАТА ПРОТИВ atom ===\n");
  double na=(double)B*T*NIT;
  T5((k_red<<<B,T>>>(g)))  double rr=na/cyc/80.0; printf("  red.global.add.f32         %7.2f оп/такт/SM\n",rr);
  T5((k_atom<<<B,T>>>(g))) printf("  atom.global.add.f32        %7.2f оп/такт/SM  (red быстрее в %.2f раза)\n",na/cyc/80.0,rr/(na/cyc/80.0));
  printf("\n=== РЕЖИМЫ ОКРУГЛЕНИЯ ===\n");
  T5((k_r_rn<<<B,T>>>(o,1.0001f,0.5f))) printf("  fma.rn (к ближайшему)      %7.2f оп/такт/SM\n",ni/cyc/80.0);
  T5((k_r_rz<<<B,T>>>(o,1.0001f,0.5f))) printf("  fma.rz (к нулю)            %7.2f оп/такт/SM\n",ni/cyc/80.0);
  T5((k_r_rm<<<B,T>>>(o,1.0001f,0.5f))) printf("  fma.rm (к минус беск.)     %7.2f оп/такт/SM\n",ni/cyc/80.0);
  printf("\n=== (1б) ИМЕНОВАННЫЕ БАРЬЕРЫ ПРИ РАЗНОЙ НАГРУЗКЕ ГРУПП (64 против 8 единиц работы) ===\n");
  T5((k_bar_asym_full<<<B,T>>>(o)))  double bf=ms/5.0; printf("  общий bar.sync 0           %7.3f мс\n",bf);
  T5((k_bar_asym_named<<<B,T>>>(o))) printf("  именованные 1/2            %7.3f мс  (x%.3f)\n",ms/5.0,bf/(ms/5.0));
  return 0; }
