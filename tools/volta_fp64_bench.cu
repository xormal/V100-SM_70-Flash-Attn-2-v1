// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
// FP64 КАК НЕТРОНУТЫЙ РЕСУРС: три разных вопроса, три разных замера.
//   1. темп и задержка DADD/DFMA
//   2. КОНКУРИРУЕТ ли FP64 за выдачу с FP32 (смесь против чистого) -- главный вопрос
//   3. мантисса 52 бита как контейнер: сколько int8 складывается ОДНОЙ инструкцией точно
#include <cstdio>
#include <cuda_runtime.h>
#define N_IT 4096
__global__ void k_d_thr(double* o, double a){ double x0=a,x1=a+1,x2=a+2,x3=a+3,x4=a+4,x5=a+5,x6=a+6,x7=a+7;
  #pragma unroll 1
  for(int i=0;i<N_IT;i++){ asm volatile("add.f64 %0,%0,%1;":"+d"(x0):"d"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(x1):"d"(a));
    asm volatile("add.f64 %0,%0,%1;":"+d"(x2):"d"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(x3):"d"(a));
    asm volatile("add.f64 %0,%0,%1;":"+d"(x4):"d"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(x5):"d"(a));
    asm volatile("add.f64 %0,%0,%1;":"+d"(x6):"d"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(x7):"d"(a)); }
  o[threadIdx.x]=x0+x1+x2+x3+x4+x5+x6+x7; }
__global__ void k_d_lat(double* o, double a){ double x=a;
  #pragma unroll 1
  for(int i=0;i<N_IT*8;i++) asm volatile("add.f64 %0,%0,%1;":"+d"(x):"d"(a));
  o[threadIdx.x]=x; }
__global__ void k_f_thr(float* o, float a){ float x0=a,x1=a+1,x2=a+2,x3=a+3,x4=a+4,x5=a+5,x6=a+6,x7=a+7;
  #pragma unroll 1
  for(int i=0;i<N_IT;i++){ asm volatile("add.f32 %0,%0,%1;":"+f"(x0):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x1):"f"(a));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x2):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x3):"f"(a));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x4):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x5):"f"(a));
    asm volatile("add.f32 %0,%0,%1;":"+f"(x6):"f"(a)); asm volatile("add.f32 %0,%0,%1;":"+f"(x7):"f"(a)); }
  o[threadIdx.x]=x0+x1+x2+x3+x4+x5+x6+x7; }
// СМЕСЬ: 8 цепочек fp32 + 8 цепочек fp64. Если FP64 добавляет мощность -- суммарный темп ВЫШЕ
// чистого fp32. Если выдача общая -- будет НЕ ВЫШЕ, и вся идея "нетронутого конвейера" мертва.
__global__ void k_mix(float* o, double* od, float a, double b){
  float f0=a,f1=a+1,f2=a+2,f3=a+3,f4=a+4,f5=a+5,f6=a+6,f7=a+7;
  double d0=b,d1=b+1,d2=b+2,d3=b+3,d4=b+4,d5=b+5,d6=b+6,d7=b+7;
  #pragma unroll 1
  for(int i=0;i<N_IT;i++){
    asm volatile("add.f32 %0,%0,%1;":"+f"(f0):"f"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(d0):"d"(b));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f1):"f"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(d1):"d"(b));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f2):"f"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(d2):"d"(b));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f3):"f"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(d3):"d"(b));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f4):"f"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(d4):"d"(b));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f5):"f"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(d5):"d"(b));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f6):"f"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(d6):"d"(b));
    asm volatile("add.f32 %0,%0,%1;":"+f"(f7):"f"(a)); asm volatile("add.f64 %0,%0,%1;":"+d"(d7):"d"(b)); }
  o[threadIdx.x]=f0+f1+f2+f3+f4+f5+f6+f7; od[threadIdx.x]=d0+d1+d2+d3+d4+d5+d6+d7; }
// МАНТИССА КАК КОНТЕЙНЕР: четыре int в непересекающихся 13-битных полях одного double.
__global__ void k_pack(double* o, unsigned* chk){
  double acc=0.0; unsigned t=threadIdx.x;
  double a = (double)(( (t&255) ) + ((t&255)*8192.0) + ((t&255)*8192.0*8192.0) + ((t&255)*8192.0*8192.0*8192.0));
  #pragma unroll 1
  for(int i=0;i<16;i++) acc += a;                       // 16 сложений -> поля <= 255*16=4080 < 8192, переноса нет
  o[threadIdx.x]=acc;
  unsigned long long u=(unsigned long long)acc;         // распаковка полей
  chk[threadIdx.x] = (unsigned)((u & 8191ull) | (((u>>13)&8191ull)<<8) ); }
static float bench(void(*f)(), int) { return 0; }
int main(){ cudaSetDevice(0); double *od,*od2; float* of; unsigned* ch;
  cudaMalloc(&od,4096*8); cudaMalloc(&od2,4096*8); cudaMalloc(&of,4096*4); cudaMalloc(&ch,4096*4);
  int blocks=80, thr=256; cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms;
  double ops = (double)blocks*thr*N_IT*8;
  #define RUN(name, call, nops) { call; cudaDeviceSynchronize(); cudaEventRecord(s); for(int r=0;r<5;r++){call;} \
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); \
    double cyc = (double)ms/5.0*1e-3*1.380e9; \
    printf("  %-34s %8.2f IPC/SM   %7.1f тактов\n", name, (nops)/cyc*5.0/5.0/80.0, cyc/(N_IT*8.0)); }
  printf("=== FP64 КАК НЕТРОНУТЫЙ РЕСУРС (V100, 80 SM, 256 нитей x 80 блоков) ===\n");
  RUN("add.f32 темп (8 цепочек)", (k_f_thr<<<blocks,thr>>>(of,1.0f)), ops)
  RUN("add.f64 темп (8 цепочек)", (k_d_thr<<<blocks,thr>>>(od,1.0)), ops)
  RUN("add.f64 задержка (1 цепь)", (k_d_lat<<<blocks,thr>>>(od,1.0)), ops)
  printf("\n  --- ГЛАВНЫЙ ВОПРОС: смесь fp32+fp64 против чистого fp32 ---\n");
  RUN("смесь: fp32 половина", (k_mix<<<blocks,thr>>>(of,od2,1.0f,1.0)), ops)
  printf("     (в смеси ВДВОЕ больше инструкций: %0.0f fp32 + столько же fp64)\n", ops);
  k_pack<<<blocks,thr>>>(od,ch); cudaDeviceSynchronize();
  double h; unsigned hc; cudaMemcpy(&h,od+7,8,cudaMemcpyDeviceToHost); cudaMemcpy(&hc,ch+7,4,cudaMemcpyDeviceToHost);
  printf("\n  --- мантисса 52 бита как контейнер: 4 поля по 13 бит, 16 сложений ---\n");
  printf("     нить 7: значение 7, ожидаем в каждом поле 7*16=112; поле0=%llu поле1=%llu поле2=%llu поле3=%llu\n",
     ((unsigned long long)h)&8191ull, (((unsigned long long)h)>>13)&8191ull,
     (((unsigned long long)h)>>26)&8191ull, (((unsigned long long)h)>>39)&8191ull);
  printf("     ТОЧНО: %s (одно add.f64 обслужило ЧЕТЫРЕ независимых целочисленных сложения)\n",
     ((((unsigned long long)h)&8191ull)==112 && ((((unsigned long long)h)>>13)&8191ull)==112 &&
      ((((unsigned long long)h)>>26)&8191ull)==112 && ((((unsigned long long)h)>>39)&8191ull)==112) ? "ДА" : "НЕТ");
  return 0; }
