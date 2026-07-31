// ЗОНД РАСПРЕДЕЛЕНИЯ БЛОКОВ ПО SM. Оборудование статично, значит отображение blockIdx -> smid можно
// НЕ УГАДЫВАТЬ, а замерить. Заодно проверяем, СОХРАНЯЕТСЯ ли оно между запусками и зависит ли от
// занятости (блоков на SM).
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__global__ __launch_bounds__(256, 2) void probe(unsigned* out, long long spin) {
  if (threadIdx.x == 0) {
    unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    out[blockIdx.x] = s;
  }
  // держим блок занятым, чтобы резидентность была реальной, а не мгновенной
  long long t0 = clock64(); while (clock64() - t0 < spin) {}
}
int main(int argc, char** argv) {
  const int NB = argc > 1 ? atoi(argv[1]) : 160;
  unsigned* d; cudaMalloc(&d, NB * 4);
  std::vector<unsigned> h(NB);
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  printf("SM на устройстве: %d, блоков в сетке: %d\n\n", nsm, NB);
  for (int run = 0; run < 2; ++run) {
    probe<<<NB, 256>>>(d, 2000000);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, NB * 4, cudaMemcpyDeviceToHost);
    printf("запуск %d, blockIdx -> smid (первые 24): ", run);
    for (int i = 0; i < 24 && i < NB; ++i) printf("%u ", h[i]);
    printf("\n");
    // проверяем гипотезу smid == blockIdx %% nsm
    int match = 0; for (int i = 0; i < NB; ++i) if ((int)h[i] == i % nsm) ++match;
    printf("   гипотеза smid = blockIdx %% %d: совпало %d из %d (%.0f%%)\n", nsm, match, NB, 100.0*match/NB);
    // кто с кем делит SM
    std::vector<std::vector<int>> bysm(128);
    for (int i = 0; i < NB; ++i) bysm[h[i]].push_back(i);
    int pairs_same_parity = 0, pairs_tot = 0, diff_is_nsm = 0;
    for (auto& v : bysm) if (v.size() == 2) {
      ++pairs_tot;
      if (((v[0] ^ v[1]) & 1) == 0) ++pairs_same_parity;
      if (v[1] - v[0] == nsm) ++diff_is_nsm;
    }
    printf("   пар на одном SM: %d; из них ОДНОЙ чётности: %d; с разностью ровно %d: %d\n",
           pairs_tot, pairs_same_parity, nsm, diff_is_nsm);
  }
  return 0;
}
