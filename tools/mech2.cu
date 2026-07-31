// Проверка возможностей, о которых мы не думали.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

// 1. Аккумулятор mma в FP16: вдвое меньше регистров на накопитель (4 регистра вместо 8).
__global__ void k_f16acc(const __half* A, const __half* B, __half* D) {
    const int lane = threadIdx.x;
    uint32_t a[2], b[2], d[4] = {0,0,0,0};
    a[0] = *reinterpret_cast<const uint32_t*>(A + lane*4);
    a[1] = *reinterpret_cast<const uint32_t*>(A + lane*4 + 2);
    b[0] = *reinterpret_cast<const uint32_t*>(B + lane*4);
    b[1] = *reinterpret_cast<const uint32_t*>(B + lane*4 + 2);
    asm("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
    *reinterpret_cast<uint32_t*>(D + lane*8) = d[0];
    *reinterpret_cast<uint32_t*>(D + lane*8 + 2) = d[1];
    *reinterpret_cast<uint32_t*>(D + lane*8 + 4) = d[2];
    *reinterpret_cast<uint32_t*>(D + lane*8 + 6) = d[3];
}

// 3. Два НЕЗАВИСИМЫХ произведения в разных квадропарах одного варпа (подваршовая специализация):
//    инструкция и так поквадропарная, поэтому операнды можно подать разные.
__global__ void k_qp_split(const __half* A0, const __half* B0, const __half* A1, const __half* B1, float* D) {
    const int lane = threadIdx.x, qp = (lane >> 2) & 3;
    const __half* A = (qp < 2) ? A0 : A1;
    const __half* B = (qp < 2) ? B0 : B1;
    uint32_t a[2], b[2]; float d[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) d[i] = 0.f;
    a[0] = *reinterpret_cast<const uint32_t*>(A + lane*4);
    a[1] = *reinterpret_cast<const uint32_t*>(A + lane*4 + 2);
    b[0] = *reinterpret_cast<const uint32_t*>(B + lane*4);
    b[1] = *reinterpret_cast<const uint32_t*>(B + lane*4 + 2);
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
        : "r"(a[0]),"r"(a[1]),"r"(b[0]),"r"(b[1]));
#pragma unroll
    for (int i = 0; i < 8; ++i) D[lane*8+i] = d[i];
}
int main(){ printf("ok\n"); return 0; }
