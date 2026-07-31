#!/usr/bin/env bash
# Каждая инструкция пробуется ОТДЕЛЬНО: собирается для sm_70 и отдельно для sm_75,
# чтобы отличить "нет в Volta" от "нет в компиляторе вовсе".
NVCC="${NVCC:-$(command -v nvcc || true)}"
[ -n "$NVCC" ] || { echo "nvcc not found; set NVCC or add it to PATH" >&2; exit 1; }
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/fa2-sm70-opcodes.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT
try() {  # $1=имя $2=строка asm $3=примечание
  cat > "$WORKDIR/t.cu" <<CU
#include <cstdint>
__global__ void k(uint32_t* o, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d = 0;
  $2
  o[0] = d;
}
CU
  r70=$($NVCC -arch=sm_70 -c "$WORKDIR/t.cu" -o /dev/null 2>&1 | grep -ciE "error")
  r75=$($NVCC -arch=sm_75 -c "$WORKDIR/t.cu" -o /dev/null 2>&1 | grep -ciE "error")
  s70=$([ "$r70" = "0" ] && echo "ЕСТЬ" || echo "нет ")
  s75=$([ "$r75" = "0" ] && echo "ЕСТЬ" || echo "нет ")
  printf "  %-22s sm70=%s sm75=%s   %s\n" "$1" "$s70" "$s75" "$3"
}
echo "=== PTX без обёртки в CUDA C: что примет ptxas для sm_70 ==="
try "lop3 произв. ЛУТ"  'asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "ЛЮБАЯ функция 3 входов за 1 инструкцию"
try "prmt.b4e"          'asm("prmt.b32 %0, %1, %2, %3, ;" : "=r"(d) : "r"(a),"r"(b),"r"(c));'  "(синтаксис-проба)"
try "prmt режим f4e"    'asm("prmt.f4e.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "сдвиг байт с переносом"
try "prmt режим b4e"    'asm("prmt.b4e.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "обратный порядок байт"
try "prmt режим ecl"    'asm("prmt.ecl.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "зажим слева"
try "prmt режим ecr"    'asm("prmt.ecr.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "зажим справа"
try "prmt режим rc8"    'asm("prmt.rc8.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "повтор байта 8 раз"
try "prmt режим rc16"   'asm("prmt.rc16.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "повтор полуслова"
try "bmsk (маска бит)"  'asm("bmsk.clamp.b32 %0, %1, %2;" : "=r"(d) : "r"(a),"r"(b));' "создать битовую маску"
try "fns (n-й бит)"     'asm("fns.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"((int)c));' "найти n-й установленный бит"
try "bfind"             'asm("bfind.u32 %0, %1;" : "=r"(d) : "r"(a));' "старший установленный бит"
try "bfe (извлечь поле)" 'asm("bfe.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "извлечение битового поля"
try "bfi (вставить поле)" 'asm("bfi.b32 %0, %1, %2, %3, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "вставка битового поля"
try "dp2a.lo"           'asm("dp2a.lo.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "2-путевое скалярное произв."
try "dp4a.u32"          'asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "4-путевое, int8"
try "vabsdiff4.sat"     'asm("vabsdiff4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "SIMD с насыщением"
try "vshl.wrap"         'asm("vshl.u32.u32.u32.wrap %0, %1, %2, %3;" : "=r"(d) : "r"(a),"r"(b),"r"(c));' "видео-сдвиг"
try "szext (расшир.знака)" 'asm("szext.clamp.s32 %0, %1, %2;" : "=r"(d) : "r"(a),"r"(b));' "расширение знака (sm_70?)"
try "activemask"        'asm("activemask.b32 %0;" : "=r"(d));' "маска активных полос"
try "redux.sync"        'asm("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(d) : "r"(a));' "варповая редукция ОДНОЙ инстр."
try "match.any.sync"    'asm("match.any.sync.b32 %0, %1, 0xffffffff;" : "=r"(d) : "r"(a));' "поиск равных полос"
try "mma m8n8k4 f16"    'asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%0,%0,%0,%0,%0,%0,%0},{%1,%1},{%2,%2},{%0,%0,%0,%0,%0,%0,%0,%0};" : "+r"(d) : "r"(a),"r"(b));' "наша тензорная"
try "mma m8n8k16 s8"    'asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%0},{%1},{%2},{%0,%0};" : "+r"(d) : "r"(a),"r"(b));' "IMMA -- ожидаем отказ"
