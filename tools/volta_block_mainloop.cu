// Мейнлуп УРОВНЯ БЛОКА на 16 варпов, на рукописном варповом операторе
// (fa2_src/fmha_kernel/gemm/volta_warp_mma.h). Прототип: ВСЕ ПЯТЬ GEMM обратного прохода,
// без причинной маски; двойная буферизация пока только в фазе 1.
//
// Сборка:
//   nvcc -O3 -std=c++17 -arch=sm_70 -I ../fa2_src/fmha_kernel/gemm -o block volta_block_mainloop.cu
//
// ЗАЧЕМ. При d>128 штатное ядро на cutlass держит накопители dK/dV в ГЛОБАЛЬНОЙ памяти и возвращает
// их на каждый блок запросов: kOutputInRF требует kMaxK<=kBI, поднять kBI мешает smem, сузить плитку
// варпа мешает `Shape % GemmShape<32,32,4>`. Форсирование резидентности внутри cutlass измерено:
// 696 Б разлива на поток и 0.46x против torch. Здесь 512 потоков, плитки варпа 16x16, оба накопителя
// живут в регистрах всю развёртку, K и V читаются из глобальной памяти ОДИН раз на блок, а поток
// Q/dO идёт с двойной буферизацией.
//
// НИ ОДНОЙ ТРАНСПОЗЫ — за счёт всех четырёх вариантов инструкции:
//   S    = K*Q^T    .row.col     dP^T = V*dO^T   .row.col  (сразу в ориентации [ключ][запрос], как S,
//   dV  += P^T*dO   .row.row      поэтому dS = P*(dP^T-Delta) — поэлементно в регистрах)
//   dK  += dS*Q     .row.row     dQ  += dS^T*K   .col.row  (dS читается как [k=ключ][m=запрос])
//
// ИЗМЕРЕНО (V100 @ 1530 МГц, Sq=Sk=2048, ПЯТЬ GEMM; H — число голов в сетке):
//   H=1, d=256, без маски:  0.859 мс = 12.5 ТФЛОП/с   128 рег, разлив 80 Б, smem 77 КБ
//   H=8, d=256, с маской:   3.499 мс = 12.3 ТФЛОП/с
//   H=8, d=128, с маской:   1.260 мс = 17.0 ТФЛОП/с
//   relL2 против fp64: dV 7.1e-06, dK 3.5e-04, dQ 3.5e-04 (маска включена в эталон)
//
// ЧЕСТНОЕ СРАВНЕНИЕ — И ОНО ПОКА НЕ В ПОЛЬЗУ ПРОТОТИПА. Чистое время ШТАТНОГО ядра backward
// (torch-профайлер, только attention_kernel_backward_batched_impl, B*H=8, S=2048, с маской):
//   d=256: 2.732 мс = 15.7 ТФЛОП/с   -> прототип медленнее в 1.28 раза
//   d=128: 0.939 мс = 22.9 ТФЛОП/с   -> прототип медленнее в 1.34 раза
//
// ПРЕДЫДУЩИЕ РЕДАКЦИИ ЭТОГО ЗАГОЛОВКА УТВЕРЖДАЛИ ОБРАТНОЕ («на 12%/31% впереди») — ЭТО БЫЛО НЕВЕРНО.
// Базой служили ~10.1 ТФЛОП/с, посчитанные из 4.25 мс, а те 4.25 мс — это FORWARD + BACKWARD. То есть
// backward-ядро прототипа сравнивалось с чужой ПАРОЙ проходов. Урок общий: базу надо мерить тем же
// инструментом и в тех же границах, что и измеряемое, иначе выигрыш возникает из разметки, а не из кода.
//
// ПОЧЕМУ РАЗРЫВ — ИЗМЕРЕНО ncu, и ответ не тот, который я предполагал трижды подряд.
// Прототип (BK=32, d=256, H=8, маска) против штатного ядра:
//                       прототип   штатное
//   варпов активно        24.9%      12.5%     <- цель достигнута, варпов вдвое больше
//   DRAM                  13.1%      47%       <- трафик втрое МЕНЬШЕ
//   тензорный конвейер    11.6%      17.8%
//   short_scoreboard       2.37       0.06     <- в 40 раз хуже
//   mio_throttle           1.51       0.08     <- в 19 раз хуже
//   long_scoreboard        4.99       2.86
// Занятость поднята, трафик срезан — и ни то ни другое не было узким местом. Ядро упирается в
// РАЗДЕЛЁННУЮ ПАМЯТЬ. Причина в геометрии: 16 варпов на блочную плитку [32][128] дают плитку варпа
// 16x16, то есть MB=NB=1, то есть ДВЕ загрузки фрагмента на одну mma. У cutlass плитка варпа 32x32
// при 8 варпах — 4 загрузки на 4 mma, отношение 1.0, вдвое лучше.
//
// ОТСЮДА НАСТОЯЩЕЕ УСЛОВИЕ ПОБЕДЫ: нужны 16 варпов И плитка варпа не меньше 32x32 одновременно,
// то есть блочная плитка >= 16*1024 = 16384 ячеек. Но резидентные dV+dK стоят 2*BK*D/512 float на
// поток, и при BK=128 это 128 float — не влезает. Занятость, трафик и отношение загрузок к mma
// нельзя максимизировать втроём при одном бюджете регистров; cutlass выбрал третье, пожертвовав
// первым, и на этих формах оказался прав.
//
// ЧТО ПРОВЕРЕНО И НЕ ПОМОГЛО (все замеры при d=256, H=8, с маской; штатное ядро 2.732 мс):
//   BK=32, K/V резидентны (лучшее)          3.498 мс  12.3 ТФЛОП/с
//   Q и dO резидентны вместо K/V (k_qres)   3.739     11.5   трафик в 1.6 раза меньше — и медленнее
//   BK=64, K/V потоком, варпы 2x8           4.381      9.8
//   BK=64, K/V потоком, варпы 4x4           4.642      9.3
// NB последние две конфигурации добавляют по два __syncthreads() на кусок и тем ломают двойную
// буферизацию — это НЕ чистая проверка гипотезы «крупнее плитка», а смешанный опыт, и вывода о
// плитке из них делать нельзя. Чистая проверка требует потокового K/V без лишних барьеров.
//
// ПУТЬ 6.9 -> 13.2, каждый шаг измерен отдельно:
//    6.9  пять GEMM, dQ сбрасывается разрозненной атомарной записью
//    9.0  сброс dQ через smem одним коалесцированным проходом (17%) + кусок по d 32->64 (~8%)
//   11.3  K и V в полной резидентности: уходят из всех внутренних циклов вместе с барьерами
//   13.2  двойная буферизация Q/dO в фазе 1: загрузка куска dc+DC1 выдаётся ДО счёта куска dc
// Тупики, оставленные в файле: DC3=128 хуже DC3=64 (плитка варпа 16x64 стоит 32 float на поток);
// BK=64 при резидентных K/V не влезает в 96 КБ (проверка размера стоит в run_case).
//
// ТРИ ОШИБКИ, ПОЙМАННЫЕ ПО ДОРОГЕ — они полезнее цифр:
//   * «dQ стоит из-за разрозненных атомарных операций»: абляция самой atomicAdd даёт лишь 2%.
//     Сравнивать надо разрозненную атомарность с КОАЛЕСЦИРОВАННОЙ, а не с её отсутствием;
//   * затем весь выигрыш был приписан размеру куска — в одной правке менялись оба. Разведение по
//     одному изменению за раз: коалесценция 17%, кусок ~8%;
//   * fp32-буфер сброса dQ переполнял арену (её размер считался только по LD1/LD2). Пока соседями
//     были короткоживущие буферы, переполнение затирало то, что и так перезаписывалось, и молчало.
//     Как только K стала резидентной — сразу NaN. Скрытая ошибка проявляется, когда рядом
//     появляется долгоживущий сосед.
// И одно ограничение, всплывшее дважды: дополнение строк на 4 половинки даёт бесконфликтность, но
// ЗАПРЕЩАЕТ 16-байтовые доступы к smem (шаг 72 Б). Из глобальной читаем по 16 Б, в smem кладём двумя
// 8-байтовыми — бесконфликтность в этом цикле важнее ширины записи.
// ---------------------------------------------------------------------------------------------
// РАЗДЕЛЬНЫЙ ПРОХОД dV (k_passV) — гипотеза о плитке ПОДТВЕРЖДЕНА замером.
// Блочная плитка [128][128] = ровно 16 варпов по 32x32, резидентен ТОЛЬКО dV (64 float на поток
// при d=256). Отношение загрузок фрагмента к mma падает с 2.0 (плитка 16x16) до 1.0 — как у cutlass.
//   слитное ядро, все пять gemm, d=256, H=8, маска:   12.3 ТФЛОП/с
//   проход dV (S + dV), та же форма:                  16.8 ТФЛОП/с   (+37%)
//   то же при d=128:                                  18.5
//   relL2 dV 7.3e-06
// Для сравнения: штатное ядро на cutlass идёт 15.7 ТФЛОП/с, но за ВСЕ ПЯТЬ gemm.
//
// МОЖНО ЛИ УКРУПНИТЬ ПЛИТКУ ДАЛЬШЕ? Измерено, ответ — нет, не даром.
//   при ФИКСИРОВАННОЙ площади ориентация решает мало:
//       32x32 (отношение 1.00)  16.8 ТФЛОП/с
//       64x16 (1.25)            17.0        <- чуть лучше квадрата, хотя отношение хуже
//       16x64 (1.25)            15.6
//     то есть отношение предсказывает порядок лишь отчасти: операнды A и B несимметричны;
//   увеличение ПЛОЩАДИ отношение действительно улучшает (32x64 -> 0.75, 64x64 -> 0.5), но
//     накопитель растёт как MB*NB*8 float на поток: 32 -> 64 -> 128. Проход dV уже держит
//     dV(64) + S(32) = 96 float при 128 регистрах и разливе 80-128 Б. 64x64 потребовало бы 192.
//   Вывод: 32x32 — крупнейшая плитка, совместимая с резидентным dV. Ровно то отношение, что у
//   cutlass, но при вдвое большей занятости — и это дало +37%.
//
// ДВА ПРОХОДА ИЗМЕРЕНЫ — И НЕ ОКУПИЛИСЬ. Проход 1 (S+dV) + проход 2 (dP^T,dK,dQ, читает P из
// глобальной памяти), пять gemm, H=8, маска:
//     d=256:  3.479 мс = 12.4 ТФЛОП/с   (слитное ядро 3.498;  штатное 2.732)
//     d=128:  1.766 мс = 12.2           (слитное 1.260;       штатное 0.939)
//   dV 7.6e-06, dK 1.8e-05, dQ 1.8e-05 — корректность полная.
// Проход dV сам по себе идёт 16.8 ТФЛОП/с, но по разности проход 2 даёт лишь ~11.5: выигрыш от
// плитки 32x32 съеден трафиком P, лишними циклами по кускам (dK по DC2, dQ по DC3 — свои барьеры)
// и атомарностью dQ. При d=128 хуже вдвойне: P-трафик тот же, а работы вдвое меньше.
// ВЫВОД: крупная плитка варпа — необходимое условие, но не достаточное. Она окупается только если
// не приходится платить за неё лишним проходом по памяти.
//
// ЭТОТ ЖЕ ЗАКОН ПОДТВЕРЖДЁН ВТОРОЙ РАЗ, другим способом (SEQ). Чтобы плитка стала 32x32 во ВСЕХ
// пяти gemm слитного ядра, нужно BK=BQ=DC2=128; тогда dV+dK = 2*BK*D/512 влезают только при d=128,
// и то лишь если S и dP^T не живут в регистрах одновременно. Разведение их по времени (P к моменту
// dP^T уже в smem, поэтому регистры S свободны) это позволяет — но требует ВТОРОГО потокового
// прохода по d, со своими барьерами и повторной загрузкой V:
//     d=128: SEQ 2.164 мс против 1.260 у лучшей конфигурации (BK=32/BQ=128)
//     d=256: SEQ 5.294 мс против 3.498
// Оба раза цена лишнего прохода по памяти больше выигрыша от плитки. Правило: отношение
// загрузок-к-mma улучшают ТОЛЬКО перекомпоновкой в пределах одного прохода.
//
// ИСХОДНАЯ ПРОЕКЦИЯ БЫЛА НЕВЕРНА: я оценил запись P в 0.16 мс, измерение дало ~0.24 мс только на
// запись (проход dV: 17.0 -> 13.8 ТФЛОП/с), плюс столько же на чтение. Пересчёт S стоил бы примерно
// столько же — то есть выбора «дешевле» там вообще не было.
//
// ПРЕЖНЯЯ ПРОЕКЦИЯ (не измерено, помечено как оценка, оказалась неверной): проход 1 = S + dV (2 единицы),
// проход 2 = dP^T + dK + dQ (3 единицы), P пишется в глобальную память вместо пересчёта S.
// Трафик P: 2 x Sq*Sk*2 Б на голову = 16 МБ, при 8 головах ~0.16 мс — ВТРОЕ дешевле, чем
// пересчитать S (0.51 мс). Итого 5 единиц при ~17 ТФЛОП/с = 2.52 мс + 0.16 = ~2.68 мс против
// 2.732 мс у штатного. То есть паритет-плюс, и только постройка прохода 2 это подтвердит.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include "volta_warp_mma.h"

using namespace fa2_sm70;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

// BK ключей на блок, BQ запросов за проход, D — голова, DC1/DC2 — куски по d.
__device__ int g_ticket;
static int g_warps = 16;   // для печати занятости      // талон очереди работ (см. PERSIST ниже)

template <int BK, int BQ, int D, int DC1, int DC2, int WM, int WN, bool CAUSAL, bool KVRES = true, bool SEQ = false, int DC3P = 64, bool SDMA = false, bool PERSIST = false, int QSPLIT = 1, int PAD = 4, bool SWZ = false>
__global__ __launch_bounds__(WM * WN * 32) void k_dv(const __half* __restrict__ Q,
                                            const __half* __restrict__ Kk,
                                            const __half* __restrict__ Vv,
                                            const __half* __restrict__ dO,
                                            const float* __restrict__ lse,
                                            const float* __restrict__ Delta,
                                            float* __restrict__ dV,
                                            float* __restrict__ dK,
                                            float* __restrict__ dQ, int Sq, int Sk, int nJobs = 0) {
  constexpr int kWarps = WM * WN;   // [ГЕОМЕТРИЯ] раньше было прибито 16; теперь плитка варпа
                                    // TM x TN = BK/WM x BQ/WN перебирается, и с ней загрузок-на-mma
  constexpr int LD1 = DC1 + PAD, LDP = SWZ ? BQ : (BQ + PAD), LD2 = DC2 + PAD;
  constexpr int DC3 = DC3P;                          // кусок по d для dQ
  constexpr int kChunks = D / DC2;
  constexpr int TM = BK / WM, TN = BQ / WN;          // плитка варпа для S/dP: TM x TN
  constexpr int GM = BK / WM, GN = DC2 / WN;         // плитка варпа для dV/dK
  // [ГЕОМЕТРИЯ] сетка варпов фазы dQ ВЫВОДИТСЯ из числа варпов, а не зашита 8x2: при kWarps != 16
  // прежняя запись оставляла половину плитки dQ несчитанной (сверка на S=512 ловит это как ОТКАЗ).
  constexpr int WNq = 2, WMq = kWarps / WNq;
  constexpr int QM = BQ / WMq, QN = DC3 / WNq;      // плитка варпа для dQ
  static_assert(TM % 16 == 0 && TN % 16 == 0, "плитка варпа для S должна быть кратна 16");
  static_assert(GM % 16 == 0 && GN % 16 == 0, "плитка варпа для dV/dK должна быть кратна 16: увеличьте DC2");
  static_assert(QM % 16 == 0 && QN % 16 == 0, "плитка варпа для dQ должна быть кратна 16");
  // Разделяемая память: sP живёт между фазами, поэтому не пересекается; sQ и sdO — пересекаются.
  extern __shared__ __half smem[];
  // KVRES=true: K и V целиком (минимум их трафика, но BK ограничен smem).
  // KVRES=false: K/V текут кусками -> BK можно удвоить, а значит и плитку варпа. Профиль показал,
  // что узкое место — разделяемая память (short_scoreboard 2.37 при плитке 16x16 = ДВЕ загрузки
  // фрагмента на одну mma), а вовсе не глобальный трафик (DRAM 13%). Плитка важнее трафика.
  // В потоковом варианте один и тот же буфер принимает куски ДВУХ разных ширин: DC1 в фазе 1 и
  // DC3 в фазе dQ. Шаг строки берётся по максимуму — иначе запись куска DC3 уезжает за строку.
  constexpr int LDK = KVRES ? (D + PAD) : ((DC1 > DC3 ? DC1 : DC3) + PAD);
  // Арена служит трём разным нуждам, и размер берётся по МАКСИМУМУ. Пропущенный здесь третий
  // случай (fp32-буфер сброса dQ) раньше переполнял арену молча: он затирал буферы, которые всё
  // равно перезаписывались следующей итерацией. Как только K стала резидентной, та же ошибка стала
  // затирать K и дала NaN. Скрытая ошибка проявляется, когда рядом появляется долгоживущий сосед.
  constexpr int kArenaA = 4 * BQ * LD1 > BQ * LD2 ? 4 * BQ * LD1 : BQ * LD2;   // 4 = 2 буфера x (Q,dO)
  constexpr int kArena  = kArenaA > 2 * BQ * DC3 ? kArenaA : 2 * BQ * DC3;
  __half* sP    = smem;                        // P / dS, живёт между фазами
  __half* arena = sP + BK * LDP;               // Q (фаза 1) -> dO/Q (фазы 2/3)
  // [опкоды] lse и Delta читались ИЗ ГЛОБАЛЬНОЙ памяти на каждый элемент аккумулятора внутри
  // визитора: аудит SASS показал 46 из 50 LDG шириной 32 бита, и 16% из них на расстоянии 0-4
  // инструкций от потребителя, то есть с полностью открытой задержкой. Кладём их в smem один раз
  // на блок запросов (2*BQ float = 1 КБ) — вместо десятков разрозненных глобальных чтений.
  __half* sK    = arena + kArena;              // K целиком [BK][LDK]
  __half* sV    = sK + BK * LDK;               // V целиком [BK][LDK]
  // Фаза 1 живёт в арене двойным буфером; фазы 2-4 берут ту же арену целиком (во времени не пересекаются).
  __half* qbuf[2] = {arena, arena + 2 * BQ * LD1};
  __half* obuf[2] = {arena + BQ * LD1, arena + 3 * BQ * LD1};

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;         // сетка варпов WM x WN
  // [PERSIST] Постоянные блоки: сетка кратна ЧИСЛУ SM (80), а работа берётся динамически по талону.
  // Статическая раскладка «блок = плитка ключей x голова» даёт 64 блока при H=1 -> 16 SM из 80
  // простаивают весь запуск, а при причинной маске ещё и перекошена: время задаёт самый тяжёлый блок.
  // Талон на atomicAdd безопасен по построению — никто никого не ждёт (см. SM70_LATENCY_HIDING §10.1).
  __shared__ int s_job;
  int job = 0;
  if constexpr (PERSIST) {
    if (tid == 0) s_job = atomicAdd(&g_ticket, 1);
    __syncthreads();
    job = s_job;
  }
  const __half *Q0 = Q, *Kk0 = Kk, *Vv0 = Vv, *dO0 = dO;   // базы: каждое задание смещает от них
  const float *lse0 = lse, *Delta0 = Delta;
  float *dV0 = dV, *dK0 = dK, *dQ0 = dQ;
  while (true) {
  if constexpr (PERSIST) { if (job >= nJobs) break; }
  // [QSPLIT] Причинность делает задания РАЗНЫМИ по весу: плитка ключей i обрабатывает Sq/BQ - i*BK/BQ
  // блоков запросов. При Sk=2048, BK=32, BQ=128 это 16..1, всего 544 единицы, а время задаёт самая
  // тяжёлая = 16. Значит машина занята на 544/(16*80) = 42.5%, и никакая раздача 64 крупных заданий
  // этого не исправит. Дробим по ЗАПРОСАМ: срез s берёт блоки запросов с шагом QSPLIT, вес задания
  // падает до ~16/QSPLIT, и жадная раздача по талону выходит на предел 544/80 = 6.8.
  // Цена — dK/dV теперь копятся атомарно (срезы одной плитки ключей складываются).
  const int slice = PERSIST ? (job % QSPLIT) : 0;
  const int kb = (PERSIST ? ((job / QSPLIT) % (Sk / BK)) : blockIdx.x) * BK;   // плитка ключей
  // Голова = blockIdx.y. При причинности вся экономия — в ЧИСЛЕ волн: 64 блока на 80 SM идут
  // одновременно, и время задаёт самый тяжёлый (kb=0). Реальный B*H даёт много волн, и пропуск
  // блоков запросов превращается в реальную экономию.
  {
    const long hoff = PERSIST ? (long)(job / (QSPLIT * (Sk / BK))) : (long)blockIdx.y;
    Q += hoff * (long)Sq * D;  Kk += hoff * (long)Sk * D;  Vv += hoff * (long)Sk * D;
    dO += hoff * (long)Sq * D; lse += hoff * (long)Sq;     Delta += hoff * (long)Sq;
    dV += hoff * (long)Sk * D; dK += hoff * (long)Sk * D;  dQ += hoff * (long)Sq * D;
  }

  // K и V читаются из глобальной памяти ОДИН раз на блок, а не на каждый кусок по d и не на каждый
  // блок запросов: для Sq/BQ проходов это во столько же раз меньше их трафика, и главное — они
  // уходят из всех внутренних циклов вместе со своими барьерами.
  if constexpr (KVRES)
    for (int i = tid; i < BK * D / 8; i += kWarps * 32) {      // 16 Б за доступ вместо 2
      const int r = i / (D / 8), c = (i % (D / 8)) * 8;
      const uint4 gk_ = *reinterpret_cast<const uint4*>(&Kk[(long)(kb + r) * D + c]);
      const uint4 gv_ = *reinterpret_cast<const uint4*>(&Vv[(long)(kb + r) * D + c]);
      __half* pk = &sK[r * LDK + c]; __half* pv = &sV[r * LDK + c];
      if constexpr (PAD % 8 == 0) { *reinterpret_cast<uint4*>(pk) = gk_; *reinterpret_cast<uint4*>(pv) = gv_; }
      else { *reinterpret_cast<uint2*>(pk)     = make_uint2(gk_.x, gk_.y);
             *reinterpret_cast<uint2*>(pk + 4) = make_uint2(gk_.z, gk_.w);
             *reinterpret_cast<uint2*>(pv)     = make_uint2(gv_.x, gv_.y);
             *reinterpret_cast<uint2*>(pv + 4) = make_uint2(gv_.z, gv_.w); }
    }

  // dV И dK в РЕГИСТРАХ на всю развёртку по блокам запросов — это и есть цель.
  WarpMma<GM / 16, GN / 16, BLayout::kNMajor> gv[kChunks], gk[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) { gv[c].clear(); gk[c].clear(); }

  // Причинность: ключ k влияет на запрос q только при k <= q. Блоки запросов ЦЕЛИКОМ левее
  // диагонали пропускаются — это и есть та половина работы, которую снимает маска; поэлементная
  // маска нужна лишь на диагональном блоке.
  const int qStart = (CAUSAL ? (kb / BQ) * BQ : 0) + slice * BQ;
  for (int q0 = qStart; q0 < Sq; q0 += BQ * QSPLIT) {

  // [АДРЕСАЦИЯ, шаг 2] Перезаливка глобальная->smem шла ПО ОДНОМУ half: BQ*DC2 = 16384 элемента на
  // 512 потоков = 32 прохода, и в каждом полный пересчёт адреса (два IMAD + деление + остаток).
  // Замер состава инструкций дал integer 2.54x и memory 3.96x против cutlass — вот их источник.
  // Читаем из глобальной по 16 Б (строка D=256 half выровнена, смещения кратны 8), кладём в smem
  // двумя по 8 Б: шаг строки LD2 = DC2+4 half не кратен 16 Б, а бесконфликтность важнее ширины.
  auto fill_arena = [&](const __half* __restrict__ src, int chunk) {
    constexpr int kRowVec = DC2 / 8;                   // векторов в строке куска
    constexpr int kIters  = BQ * kRowVec / (WM * WN * 32);
    static_assert(BQ * kRowVec % (WM * WN * 32) == 0, "перезаливка арены не делится на блок");
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int v = tid + it * (WM * WN * 32), r = v / kRowVec, cc = (v % kRowVec) * 8;
      const uint4 g = *reinterpret_cast<const uint4*>(&src[(long)(q0 + r) * D + chunk * DC2 + cc]);
      __half* d = &arena[r * LD2 + cc];
      if constexpr (PAD % 8 == 0) *reinterpret_cast<uint4*>(d) = g;   // шаг кратен 16 Б
      else { *reinterpret_cast<uint2*>(d)     = make_uint2(g.x, g.y);
             *reinterpret_cast<uint2*>(d + 4) = make_uint2(g.z, g.w); }
    }
  };
    // ---- фаза 1: S = K * Q^T, редукция по d кусками DC1
    // SEQ: S и dP^T НЕ живут одновременно. P к моменту dP^T уже лежит в smem, поэтому регистры
    // S можно освободить и посчитать в них dP^T. Пик падает на 32 float — ровно столько нужно,
    // чтобы при BK=BQ=DC2=128 влезли резидентные dV+dK и плитка стала 32x32 во ВСЕХ gemm.
    WarpMma<TM / 16, TN / 16, BLayout::kKMajor> qk, dpt;
    qk.clear();
    if constexpr (!SEQ) dpt.clear();
    // Двойная буферизация глобального этапа: загрузка куска dc+DC1 ВЫДАЁТСЯ до счёта куска dc, и её
    // задержку покрывают mma текущего куска. Без этого цикл строго последователен:
    // барьер -> загрузка -> барьер -> счёт, и задержка глобальной памяти ничем не закрыта.
    constexpr int kPerThr = BQ * DC1 / (kWarps * 32);
    static_assert(kPerThr % 8 == 0, "нужен целый 16-байтовый доступ на поток");
    constexpr int kVec = kPerThr / 8, kChkRow = DC1 / 8;
    uint4 stq[kVec], sto[kVec];
    // [softDMA] Выдача ОДНОЙ порции: слот idx. Порции Q идут первыми, затем dO.
    auto issue_one = [&](int dcx, int idx) {
      if (idx < kVec) {
        const int v = idx, i2 = tid + v * kWarps * 32, r = i2 / kChkRow, c = (i2 % kChkRow) * 8;
        stq[v] = *reinterpret_cast<const uint4*>(&Q[(long)(q0 + r) * D + dcx + c]);
      } else {
        const int v = idx - kVec, i2 = tid + v * kWarps * 32, r = i2 / kChkRow, c = (i2 % kChkRow) * 8;
        sto[v] = *reinterpret_cast<const uint4*>(&dO[(long)(q0 + r) * D + dcx + c]);
      }
    };
    auto ldg_chunk = [&](int dcx) {
#pragma unroll
      for (int v = 0; v < kVec; ++v) {
        const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
        stq[v] = *reinterpret_cast<const uint4*>(&Q[(long)(q0 + r) * D + dcx + c]);
        sto[v] = *reinterpret_cast<const uint4*>(&dO[(long)(q0 + r) * D + dcx + c]);
      }
    };
    auto ldg_chunk_o = [&](int dcx) {
#pragma unroll
      for (int v = 0; v < kVec; ++v) {
        const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
        sto[v] = *reinterpret_cast<const uint4*>(&dO[(long)(q0 + r) * D + dcx + c]);
      }
    };
    auto sts_chunk_o = [&](int b) {
#pragma unroll
      for (int v = 0; v < kVec; ++v) {
        const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
        __half* po = &obuf[b][r * LD1 + c];
        if constexpr (PAD % 8 == 0) *reinterpret_cast<uint4*>(po) = sto[v];
        else { *reinterpret_cast<uint2*>(po)     = make_uint2(sto[v].x, sto[v].y);
               *reinterpret_cast<uint2*>(po + 4) = make_uint2(sto[v].z, sto[v].w); }
      }
    };
    auto sts_chunk = [&](int b) {
#pragma unroll
      for (int v = 0; v < kVec; ++v) {
        const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
        // Шаг строки LD1 = DC1+4 половинок = 72 Б: строки ложатся на РАЗНЫЕ банки (бесконфликтно),
        // но 16-байтовая запись для нечётных строк невыровнена. Из глобальной читаем по 16 Б,
        // в smem кладём двумя 8-байтовыми. Дополнение и широкая запись несовместимы — выбираем
        // бесконфликтность, она в этом цикле важнее.
        __half* pq = &qbuf[b][r * LD1 + c];
        __half* po = &obuf[b][r * LD1 + c];
        if constexpr (PAD % 8 == 0) { *reinterpret_cast<uint4*>(pq) = stq[v]; *reinterpret_cast<uint4*>(po) = sto[v]; }
        else { *reinterpret_cast<uint2*>(pq)     = make_uint2(stq[v].x, stq[v].y);
               *reinterpret_cast<uint2*>(pq + 4) = make_uint2(stq[v].z, stq[v].w);
               *reinterpret_cast<uint2*>(po)     = make_uint2(sto[v].x, sto[v].y);
               *reinterpret_cast<uint2*>(po + 4) = make_uint2(sto[v].z, sto[v].w); }
      }
    };
    __syncthreads();
    ldg_chunk(0); sts_chunk(0);
    __syncthreads();
    int cur = 0;
    for (int dc = 0; dc < D; dc += DC1) {
      if (dc + DC1 < D) {
        ldg_chunk(dc + DC1);                          // выдаётся заранее
        // БАРЬЕР ПЛАНИРОВЩИКА. Аудит SASS: ptxas УТОПИЛ эти LDG к самому потребителю — медиана
        // расстояния «загрузка -> первое использование» 5-16 инструкций против 58 у cutlass, и 16%
        // загрузок стоят на расстоянии 0-4, то есть их задержка не спрятана ничем. Планировщик
        // минимизирует время жизни регистров и потому сдвигает загрузку вниз. Пустой asm с
        // клоббером памяти запрещает переставлять обращения к памяти через эту точку: LDG остаётся
        // сверху, а весь блок mma+LDS оказывается между ним и потребляющей STS.
        // asm volatile("" ::: "memory");   // A/B: барьер планировщика
      }
      if constexpr (!KVRES) {
        __syncthreads();
        for (int i = tid; i < BK * DC1; i += kWarps * 32) {
          const int r = i / DC1, c = i % DC1;
          sK[r * LDK + c] = Kk[(long)(kb + r) * D + dc + c];
          sV[r * LDK + c] = Vv[(long)(kb + r) * D + dc + c];
        }
        __syncthreads();
      }
      const int koff = KVRES ? dc : 0;
      qk.template accumulate<DC1>(sK + koff, LDK, wm * TM, qbuf[cur], LD1, wn * TN, lane);
      if constexpr (!SEQ)
        dpt.template accumulate<DC1>(sV + koff, LDK, wm * TM, obuf[cur], LD1, wn * TN, lane);
      if (dc + DC1 < D) {
        __syncthreads();
        sts_chunk(cur ^ 1);
        __syncthreads();
        cur ^= 1;
      }
    }
    // ---- P = exp(S*scale - lse[запрос]) и выгрузка в fp16 для следующего gemm
    const float scale = rsqrtf((float)D);
    const int r0 = wm * TM, c0 = wn * TN;
    // [опкоды] lse и Delta читались из ГЛОБАЛЬНОЙ памяти на КАЖДЫЙ элемент аккумулятора: аудит SASS
    // насчитал 46 из 50 LDG шириной 32 бита, и 16% из них на расстоянии 0-4 от потребителя.
    // Поток трогает всего ЧЕТЫРЕ различных столбца на 16-широкий блок, поэтому поднимаем их в
    // регистры один раз. Номер слота выводится из столбца: colbase(lane) = (l&2)|((l&4)<<1) имеет
    // нулевые биты 0 и 2, значит c&1 и (c>>2)&1 — это в точности биты номера регистра.
    // NB перенос этих же чтений в SMEM был измерен и оказался ХУЖЕ на 5.5%: ядро упирается в
    // разделяемую память, и переносить туда работу нельзя. Регистры — единственный верный адрес.
    // [ГЕОМЕТРИЯ] Кэш был рассчитан ровно на ЧЕТЫРЕ столбца, то есть на ОДИН столбцовый блок
    // (TN=16). При TN=32 и шире visit выдаёт c = j*16 + wmma_acc_col(lane,r), и номер блока j=c>>4
    // в индексе отсутствовал -> второй блок брал чужие lse/Delta, портилась P, а с ней ВСЕ ТРИ
    // выхода одинаково (почерк: dV/dK/dQ = 3e-01 при 8 варпах). Слотов теперь 4 на каждый блок.
    constexpr int kNB = TN / 16;                       // столбцовых блоков в плитке варпа
    const int cb = (lane & 2) | ((lane & 4) << 1);
    float lseq[4 * kNB], delq[4 * kNB];
#pragma unroll
    for (int b = 0; b < kNB; ++b)
#pragma unroll
      for (int sI = 0; sI < 4; ++sI) {
        const int cc = c0 + b * 16 + cb + (sI & 1) + 4 * (sI >> 1);
        lseq[b * 4 + sI] = lse[q0 + cc];
        delq[b * 4 + sI] = Delta[q0 + cc];
      }
#define FA2_SLOT(c) (((c) >> 4) * 4 + ((c) & 1) + 2 * (((c) >> 2) & 1))
    qk.visit(lane, [&](float& v, int r, int c) {
      const float p = __expf(v * scale - lseq[FA2_SLOT(c)]);
      v = (CAUSAL && (kb + r0 + r) > (q0 + c0 + c)) ? 0.f : p;
    });
    if constexpr (SWZ) qk.store_half_swz(sP, LDP, r0, c0, lane);
    else qk.store_half(sP, LDP, r0, c0, lane);

    // ---- фаза 2: dV += P^T * dO, редукция по запросам (BQ), кусками по d
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      __syncthreads();
      fill_arena(dO, c);
      __syncthreads();
      if constexpr (SWZ) gv[c].template accumulate_swzA<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
      else gv[c].template accumulate<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
    }

    // ---- SEQ: только теперь считаем dP^T, в регистрах, освободившихся от S
    if constexpr (SEQ) {
      dpt.clear();
      __syncthreads();
      ldg_chunk(0); sts_chunk(0);
      __syncthreads();
      int c2 = 0;
      for (int dc = 0; dc < D; dc += DC1) {
        if (dc + DC1 < D) ldg_chunk_o(dc + DC1);
        // При потоковом K/V кусок V нужен и здесь: фаза 1 оставила в sV только ПОСЛЕДНИЙ кусок.
        if constexpr (!KVRES) {
          __syncthreads();
          for (int i = tid; i < BK * DC1; i += kWarps * 32) {
            const int r = i / DC1, c = i % DC1;
            sV[r * LDK + c] = Vv[(long)(kb + r) * D + dc + c];
          }
          __syncthreads();
        }
        dpt.template accumulate<DC1>(sV + (KVRES ? dc : 0), LDK, wm * TM, obuf[c2], LD1, wn * TN, lane);
        if (dc + DC1 < D) { __syncthreads(); sts_chunk_o(c2 ^ 1); __syncthreads(); c2 ^= 1; }
      }
    }
    // ---- dS = P * (dP^T - Delta[запрос])
    __syncthreads();
    if constexpr (SEQ) {
      // P взята из smem: в регистрах её больше нет
      dpt.visit(lane, [&](float& v, int r, int c) {
        const int rr_ = r0 + r;
        const int cc_ = SWZ ? decltype(dpt)::swz(c0 + c, rr_ & 15) : (c0 + c);
        v = __half2float(sP[(long)rr_ * LDP + cc_]) * (v - delq[FA2_SLOT(c)]);
      });
      __syncthreads();
      dpt.store_half(sP, LDP, r0, c0, lane);
    } else {
      const float* dp = &dpt.acc[0][0][0];
      int idx = 0;
      qk.visit(lane, [&](float& p, int r, int c) { p = p * (dp[idx++] - delq[FA2_SLOT(c)]); });
      // dS ложится в sP теми же правилами
#undef FA2_SLOT
      if constexpr (SWZ) qk.store_half_swz(sP, LDP, r0, c0, lane);
      else qk.store_half(sP, LDP, r0, c0, lane);
    }
    // ---- dK += dS * Q  (.row.row: Q[запрос][d] читается как есть)
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      __syncthreads();
      fill_arena(Q, c);
      __syncthreads();
      if constexpr (SWZ) gk[c].template accumulate_swzA<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
      else gk[c].template accumulate<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
    }

    // ---- dQ[запрос][d] += dS^T * K  (.col.row): dS читается КАК ЕСТЬ как [k=ключ][m=запрос],
    // K — как [k=ключ][n=d]. Ни одной транспозы. Накопитель временный (8 float на поток) и
    // сбрасывается атомарно: dQ принадлежит блоку ЗАПРОСОВ, а блок владеет ключами.
    {
      const int wmq = warp / WNq, wnq = warp % WNq;
#pragma unroll
      for (int c = 0; c < D; c += DC3) {
        WarpMma<QM / 16, QN / 16, BLayout::kNMajor, ALayout::kMMajor> gq;
        gq.clear();
        if constexpr (!KVRES) {
          __syncthreads();
          for (int i = tid; i < BK * DC3; i += kWarps * 32)
            sK[(i / DC3) * LDK + i % DC3] = Kk[(long)(kb + i / DC3) * D + c + i % DC3];
          __syncthreads();
        }
        if constexpr (SWZ) gq.template accumulate_swzA_mmajor<BK>(sP, LDP, wmq * QM, sK + (KVRES ? c : 0), LDK, wnq * QN, lane);
        else gq.template accumulate<BK>(sP, LDP, wmq * QM, sK + (KVRES ? c : 0), LDK, wnq * QN, lane);
        // Раскладка аккумулятора разрозненна по (строка, столбец), поэтому прямой atomicAdd на
        // элемент бьёт по многим гранулам. Складываем плитку в smem (там разрозненность дёшева) и
        // сбрасываем ОДНИМ коалесцированным проходом.
        // ИЗМЕРЕНО при DC3=64: с промежуточным хранением 1.190 мс, без него 1.436 -> коалесценция
        // стоит 17%. NB абляция «убрать саму atomicAdd» показывает лишь 2% — и это НЕ значит, что
        // атомарность бесплатна: при уже коалесцированном доступе она действительно дёшева.
        // Сравнивать надо разрозненную атомарность с коалесцированной, а не с её отсутствием.
        float* stage = reinterpret_cast<float*>(arena);
        __syncthreads();
        gq.visit(lane, [&](float v, int r, int cc) {
          stage[(wmq * QM + r) * DC3 + wnq * QN + cc] = v;
        });
        __syncthreads();
        for (int i = tid; i < BQ * DC3; i += kWarps * 32)
          atomicAdd(&dQ[(long)(q0 + i / DC3) * D + c + i % DC3], stage[i]);
      }
    }
  }

  // ---- эпилог: один проход записи вместо круговых рейсов на каждый блок запросов
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    gv[c].visit(lane, [&](float v, int r, int cc) {
      const long a = (long)(kb + wm * GM + r) * D + c * DC2 + wn * GN + cc;
      if constexpr (QSPLIT > 1) atomicAdd(&dV[a], v); else dV[a] = v;
    });
    gk[c].visit(lane, [&](float v, int r, int cc) {
      const long a = (long)(kb + wm * GM + r) * D + c * DC2 + wn * GN + cc;
      if constexpr (QSPLIT > 1) atomicAdd(&dK[a], v); else dK[a] = v;
    });
  }
  if constexpr (!PERSIST) break;
  // следующее задание по талону; указатели — от баз, накопители обнулить
  Q = Q0; Kk = Kk0; Vv = Vv0; dO = dO0; lse = lse0; Delta = Delta0; dV = dV0; dK = dK0; dQ = dQ0;
  __syncthreads();                       // smem переиспользуется — дождаться конца текущего задания
  if (tid == 0) s_job = atomicAdd(&g_ticket, 1);
  __syncthreads();
  job = s_job;
  }
}


template <int BK, int BQ, int D, int DC1, int DC2, int WM, int WN, bool CAUSAL = false, bool KVRES = true, bool SEQ = false, int DC3P = 64, bool SDMA = false, bool PERSIST = false, int QSPLIT = 1, int PAD = 4, bool SWZ = false>
bool run_case(int Sq, int Sk, bool bench, int H = 1) {
  std::vector<__half> hQ((size_t)Sq * D), hK((size_t)Sk * D), hV((size_t)Sk * D), hdO((size_t)Sq * D);
  std::vector<float> hlse(Sq), hDelta(Sq);
  for (auto& x : hQ) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hK) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hV) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hdO) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hlse) x = 2.0f + (rand() / (float)RAND_MAX);
  for (auto& x : hDelta) x = (rand() / (float)RAND_MAX) * 0.3f;
  __half *dQ, *dKk, *dVv, *ddO; float *dlse, *dDel, *ddV, *ddK;
  CK(cudaMalloc(&dQ, hQ.size() * 2 * H));  CK(cudaMalloc(&dKk, hK.size() * 2 * H));
  CK(cudaMalloc(&dVv, hV.size() * 2 * H)); CK(cudaMalloc(&ddO, hdO.size() * 2 * H));
  CK(cudaMalloc(&dlse, (size_t)Sq * 4 * H)); CK(cudaMalloc(&dDel, (size_t)Sq * 4 * H));
  CK(cudaMalloc(&ddV, (size_t)Sk * D * 4 * H)); CK(cudaMalloc(&ddK, (size_t)Sk * D * 4 * H));
  float* ddQ; CK(cudaMalloc(&ddQ, (size_t)Sq * D * 4 * H));
  for (int h = 0; h < H; ++h) {
    CK(cudaMemcpy(dQ + (size_t)h * Sq * D, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dKk + (size_t)h * Sk * D, hK.data(), hK.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVv + (size_t)h * Sk * D, hV.data(), hV.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddO + (size_t)h * Sq * D, hdO.data(), hdO.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlse + (size_t)h * Sq, hlse.data(), Sq * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dDel + (size_t)h * Sq, hDelta.data(), Sq * 4, cudaMemcpyHostToDevice));
  }
  CK(cudaMemset(ddV, 0x7f, (size_t)Sk * D * 4)); CK(cudaMemset(ddK, 0x7f, (size_t)Sk * D * 4));
  auto kern = k_dv<BK, BQ, D, DC1, DC2, WM, WN, CAUSAL, KVRES, SEQ, DC3P, SDMA, PERSIST, QSPLIT, PAD, SWZ>;
  constexpr int LD1_ = DC1 + PAD, LDP_ = SWZ ? BQ : (BQ + PAD), LD2_ = DC2 + PAD;
  constexpr int kArena_ = BQ * LD1_ > BQ * LD2_ ? BQ * LD1_ : BQ * LD2_;
  constexpr int DC3_ = DC3P;
  constexpr int LDK_ = KVRES ? (D + PAD) : ((DC1 > DC3_ ? DC1 : DC3_) + PAD);
  static_assert(!KVRES || DC3_ <= D, "");
  constexpr int kArenaA_ = 4 * BQ * LD1_ > BQ * LD2_ ? 4 * BQ * LD1_ : BQ * LD2_;
  constexpr int kArenaB_ = kArenaA_ > 2 * BQ * DC3_ ? kArenaA_ : 2 * BQ * DC3_;
  const size_t sh = (size_t)(BK * LDP_ + kArenaB_ + 2 * BK * LDK_) * sizeof(__half);
  if (sh > 96 * 1024) { printf("  BK=%-3d BQ=%-3d D=%-4d варпы %dx%-2d | ПРОПУСК: smem %zu Б > 96 КБ\n",
                               BK, BQ, D, WM, WN, sh); return true; }
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  // [PERSIST] сетка = ЧИСЛО SM (80), а не (Sk/BK)xH: при H=1 статическая сетка = 64 блока и 16 SM
  // простаивают весь запуск; при H=8 это 6.4 волны и в последней занято 32 SM из 80.
  const int nJobs = (Sk / BK) * QSPLIT * H;
  const int nCTA  = PERSIST ? (nJobs < 80 ? nJobs : 80) : 0;
  auto go = [&] {
    CK(cudaMemset(ddQ, 0, (size_t)Sq * D * 4 * H)); // dQ копится атомарно всеми блоками ключей
    if constexpr (PERSIST) {
      const int zero = 0; CK(cudaMemcpyToSymbol(g_ticket, &zero, 4));
      if constexpr (QSPLIT > 1) {          // срезы складываются -> накопители обязаны стартовать с нуля
        CK(cudaMemset(ddV, 0, (size_t)Sk * D * 4 * H)); CK(cudaMemset(ddK, 0, (size_t)Sk * D * 4 * H));
      }
      kern<<<nCTA, WM * WN * 32, sh>>>(dQ, dKk, dVv, ddO, dlse, dDel, ddV, ddK, ddQ, Sq, Sk, nJobs);
    } else
      kern<<<dim3(Sk / BK, H), WM * WN * 32, sh>>>(dQ, dKk, dVv, ddO, dlse, dDel, ddV, ddK, ddQ, Sq, Sk, 0);
  };
  go(); CK(cudaDeviceSynchronize());
  if (getenv("FA2_PROF")) {                      // ncu: один запуск на вариант, без эталона
    cudaFuncAttributes at_; cudaFuncGetAttributes(&at_, (const void*)kern);
    printf("  D=%d H=%d QSPLIT=%d PERSIST=%d сетка %d\n", D, H, QSPLIT, (int)PERSIST,
           PERSIST ? nCTA : (Sk / BK) * H);
    go(); CK(cudaDeviceSynchronize());
    cudaFree(dQ); cudaFree(dKk); cudaFree(dVv); cudaFree(ddO);
    cudaFree(dlse); cudaFree(dDel); cudaFree(ddV); cudaFree(ddK); cudaFree(ddQ);
    return true;
  }
  const bool noverify = getenv("FA2_NOVERIFY") != nullptr;   // скорость мерим без эталона на CPU
  std::vector<float> hdV((size_t)Sk * D), hdK((size_t)Sk * D), hdQ((size_t)Sq * D);
  CK(cudaMemcpy(hdV.data(), ddV, (size_t)Sk * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hdK.data(), ddK, (size_t)Sk * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hdQ.data(), ddQ, (size_t)Sq * D * 4, cudaMemcpyDeviceToHost));

  double rV = -1, rK = -1, rQ = -1; int bad = 0;
  cudaFuncAttributes at; cudaFuncGetAttributes(&at, (const void*)kern);
  if (!noverify) {
  const double scale = 1.0 / sqrt((double)D);
  double nV = 0, eV = 0, nK = 0, eK = 0, nQ = 0, eQ = 0;
  std::vector<double> refQ((size_t)Sq * D, 0.0);
  for (int k = 0; k < Sk; ++k) {
    std::vector<double> av(D, 0.0), ak(D, 0.0);
    for (int q = 0; q < Sq; ++q) {
      double s = 0, dp = 0;
      for (int t = 0; t < D; ++t) {
        s  += (double)__half2float(hK[(size_t)k * D + t]) * __half2float(hQ[(size_t)q * D + t]);
        dp += (double)__half2float(hV[(size_t)k * D + t]) * __half2float(hdO[(size_t)q * D + t]);
      }
      float p = __half2float(__float2half((float)exp(s * scale - hlse[q])));
      if (CAUSAL && k > q) p = 0.f;
      const float ds = __half2float(__float2half((float)(p * (dp - hDelta[q]))));
      for (int t = 0; t < D; ++t) {
        av[t] += (double)p  * __half2float(hdO[(size_t)q * D + t]);
        ak[t] += (double)ds * __half2float(hQ[(size_t)q * D + t]);
        refQ[(size_t)q * D + t] += (double)ds * __half2float(hK[(size_t)k * D + t]);
      }
    }
    for (int t = 0; t < D; ++t) {
      const double gv = hdV[(size_t)k * D + t], gk = hdK[(size_t)k * D + t];
      if (!isfinite(gv) || !isfinite(gk)) ++bad;
      eV += (gv - av[t]) * (gv - av[t]); nV += av[t] * av[t];
      eK += (gk - ak[t]) * (gk - ak[t]); nK += ak[t] * ak[t];
    }
  }
  for (size_t i = 0; i < refQ.size(); ++i) {
    if (!isfinite(hdQ[i])) ++bad;
    eQ += (hdQ[i] - refQ[i]) * (hdQ[i] - refQ[i]); nQ += refQ[i] * refQ[i];
  }
  rV = sqrt(eV / fmax(nV, 1e-30)); rK = sqrt(eK / fmax(nK, 1e-30));
  rQ = sqrt(eQ / fmax(nQ, 1e-30));
  }
  const bool ok = noverify || (rV < 3e-3 && rK < 3e-3 && rQ < 3e-3 && bad == 0);
  double ms = 0;
  if (bench) {
    for (int i = 0; i < 5; ++i) go();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) go();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float t; cudaEventElapsedTime(&t, a, b); ms = t / 20;
  }
  {
    const int bl_smem = (int)(98304 / sh);                                  // блоков на SM по smem
    const int bl_reg  = 65536 / (at.numRegs * WM * WN * 32) ;               // и по регистрам
    const int bl = bl_smem < bl_reg ? bl_smem : bl_reg;
    printf("smem %2zu КБ бл/SM %d(smem %d,рег %d) варп/SM %2d | ", sh >> 10, bl, bl_smem, bl_reg, bl * WM * WN);
  }
  printf("dV %.0e dK %.0e dQ %.0e рег %3d разлив %4d Б", rV, rK, rQ, at.numRegs, (int)at.localSizeBytes);
  if (bench) {
    // При причинности полезная работа — пары (ключ,запрос) с key<=query.
    const double pairs = CAUSAL ? (double)Sq * (Sq + 1) / 2.0 : (double)Sq * Sk;
    const double flop = 5.0 * pairs * D * 2.0 * H;
    printf(" | %.3f мс  %.1f ТФЛОП/с", ms, flop / (ms * 1e-3) / 1e12);
  }
  printf("  %s\n", ok ? "ОК" : "ОТКАЗ");
  cudaFree(dQ); cudaFree(dKk); cudaFree(dVv); cudaFree(ddO);
  cudaFree(dlse); cudaFree(dDel); cudaFree(ddV); cudaFree(ddK); cudaFree(ddQ);
  return ok;
}


// ---------------------------------------------------------------------------------------------
// Вариант 2: резидентны Q и dO, а не K и V.
// Трафик считается точно. На пару (блок ключей, блок запросов) первый вариант читает Q и dO ДВАЖДЫ
// (фаза 1, затем фазы 2/3): 4*D*Sq*Sk/BK. Резидентность K/V экономит лишь D*Sq*Sk/BQ — при BQ=128
// это в 16 раз меньше, то есть почти ничего, — и при этом ЗАПРЕЩАЕТ BK=64, который вдвое режет
// доминирующий член. Обмен был невыгодным.
// Здесь наоборот: Q и dO целиком лежат в smem на весь блок запросов (их читают фазы 1,2,3), а K/V
// текут кусками с двойной буферизацией. Фазы 2 и 3 теряют цикл по d вместе со всеми барьерами:
// выход [BK][D] делится между варпами сразу, потому что операнд целиком под рукой.
// Фаза 4 переиспользует буфер Q под K — к этому моменту Q больше не нужен.
// Трафик: 5*D*Sq*Sk/64 против 8*D*Sq*Sk/64, то есть в 1.6 раза меньше.
template <int BK, int BQ, int D, int DC1, int WM, int WN, bool CAUSAL>
__global__ __launch_bounds__(512) void k_qres(const __half* __restrict__ Q,
                                              const __half* __restrict__ Kk,
                                              const __half* __restrict__ Vv,
                                              const __half* __restrict__ dO,
                                              const float* __restrict__ lse,
                                              const float* __restrict__ Delta,
                                              float* __restrict__ dV, float* __restrict__ dK,
                                              float* __restrict__ dQ, int Sq, int Sk, int nJobs = 0) {
  constexpr int kWarps = WM * WN;   // [ГЕОМЕТРИЯ] раньше было прибито 16; теперь плитка варпа
                                    // TM x TN = BK/WM x BQ/WN перебирается, и с ней загрузок-на-mma
  constexpr int LDQ = D + 4, LDP = BQ + 4, LD1 = DC1 + 4;
  constexpr int TM = BK / WM, TN = BQ / WN;      // плитка варпа для S/dP^T
  constexpr int GM = BK / WM, GN = D / WN;       // плитка варпа для dV/dK — весь d сразу
  constexpr int QM = BQ / WM, QN = D / WN;       // плитка варпа для dQ
  static_assert(TM % 16 == 0 && TN % 16 == 0 && GM % 16 == 0 && GN % 16 == 0, "плитки кратны 16");
  static_assert(QM % 16 == 0 && QN % 16 == 0, "плитка dQ кратна 16");

  extern __shared__ __half smem[];
  __half* sQ  = smem;                       // Q целиком [BQ][LDQ]; в фазе 4 переиспользуется под K
  __half* sdO = sQ + BQ * LDQ;              // dO целиком [BQ][LDQ]
  __half* sP  = sdO + BQ * LDQ;             // P / dS
  __half* kb0 = sP + BK * LDP;              // K/V кусками, двойной буфер
  __half* kbuf[2] = {kb0, kb0 + BK * LD1};
  __half* vbuf[2] = {kb0 + 2 * BK * LD1, kb0 + 3 * BK * LD1};

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int kb = blockIdx.x * BK;
  {
    const long h = (long)blockIdx.y;
    Q += h * (long)Sq * D;  Kk += h * (long)Sk * D; Vv += h * (long)Sk * D;
    dO += h * (long)Sq * D; lse += h * (long)Sq;    Delta += h * (long)Sq;
    dV += h * (long)Sk * D; dK += h * (long)Sk * D; dQ += h * (long)Sq * D;
  }

  WarpMma<GM / 16, GN / 16, BLayout::kNMajor> gv, gk;
  gv.clear(); gk.clear();

  // Загрузка куска K/V: BK*DC1 половинок на 512 потоков, по 8 на поток -> LDG.128, STS двумя по 8 Б
  // Плитка K/V мала (BK*DC1 половинок), поэтому 16 байт на поток достаётся не всем: грузим с
  // предикатом. Узкая загрузка была бы хуже — 4 байта на поток дают вчетверо больше транзакций.
  constexpr int kChunksKV = BK * DC1 / 8, kChkRow = DC1 / 8;
  constexpr int kVecKV = (kChunksKV + kWarps * 32 - 1) / (kWarps * 32);
  uint4 stk[kVecKV], stv[kVecKV];
  auto ldg_kv = [&](int dcx) {
#pragma unroll
    for (int v = 0; v < kVecKV; ++v) {
      const int idx = tid + v * kWarps * 32;
      if (idx >= kChunksKV) break;
      const int r = idx / kChkRow, c = (idx % kChkRow) * 8;
      stk[v] = *reinterpret_cast<const uint4*>(&Kk[(long)(kb + r) * D + dcx + c]);
      stv[v] = *reinterpret_cast<const uint4*>(&Vv[(long)(kb + r) * D + dcx + c]);
    }
  };
  auto sts_kv = [&](int b) {
#pragma unroll
    for (int v = 0; v < kVecKV; ++v) {
      const int idx = tid + v * kWarps * 32;
      if (idx >= kChunksKV) break;
      const int r = idx / kChkRow, c = (idx % kChkRow) * 8;
      __half* pk = &kbuf[b][r * LD1 + c]; __half* pv = &vbuf[b][r * LD1 + c];
      *reinterpret_cast<uint2*>(pk)     = make_uint2(stk[v].x, stk[v].y);
      *reinterpret_cast<uint2*>(pk + 4) = make_uint2(stk[v].z, stk[v].w);
      *reinterpret_cast<uint2*>(pv)     = make_uint2(stv[v].x, stv[v].y);
      *reinterpret_cast<uint2*>(pv + 4) = make_uint2(stv[v].z, stv[v].w);
    }
  };

  const int qStart = CAUSAL ? (kb / BQ) * BQ : 0;
  for (int q0 = qStart; q0 < Sq; q0 += BQ) {
    // ---- Q и dO целиком, ОДИН раз на блок запросов
    __syncthreads();
    for (int i = tid; i < BQ * D; i += kWarps * 32) {
      const int r = i / D, c = i % D;
      sQ[r * LDQ + c]  = Q[(long)(q0 + r) * D + c];
      sdO[r * LDQ + c] = dO[(long)(q0 + r) * D + c];
    }
    // ---- фаза 1: S и dP^T, K/V текут кусками с перекрытием
    WarpMma<TM / 16, TN / 16, BLayout::kKMajor> qk, dpt;
    qk.clear(); dpt.clear();
    ldg_kv(0); sts_kv(0);
    __syncthreads();
    int cur = 0;
    for (int dc = 0; dc < D; dc += DC1) {
      if (dc + DC1 < D) ldg_kv(dc + DC1);
      qk.template accumulate<DC1>(kbuf[cur], LD1, wm * TM, sQ + dc, LDQ, wn * TN, lane);
      dpt.template accumulate<DC1>(vbuf[cur], LD1, wm * TM, sdO + dc, LDQ, wn * TN, lane);
      if (dc + DC1 < D) { __syncthreads(); sts_kv(cur ^ 1); __syncthreads(); cur ^= 1; }
    }
    const float scale = rsqrtf((float)D);
    const int r0 = wm * TM, c0 = wn * TN;
    qk.visit(lane, [&](float& v, int r, int c) {
      const float p = __expf(v * scale - lse[q0 + c0 + c]);
      v = (CAUSAL && (kb + r0 + r) > (q0 + c0 + c)) ? 0.f : p;
    });
    __syncthreads();
    qk.store_half(sP, LDP, r0, c0, lane);
    __syncthreads();
    // ---- фаза 2: dV += P^T*dO. Ни цикла по d, ни барьеров: dO целиком под рукой.
    gv.template accumulate<BQ>(sP, LDP, wm * GM, sdO, LDQ, wn * GN, lane);
    // ---- dS, затем dK += dS*Q — тоже одним заходом
    {
      const float* dp = &dpt.acc[0][0][0];
      int idx = 0;
      qk.visit(lane, [&](float& p, int r, int c) { p = p * (dp[idx++] - Delta[q0 + c0 + c]); });
    }
    __syncthreads();
    qk.store_half(sP, LDP, r0, c0, lane);
    __syncthreads();
    gk.template accumulate<BQ>(sP, LDP, wm * GM, sQ, LDQ, wn * GN, lane);
    // ---- фаза 4: dQ += dS^T*K. Q больше не нужен -> кладём K в его буфер целиком.
    __syncthreads();
    for (int i = tid; i < BK * D; i += kWarps * 32)
      sQ[(i / D) * LDQ + i % D] = Kk[(long)(kb + i / D) * D + i % D];
    __syncthreads();
    WarpMma<QM / 16, QN / 16, BLayout::kNMajor, ALayout::kMMajor> gq;
    gq.clear();
    gq.template accumulate<BK>(sP, LDP, wm * QM, sQ, LDQ, wn * QN, lane);
    gq.visit(lane, [&](float v, int r, int c) {
      atomicAdd(&dQ[(long)(q0 + wm * QM + r) * D + wn * QN + c], v);
    });
  }
  gv.visit(lane, [&](float v, int r, int c) { dV[(long)(kb + wm * GM + r) * D + wn * GN + c] = v; });
  gk.visit(lane, [&](float v, int r, int c) { dK[(long)(kb + wm * GM + r) * D + wn * GN + c] = v; });
}

template <int BK, int BQ, int D, int DC1, int WM, int WN, bool CAUSAL>
bool run_qres(int Sq, int Sk, int H) {
  std::vector<__half> hQ((size_t)Sq * D), hK((size_t)Sk * D), hV((size_t)Sk * D), hdO((size_t)Sq * D);
  std::vector<float> hlse(Sq), hDelta(Sq);
  for (auto& x : hQ) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hK) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hV) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hdO) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hlse) x = 2.0f + (rand() / (float)RAND_MAX);
  for (auto& x : hDelta) x = (rand() / (float)RAND_MAX) * 0.3f;
  __half *dQb, *dKk, *dVv, *ddO; float *dlse, *dDel, *ddV, *ddK, *ddQ;
  CK(cudaMalloc(&dQb, hQ.size() * 2 * H));  CK(cudaMalloc(&dKk, hK.size() * 2 * H));
  CK(cudaMalloc(&dVv, hV.size() * 2 * H));  CK(cudaMalloc(&ddO, hdO.size() * 2 * H));
  CK(cudaMalloc(&dlse, (size_t)Sq * 4 * H)); CK(cudaMalloc(&dDel, (size_t)Sq * 4 * H));
  CK(cudaMalloc(&ddV, (size_t)Sk * D * 4 * H)); CK(cudaMalloc(&ddK, (size_t)Sk * D * 4 * H));
  CK(cudaMalloc(&ddQ, (size_t)Sq * D * 4 * H));
  for (int h = 0; h < H; ++h) {
    CK(cudaMemcpy(dQb + (size_t)h * Sq * D, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dKk + (size_t)h * Sk * D, hK.data(), hK.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVv + (size_t)h * Sk * D, hV.data(), hV.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddO + (size_t)h * Sq * D, hdO.data(), hdO.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlse + (size_t)h * Sq, hlse.data(), Sq * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dDel + (size_t)h * Sq, hDelta.data(), Sq * 4, cudaMemcpyHostToDevice));
  }
  CK(cudaMemset(ddV, 0x7f, (size_t)Sk * D * 4 * H)); CK(cudaMemset(ddK, 0x7f, (size_t)Sk * D * 4 * H));
  constexpr int LDQ_ = D + 4, LDP_ = BQ + 4, LD1_ = DC1 + 4;
  const size_t sh = (size_t)(2 * BQ * LDQ_ + BK * LDP_ + 4 * BK * LD1_) * sizeof(__half);
  auto kern = k_qres<BK, BQ, D, DC1, WM, WN, CAUSAL>;
  if (sh > 96 * 1024) { printf("  QRES BK=%-3d BQ=%-3d D=%-4d | ПРОПУСК: smem %zu Б\n", BK, BQ, D, sh); return true; }
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  auto go = [&] {
    CK(cudaMemset(ddQ, 0, (size_t)Sq * D * 4 * H));
    kern<<<dim3(Sk / BK, H), WM * WN * 32, sh>>>(dQb, dKk, dVv, ddO, dlse, dDel, ddV, ddK, ddQ, Sq, Sk);
  };
  go(); CK(cudaDeviceSynchronize());
  std::vector<float> hdV((size_t)Sk * D), hdK((size_t)Sk * D), hdQ((size_t)Sq * D);
  CK(cudaMemcpy(hdV.data(), ddV, (size_t)Sk * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hdK.data(), ddK, (size_t)Sk * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hdQ.data(), ddQ, (size_t)Sq * D * 4, cudaMemcpyDeviceToHost));
  const double scale = 1.0 / sqrt((double)D);
  double nV = 0, eV = 0, nK = 0, eK = 0, nQ = 0, eQ = 0; int bad = 0;
  std::vector<double> refQ((size_t)Sq * D, 0.0);
  for (int k = 0; k < Sk; ++k) {
    std::vector<double> av(D, 0.0), ak(D, 0.0);
    for (int q = 0; q < Sq; ++q) {
      double s = 0, dp = 0;
      for (int t = 0; t < D; ++t) {
        s  += (double)__half2float(hK[(size_t)k * D + t]) * __half2float(hQ[(size_t)q * D + t]);
        dp += (double)__half2float(hV[(size_t)k * D + t]) * __half2float(hdO[(size_t)q * D + t]);
      }
      float p = __half2float(__float2half((float)exp(s * scale - hlse[q])));
      if (CAUSAL && k > q) p = 0.f;
      const float ds = __half2float(__float2half((float)(p * (dp - hDelta[q]))));
      for (int t = 0; t < D; ++t) {
        av[t] += (double)p  * __half2float(hdO[(size_t)q * D + t]);
        ak[t] += (double)ds * __half2float(hQ[(size_t)q * D + t]);
        refQ[(size_t)q * D + t] += (double)ds * __half2float(hK[(size_t)k * D + t]);
      }
    }
    for (int t = 0; t < D; ++t) {
      const double gvv = hdV[(size_t)k * D + t], gkk = hdK[(size_t)k * D + t];
      if (!isfinite(gvv) || !isfinite(gkk)) ++bad;
      eV += (gvv - av[t]) * (gvv - av[t]); nV += av[t] * av[t];
      eK += (gkk - ak[t]) * (gkk - ak[t]); nK += ak[t] * ak[t];
    }
  }
  for (size_t i = 0; i < refQ.size(); ++i) {
    if (!isfinite(hdQ[i])) ++bad;
    eQ += (hdQ[i] - refQ[i]) * (hdQ[i] - refQ[i]); nQ += refQ[i] * refQ[i];
  }
  for (int i = 0; i < 5; ++i) go();
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
  for (int i = 0; i < 20; ++i) go();
  cudaEventRecord(b); CK(cudaEventSynchronize(b));
  float t; cudaEventElapsedTime(&t, a, b); const double ms = t / 20;
  const double pairs = CAUSAL ? (double)Sq * (Sq + 1) / 2.0 : (double)Sq * Sk;
  const double flop = 5.0 * pairs * D * 2.0 * H;
  cudaFuncAttributes at; cudaFuncGetAttributes(&at, (const void*)kern);
  const double rV = sqrt(eV / fmax(nV, 1e-30)), rK = sqrt(eK / fmax(nK, 1e-30)), rQ = sqrt(eQ / fmax(nQ, 1e-30));
  const bool ok = (rV < 3e-3 && rK < 3e-3 && rQ < 3e-3 && bad == 0);
  printf("  QRES BK=%-3d BQ=%-3d D=%-4d H=%-2d | dV %.1e dK %.1e dQ %.1e | рег %3d разлив %d Б smem %5zu | %.3f мс %5.1f ТФЛОП/с %s\n",
         BK, BQ, D, H, rV, rK, rQ, at.numRegs, (int)at.localSizeBytes, sh, ms, flop / (ms * 1e-3) / 1e12, ok ? "ОК" : "ОТКАЗ");
  cudaFree(dQb); cudaFree(dKk); cudaFree(dVv); cudaFree(ddO);
  cudaFree(dlse); cudaFree(dDel); cudaFree(ddV); cudaFree(ddK); cudaFree(ddQ);
  return ok;
}


// ---------------------------------------------------------------------------------------------
// Раздельный проход dV с плиткой варпа 32x32 в ОБОИХ gemm.
// Профиль сказал: узкое место — разделённая память, и виновата геометрия. При 16 варпах и блочной
// плитке [32][128] плитка варпа выходит 16x16, то есть MB=NB=1 -> ДВЕ загрузки фрагмента на mma.
// Здесь блочная плитка [128][128] = 16384 ячейки = ровно 16 варпов по 32x32:
//   S  = K*Q^T : MB=2, NB=2 -> (2+2)/(2*2) = 1.0 загрузки на mma  (как у cutlass)
//   dV += P^T*dO, кусок [128][128]: тоже 32x32 -> 1.0
// Резидентен ТОЛЬКО dV: BK*D/512 = 64 float на поток при d=256. Держать одновременно dV и dK при
// такой плитке нельзя — это и есть цена раздельных проходов.
template <int BK, int BQ, int D, int DC1, int DC2, int WM, int WN, bool CAUSAL>
__global__ __launch_bounds__(512) void k_passV(const __half* __restrict__ Q,
                                               const __half* __restrict__ Kk,
                                               const __half* __restrict__ dO,
                                               const float* __restrict__ lse,
                                               float* __restrict__ dV, __half* __restrict__ Pg,
                                               int Sq, int Sk) {
  constexpr int kWarps = WM * WN;   // [ГЕОМЕТРИЯ] раньше было прибито 16; теперь плитка варпа
                                    // TM x TN = BK/WM x BQ/WN перебирается, и с ней загрузок-на-mma
  constexpr int LD1 = DC1 + 4, LDP = BQ + 4, LD2 = DC2 + 4;
  constexpr int TM = BK / WM, TN = BQ / WN;      // плитка варпа для S
  constexpr int GM = BK / WM, GN = DC2 / WN;     // плитка варпа для dV
  constexpr int kChunks = D / DC2;
  static_assert(TM % 16 == 0 && TN % 16 == 0 && GM % 16 == 0 && GN % 16 == 0, "плитки кратны 16");

  extern __shared__ __half smem[];
  __half* sP = smem;                              // P
  __half* arena = sP + BK * LDP;                  // фаза 1: Q двойным буфером; фаза 2: кусок dO
  constexpr int kArena = (2 * BQ * LD1 > BQ * LD2) ? 2 * BQ * LD1 : BQ * LD2;
  __half* qbuf[2] = {arena, arena + BQ * LD1};
  __half* kb0 = arena + kArena;
  __half* kbuf[2] = {kb0, kb0 + BK * LD1};

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int kb = blockIdx.x * BK;
  {
    const long h = (long)blockIdx.y;
    Q += h * (long)Sq * D; Kk += h * (long)Sk * D; dO += h * (long)Sq * D;
    lse += h * (long)Sq;   dV += h * (long)Sk * D;  Pg += h * (long)Sk * Sq;
  }

  WarpMma<GM / 16, GN / 16, BLayout::kNMajor> gv[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) gv[c].clear();

  int qb_ = 0;
  constexpr int kChkRow = DC1 / 8;
  constexpr int kVecQ = BQ * DC1 / (kWarps * 32 * 8), kVecK = BK * DC1 / (kWarps * 32 * 8);
  uint4 stq[kVecQ > 0 ? kVecQ : 1], stk[kVecK > 0 ? kVecK : 1];
  auto ldg = [&](int dcx) {
#pragma unroll
    for (int v = 0; v < kVecQ; ++v) {
      const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
      stq[v] = *reinterpret_cast<const uint4*>(&Q[(long)(qb_ + r) * D + dcx + c]);
    }
#pragma unroll
    for (int v = 0; v < kVecK; ++v) {
      const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
      stk[v] = *reinterpret_cast<const uint4*>(&Kk[(long)(kb + r) * D + dcx + c]);
    }
  };
  auto sts = [&](int b) {
#pragma unroll
    for (int v = 0; v < kVecQ; ++v) {
      const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
      __half* p = &qbuf[b][r * LD1 + c];
      *reinterpret_cast<uint2*>(p)     = make_uint2(stq[v].x, stq[v].y);
      *reinterpret_cast<uint2*>(p + 4) = make_uint2(stq[v].z, stq[v].w);
    }
#pragma unroll
    for (int v = 0; v < kVecK; ++v) {
      const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
      __half* p = &kbuf[b][r * LD1 + c];
      *reinterpret_cast<uint2*>(p)     = make_uint2(stk[v].x, stk[v].y);
      *reinterpret_cast<uint2*>(p + 4) = make_uint2(stk[v].z, stk[v].w);
    }
  };

  const int qStart = CAUSAL ? (kb / BQ) * BQ : 0;
  for (qb_ = qStart; qb_ < Sq; qb_ += BQ) {   // НЕ затенять: лямбды загрузки захватывают именно её
    WarpMma<TM / 16, TN / 16, BLayout::kKMajor> qk;
    qk.clear();
    __syncthreads();
    ldg(0); sts(0);
    __syncthreads();
    int cur = 0;
    for (int dc = 0; dc < D; dc += DC1) {
      if (dc + DC1 < D) ldg(dc + DC1);
      qk.template accumulate<DC1>(kbuf[cur], LD1, wm * TM, qbuf[cur], LD1, wn * TN, lane);
      if (dc + DC1 < D) { __syncthreads(); sts(cur ^ 1); __syncthreads(); cur ^= 1; }
    }
    const float scale = rsqrtf((float)D);
    const int r0 = wm * TM, c0 = wn * TN;
    qk.visit(lane, [&](float& v, int r, int c) {
      const float p = __expf(v * scale - lse[qb_ + c0 + c]);
      v = (CAUSAL && (kb + r0 + r) > (qb_ + c0 + c)) ? 0.f : p;
    });
    __syncthreads();
    qk.store_half(sP, LDP, r0, c0, lane);
    __syncthreads();
    // P уходит в глобальную память ОДНИМ коалесцированным проходом: проходу 2 дешевле её прочитать
    // (16 МБ на голову, ~0.16 мс при H=8), чем пересчитать S (0.51 мс).
    for (int i = tid; i < BK * BQ; i += kWarps * 32)
      Pg[(long)(kb + i / BQ) * Sq + qb_ + i % BQ] = sP[(i / BQ) * LDP + i % BQ];
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      __syncthreads();
      for (int i = tid; i < BQ * DC2; i += kWarps * 32)
        arena[(i / DC2) * LD2 + i % DC2] = dO[(long)(qb_ + i / DC2) * D + c * DC2 + i % DC2];
      __syncthreads();
      gv[c].template accumulate<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
    }
  }
#pragma unroll
  for (int c = 0; c < kChunks; ++c)
    gv[c].visit(lane, [&](float v, int r, int cc) {
      dV[(long)(kb + wm * GM + r) * D + c * DC2 + wn * GN + cc] = v;
    });
}

template <int BK, int BQ, int D, int DC1, int DC2, int WM, int WN, bool CAUSAL>
bool run_passV(int Sq, int Sk, int H) {
  std::vector<__half> hQ((size_t)Sq * D), hK((size_t)Sk * D), hdO((size_t)Sq * D);
  std::vector<float> hlse(Sq);
  for (auto& x : hQ) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hK) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hdO) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hlse) x = 2.0f + (rand() / (float)RAND_MAX);
  __half *dQb, *dKk, *ddO, *dPg; float *dlse, *ddV;
  CK(cudaMalloc(&dQb, hQ.size() * 2 * H)); CK(cudaMalloc(&dKk, hK.size() * 2 * H));
  CK(cudaMalloc(&ddO, hdO.size() * 2 * H)); CK(cudaMalloc(&dlse, (size_t)Sq * 4 * H));
  CK(cudaMalloc(&ddV, (size_t)Sk * D * 4 * H));
  CK(cudaMalloc(&dPg, (size_t)Sk * Sq * 2 * H)); CK(cudaMemset(dPg, 0, (size_t)Sk * Sq * 2 * H));
  for (int h = 0; h < H; ++h) {
    CK(cudaMemcpy(dQb + (size_t)h * Sq * D, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dKk + (size_t)h * Sk * D, hK.data(), hK.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddO + (size_t)h * Sq * D, hdO.data(), hdO.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlse + (size_t)h * Sq, hlse.data(), Sq * 4, cudaMemcpyHostToDevice));
  }
  CK(cudaMemset(ddV, 0x7f, (size_t)Sk * D * 4 * H));
  constexpr int LD1_ = DC1 + 4, LDP_ = BQ + 4, LD2_ = DC2 + 4;
  constexpr int kArena_ = (2 * BQ * LD1_ > BQ * LD2_) ? 2 * BQ * LD1_ : BQ * LD2_;
  const size_t sh = (size_t)(BK * LDP_ + kArena_ + 2 * BK * LD1_) * sizeof(__half);
  auto kern = k_passV<BK, BQ, D, DC1, DC2, WM, WN, CAUSAL>;
  if (sh > 96 * 1024) { printf("  passV BK=%-3d BQ=%-3d D=%-4d | ПРОПУСК: smem %zu Б\n", BK, BQ, D, sh); return true; }
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  auto go = [&] { kern<<<dim3(Sk / BK, H), WM * WN * 32, sh>>>(dQb, dKk, ddO, dlse, ddV, dPg, Sq, Sk); };
  go(); CK(cudaDeviceSynchronize());
  std::vector<float> hdV((size_t)Sk * D);
  CK(cudaMemcpy(hdV.data(), ddV, (size_t)Sk * D * 4, cudaMemcpyDeviceToHost));
  const double scale = 1.0 / sqrt((double)D);
  double nV = 0, eV = 0; int bad = 0;
  for (int k = 0; k < Sk; ++k) {
    std::vector<double> av(D, 0.0);
    for (int q = 0; q < Sq; ++q) {
      double s = 0;
      for (int t = 0; t < D; ++t) s += (double)__half2float(hK[(size_t)k * D + t]) * __half2float(hQ[(size_t)q * D + t]);
      float p = __half2float(__float2half((float)exp(s * scale - hlse[q])));
      if (CAUSAL && k > q) p = 0.f;
      for (int t = 0; t < D; ++t) av[t] += (double)p * __half2float(hdO[(size_t)q * D + t]);
    }
    for (int t = 0; t < D; ++t) {
      const double g = hdV[(size_t)k * D + t];
      if (!isfinite(g)) ++bad;
      eV += (g - av[t]) * (g - av[t]); nV += av[t] * av[t];
    }
  }
  for (int i = 0; i < 5; ++i) go();
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
  for (int i = 0; i < 20; ++i) go();
  cudaEventRecord(b); CK(cudaEventSynchronize(b));
  float t; cudaEventElapsedTime(&t, a, b); const double ms = t / 20;
  const double pairs = CAUSAL ? (double)Sq * (Sq + 1) / 2.0 : (double)Sq * Sk;
  const double flop = 2.0 * pairs * D * 2.0 * H;      // ДВА gemm: S и dV
  cudaFuncAttributes at; cudaFuncGetAttributes(&at, (const void*)kern);
  const double rV = sqrt(eV / fmax(nV, 1e-30));
  const bool ok = (rV < 3e-3 && bad == 0);
  printf("  passV BK=%-3d BQ=%-3d D=%-4d H=%-2d | dV %.1e | рег %3d разлив %d Б smem %5zu | %.3f мс %5.1f ТФЛОП/с %s\n",
         BK, BQ, D, H, rV, at.numRegs, (int)at.localSizeBytes, sh, ms, flop / (ms * 1e-3) / 1e12, ok ? "ОК" : "ОТКАЗ");
  cudaFree(dQb); cudaFree(dKk); cudaFree(ddO); cudaFree(dlse); cudaFree(ddV); cudaFree(dPg);
  return ok;
}


// ---------------------------------------------------------------------------------------------
// Проход 2: dP^T, dS, dK, dQ. P берётся из глобальной памяти (её записал проход 1) — вместо
// пересчёта S. Резидентен ТОЛЬКО dK, поэтому блочная плитка [128][128] снова даёт 16 варпов по 32x32.
template <int BK, int BQ, int D, int DC1, int DC2, int DC3, int WM, int WN, bool CAUSAL>
__global__ __launch_bounds__(512) void k_passKQ(const __half* __restrict__ Pg,
                                                const __half* __restrict__ Q,
                                                const __half* __restrict__ Kk,
                                                const __half* __restrict__ Vv,
                                                const __half* __restrict__ dO,
                                                const float* __restrict__ Delta,
                                                float* __restrict__ dK, float* __restrict__ dQ,
                                                int Sq, int Sk) {
  constexpr int kWarps = WM * WN;   // [ГЕОМЕТРИЯ] раньше было прибито 16; теперь плитка варпа
                                    // TM x TN = BK/WM x BQ/WN перебирается, и с ней загрузок-на-mma
  constexpr int LD1 = DC1 + 4, LDP = BQ + 4, LD2 = DC2 + 4, LD3 = DC3 + 4;
  constexpr int TM = BK / WM, TN = BQ / WN;        // dP^T
  constexpr int GM = BK / WM, GN = DC2 / WN;       // dK
  constexpr int QM = BQ / WM, QN = DC3 / WN;       // dQ
  constexpr int kChunksK = D / DC2, kChunksQ = D / DC3;
  static_assert(TM % 16 == 0 && TN % 16 == 0 && GM % 16 == 0 && GN % 16 == 0, "плитки кратны 16");
  static_assert(QM % 16 == 0 && QN % 16 == 0, "плитка dQ кратна 16");

  extern __shared__ __half smem[];
  constexpr int kArena = (2 * BQ * LD1 + 2 * BK * LD1) > (BQ * LD2 > BK * LD3 ? BQ * LD2 : BK * LD3)
                       ? (2 * BQ * LD1 + 2 * BK * LD1)
                       : (BQ * LD2 > BK * LD3 ? BQ * LD2 : BK * LD3);
  __half* sP = smem;                     // P, затем dS
  __half* arena = sP + BK * LDP;
  __half* obuf[2] = {arena, arena + BQ * LD1};                       // dO кусками (фаза A)
  __half* vbuf[2] = {arena + 2 * BQ * LD1, arena + 2 * BQ * LD1 + BK * LD1};  // V кусками

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int kb = blockIdx.x * BK;
  {
    const long h = (long)blockIdx.y;
    Pg += h * (long)Sk * Sq; Q += h * (long)Sq * D; Kk += h * (long)Sk * D;
    Vv += h * (long)Sk * D;  dO += h * (long)Sq * D; Delta += h * (long)Sq;
    dK += h * (long)Sk * D;  dQ += h * (long)Sq * D;
  }

  WarpMma<GM / 16, GN / 16, BLayout::kNMajor> gk[kChunksK];
#pragma unroll
  for (int c = 0; c < kChunksK; ++c) gk[c].clear();

  constexpr int kChkRow = DC1 / 8;
  constexpr int kVecO = BQ * DC1 / (kWarps * 32 * 8), kVecV = BK * DC1 / (kWarps * 32 * 8);
  uint4 sto[kVecO > 0 ? kVecO : 1], stv[kVecV > 0 ? kVecV : 1];
  int qb_ = 0;
  auto ldg = [&](int dcx) {
#pragma unroll
    for (int v = 0; v < kVecO; ++v) {
      const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
      sto[v] = *reinterpret_cast<const uint4*>(&dO[(long)(qb_ + r) * D + dcx + c]);
    }
#pragma unroll
    for (int v = 0; v < kVecV; ++v) {
      const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
      stv[v] = *reinterpret_cast<const uint4*>(&Vv[(long)(kb + r) * D + dcx + c]);
    }
  };
  auto sts = [&](int b) {
#pragma unroll
    for (int v = 0; v < kVecO; ++v) {
      const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
      __half* p = &obuf[b][r * LD1 + c];
      *reinterpret_cast<uint2*>(p)     = make_uint2(sto[v].x, sto[v].y);
      *reinterpret_cast<uint2*>(p + 4) = make_uint2(sto[v].z, sto[v].w);
    }
#pragma unroll
    for (int v = 0; v < kVecV; ++v) {
      const int idx = tid + v * kWarps * 32, r = idx / kChkRow, c = (idx % kChkRow) * 8;
      __half* p = &vbuf[b][r * LD1 + c];
      *reinterpret_cast<uint2*>(p)     = make_uint2(stv[v].x, stv[v].y);
      *reinterpret_cast<uint2*>(p + 4) = make_uint2(stv[v].z, stv[v].w);
    }
  };

  const int qStart = CAUSAL ? (kb / BQ) * BQ : 0;
  for (qb_ = qStart; qb_ < Sq; qb_ += BQ) {
    __syncthreads();
    for (int i = tid; i < BK * BQ; i += kWarps * 32)
      sP[(i / BQ) * LDP + i % BQ] = Pg[(long)(kb + i / BQ) * Sq + qb_ + i % BQ];
    WarpMma<TM / 16, TN / 16, BLayout::kKMajor> dpt;
    dpt.clear();
    ldg(0); sts(0);
    __syncthreads();
    int cur = 0;
    for (int dc = 0; dc < D; dc += DC1) {
      if (dc + DC1 < D) ldg(dc + DC1);
      dpt.template accumulate<DC1>(vbuf[cur], LD1, wm * TM, obuf[cur], LD1, wn * TN, lane);
      if (dc + DC1 < D) { __syncthreads(); sts(cur ^ 1); __syncthreads(); cur ^= 1; }
    }
    // dS = P*(dP^T - Delta): P лежит в smem, читаем её в раскладке аккумулятора
    const int r0 = wm * TM, c0 = wn * TN;
    dpt.visit(lane, [&](float& v, int r, int c) {
      const float p = __half2float(sP[(r0 + r) * LDP + c0 + c]);
      v = p * (v - Delta[qb_ + c0 + c]);
    });
    __syncthreads();
    dpt.store_half(sP, LDP, r0, c0, lane);
    __syncthreads();
    // dK += dS*Q  (.row.row)
#pragma unroll
    for (int c = 0; c < kChunksK; ++c) {
      __syncthreads();
      for (int i = tid; i < BQ * DC2; i += kWarps * 32)
        arena[(i / DC2) * LD2 + i % DC2] = Q[(long)(qb_ + i / DC2) * D + c * DC2 + i % DC2];
      __syncthreads();
      gk[c].template accumulate<BQ>(sP, LDP, wm * GM, arena, LD2, wn * GN, lane);
    }
    // dQ += dS^T*K  (.col.row)
#pragma unroll
    for (int c = 0; c < kChunksQ; ++c) {
      __syncthreads();
      for (int i = tid; i < BK * DC3; i += kWarps * 32)
        arena[(i / DC3) * LD3 + i % DC3] = Kk[(long)(kb + i / DC3) * D + c * DC3 + i % DC3];
      __syncthreads();
      WarpMma<QM / 16, QN / 16, BLayout::kNMajor, ALayout::kMMajor> gq;
      gq.clear();
      gq.template accumulate<BK>(sP, LDP, wm * QM, arena, LD3, wn * QN, lane);
      gq.visit(lane, [&](float v, int r, int cc) {
        atomicAdd(&dQ[(long)(qb_ + wm * QM + r) * D + c * DC3 + wn * QN + cc], v);
      });
    }
  }
#pragma unroll
  for (int c = 0; c < kChunksK; ++c)
    gk[c].visit(lane, [&](float v, int r, int cc) {
      dK[(long)(kb + wm * GM + r) * D + c * DC2 + wn * GN + cc] = v;
    });
}


// ---------------------------------------------------------------------------------------------
// Связка двух проходов: проверка всех трёх градиентов и суммарное время.
template <int BK, int BQ, int D, int DC1, int DC2, int DC3, int WM, int WN, bool CAUSAL>
bool run_two(int Sq, int Sk, int H, bool quiet = false) {
  std::vector<__half> hQ((size_t)Sq * D), hK((size_t)Sk * D), hV((size_t)Sk * D), hdO((size_t)Sq * D);
  std::vector<float> hlse(Sq), hDelta(Sq);
  for (auto& x : hQ) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hK) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hV) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hdO) x = __float2half((rand() / (float)RAND_MAX - .5f));
  for (auto& x : hlse) x = 2.0f + (rand() / (float)RAND_MAX);
  for (auto& x : hDelta) x = (rand() / (float)RAND_MAX) * 0.3f;
  __half *dQb, *dKk, *dVv, *ddO, *dPg; float *dlse, *dDel, *ddV, *ddK, *ddQ;
  CK(cudaMalloc(&dQb, (size_t)Sq * D * 2 * H)); CK(cudaMalloc(&dKk, (size_t)Sk * D * 2 * H));
  CK(cudaMalloc(&dVv, (size_t)Sk * D * 2 * H)); CK(cudaMalloc(&ddO, (size_t)Sq * D * 2 * H));
  CK(cudaMalloc(&dlse, (size_t)Sq * 4 * H)); CK(cudaMalloc(&dDel, (size_t)Sq * 4 * H));
  CK(cudaMalloc(&ddV, (size_t)Sk * D * 4 * H)); CK(cudaMalloc(&ddK, (size_t)Sk * D * 4 * H));
  CK(cudaMalloc(&ddQ, (size_t)Sq * D * 4 * H)); CK(cudaMalloc(&dPg, (size_t)Sk * Sq * 2 * H));
  CK(cudaMemset(dPg, 0, (size_t)Sk * Sq * 2 * H));
  for (int h = 0; h < H; ++h) {
    CK(cudaMemcpy(dQb + (size_t)h * Sq * D, hQ.data(), (size_t)Sq * D * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dKk + (size_t)h * Sk * D, hK.data(), (size_t)Sk * D * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVv + (size_t)h * Sk * D, hV.data(), (size_t)Sk * D * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddO + (size_t)h * Sq * D, hdO.data(), (size_t)Sq * D * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlse + (size_t)h * Sq, hlse.data(), Sq * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dDel + (size_t)h * Sq, hDelta.data(), Sq * 4, cudaMemcpyHostToDevice));
  }
  constexpr int LD1_ = DC1 + 4, LDP_ = BQ + 4, LD2_ = DC2 + 4, LD3_ = DC3 + 4;
  constexpr int aV_ = (2 * BQ * LD1_ > BQ * LD2_) ? 2 * BQ * LD1_ : BQ * LD2_;
  const size_t shV = (size_t)(BK * LDP_ + aV_ + 2 * BK * LD1_) * sizeof(__half);
  constexpr int a2a_ = 2 * BQ * LD1_ + 2 * BK * LD1_;
  constexpr int a2b_ = BQ * LD2_ > BK * LD3_ ? BQ * LD2_ : BK * LD3_;
  const size_t shK = (size_t)(BK * LDP_ + (a2a_ > a2b_ ? a2a_ : a2b_)) * sizeof(__half);
  auto kv = k_passV<BK, BQ, D, DC1, DC2, WM, WN, CAUSAL>;
  auto kq = k_passKQ<BK, BQ, D, DC1, DC2, DC3, WM, WN, CAUSAL>;
  if (shV > 96 * 1024 || shK > 96 * 1024) {
    if (!quiet) printf("  два прохода BK=%-3d BQ=%-3d D=%-4d DC2=%-3d DC3=%-3d %dx%-2d | ПРОПУСК smem %zu/%zu\n",
                       BK, BQ, D, DC2, DC3, WM, WN, shV, shK);
    return true;
  }
  CK(cudaFuncSetAttribute(kv, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shV));
  CK(cudaFuncSetAttribute(kq, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shK));
  auto go = [&] {
    CK(cudaMemset(ddQ, 0, (size_t)Sq * D * 4 * H));
    kv<<<dim3(Sk / BK, H), 512, shV>>>(dQb, dKk, ddO, dlse, ddV, dPg, Sq, Sk);
    kq<<<dim3(Sk / BK, H), 512, shK>>>(dPg, dQb, dKk, dVv, ddO, dDel, ddK, ddQ, Sq, Sk);
  };
  CK(cudaMemset(ddV, 0x7f, (size_t)Sk * D * 4 * H)); CK(cudaMemset(ddK, 0x7f, (size_t)Sk * D * 4 * H));
  go(); CK(cudaDeviceSynchronize());
  std::vector<float> hdV((size_t)Sk * D), hdK((size_t)Sk * D), hdQ((size_t)Sq * D);
  CK(cudaMemcpy(hdV.data(), ddV, (size_t)Sk * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hdK.data(), ddK, (size_t)Sk * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hdQ.data(), ddQ, (size_t)Sq * D * 4, cudaMemcpyDeviceToHost));
  const double scale = 1.0 / sqrt((double)D);
  double nV = 0, eV = 0, nK = 0, eK = 0, nQ = 0, eQ = 0; int bad = 0;
  std::vector<double> refQ((size_t)Sq * D, 0.0);
  for (int k = 0; k < Sk; ++k) {
    std::vector<double> av(D, 0.0), ak(D, 0.0);
    for (int q = 0; q < Sq; ++q) {
      double s = 0, dp = 0;
      for (int t = 0; t < D; ++t) {
        s  += (double)__half2float(hK[(size_t)k * D + t]) * __half2float(hQ[(size_t)q * D + t]);
        dp += (double)__half2float(hV[(size_t)k * D + t]) * __half2float(hdO[(size_t)q * D + t]);
      }
      float p = __half2float(__float2half((float)exp(s * scale - hlse[q])));
      if (CAUSAL && k > q) p = 0.f;
      const float ds = __half2float(__float2half((float)(p * (dp - hDelta[q]))));
      for (int t = 0; t < D; ++t) {
        av[t] += (double)p  * __half2float(hdO[(size_t)q * D + t]);
        ak[t] += (double)ds * __half2float(hQ[(size_t)q * D + t]);
        refQ[(size_t)q * D + t] += (double)ds * __half2float(hK[(size_t)k * D + t]);
      }
    }
    for (int t = 0; t < D; ++t) {
      const double gv = hdV[(size_t)k * D + t], gkk = hdK[(size_t)k * D + t];
      if (!isfinite(gv) || !isfinite(gkk)) ++bad;
      eV += (gv - av[t]) * (gv - av[t]); nV += av[t] * av[t];
      eK += (gkk - ak[t]) * (gkk - ak[t]); nK += ak[t] * ak[t];
    }
  }
  for (size_t i = 0; i < refQ.size(); ++i) {
    if (!isfinite(hdQ[i])) ++bad;
    eQ += (hdQ[i] - refQ[i]) * (hdQ[i] - refQ[i]); nQ += refQ[i] * refQ[i];
  }
  for (int i = 0; i < 3; ++i) go();
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
  for (int i = 0; i < 10; ++i) go();
  cudaEventRecord(b); CK(cudaEventSynchronize(b));
  float t; cudaEventElapsedTime(&t, a, b); const double ms = t / 10;
  const double pairs = CAUSAL ? (double)Sq * (Sq + 1) / 2.0 : (double)Sq * Sk;
  const double flop = 5.0 * pairs * D * 2.0 * H;
  cudaFuncAttributes aV, aK; cudaFuncGetAttributes(&aV, (const void*)kv); cudaFuncGetAttributes(&aK, (const void*)kq);
  const double rV = sqrt(eV / fmax(nV, 1e-30)), rK = sqrt(eK / fmax(nK, 1e-30)), rQ = sqrt(eQ / fmax(nQ, 1e-30));
  const bool ok = (rV < 3e-3 && rK < 3e-3 && rQ < 3e-3 && bad == 0);
  printf("  BK=%-3d BQ=%-3d D=%-4d DC1=%-3d DC2=%-3d DC3=%-3d %dx%-2d | dV %.1e dK %.1e dQ %.1e | рег %3d/%3d разлив %d/%d | %6.3f мс %5.1f ТФЛОП/с %s\n",
         BK, BQ, D, DC1, DC2, DC3, WM, WN, rV, rK, rQ, aV.numRegs, aK.numRegs,
         (int)aV.localSizeBytes, (int)aK.localSizeBytes, ms, flop / (ms * 1e-3) / 1e12, ok ? "ОК" : "ОТКАЗ");
  cudaFree(dQb); cudaFree(dKk); cudaFree(dVv); cudaFree(ddO); cudaFree(dPg);
  cudaFree(dlse); cudaFree(dDel); cudaFree(ddV); cudaFree(ddK); cudaFree(ddQ);
  return ok;
}

template <int BK,int BQ,int D,int DC1,int DC2,int WM,int WN,int QS,bool KVRES=true,int DC3P=64,int PAD=4,bool SWZ=false>
static void geom(int S, int H) {
  constexpr int TM = BK/WM, TN = BQ/WN, GM = BK/WM, GN = DC2/WN;
  constexpr int accum = 2 * (D/DC2) * (GM/16) * (GN/16) * 8;      // dV+dK, float на поток
  printf("  %dx%-2d плитка %dx%-3d %-6s накоп %3d fl | ", WM, WN, TM, TN, SWZ ? "СВИЗЛ" : "PAD+4", accum); (void)0; if(0) printf("%d%d%f",
         WM, WN, WM*WN, TM, TN, (double)(TM+TN)/((double)TM*TN), accum);
  g_warps = WM * WN;
  run_case<BK,BQ,D,DC1,DC2,WM,WN,true,KVRES,false,DC3P,false,true,QS,PAD,SWZ>(S,S,true,H);
}
int main() {
  CK(cudaSetDevice(0)); srand(5);
  geom<32,128,256,32,128,1,8,2,true,64,4,false>(2048,8);
  geom<32,128,256,32,128,1,8,2,true,64,4,true >(2048,8);
  return 0;
}
