// Рукописный forward-мейнлуп Volta (sm_70), блочный уровень.
#include <cmath>
// Мишень (docs/SM70_KERNEL_PLAYBOOK.md §25.5): у cutlass на одну HMMA приходится 5-6 целочисленных
// инструкций, тензорная работа — 6.7% статического потока. Здесь адрес считается ОДИН раз на плитку,
// перезаливка глобальная->smem векторная (16 Б), Q лежит в smem на весь блок запросов.
//
// Редукция онлайн-софтмакса ПО СТРОКЕ выведена из карты аккумулятора:
//   acc_row(lane,reg) = (lane&1) | ((lane&16)>>2) | (lane&8) | 2*((reg>>1)&1)
//   acc_col(lane,reg) = (lane&2) | ((lane&4)<<1) | (reg&1) | 4*((reg>>2)&1)
// Строка зависит от полос {0,3,4}; свободны биты полосы {1,2} -> одну строку держат РОВНО 4 полосы,
// и полная редукция строки внутри 16-столбцового блока = два __shfl_xor с масками 2 и 4.
// Каждый поток держит 2 строки (смещение 0 и 2) по 4 столбца.
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <math_constants.h>
#include "volta_warp_mma.h"
using namespace fa2_sm70;
#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

template <int BQ, int BK, int D, int WM, int WN, bool CAUSAL, int DC = D, bool SWZ = false, int PIPE = 0, bool QREG = false, int MINB = 0, int QPRE = 0, int REV = 0, int PADH = 0, int WIDE = 0, int PS = 0>
// [КОМПИЛЯТОР] Без второго аргумента __launch_bounds__ nvcc САМ выбирает цель по блокам на SM. Освободив
// 26 регистров (гипотеза 1), он ужался до 128 регистров ради ЧЕТЫРЁХ блоков и добавил 24 Б разлива —
// время 0.189 -> 0.202. Цель задаём мы: MINB=3 при 128 потоках даёт 65536/(128*3) = 170 регистров.
__global__ __launch_bounds__(WM * WN * 32, MINB ? MINB : 1) void k_fwd(const __half* __restrict__ Q,
                                                      const __half* __restrict__ Kk,
                                                      const __half* __restrict__ Vv,
                                                      __half* __restrict__ O, float* __restrict__ Lse,
                                                      int Sq, int Sk, float qk_scale) {
  constexpr int kWarps = WM * WN, PAD = PADH ? PADH : 4;
  // [SMEM] K подаётся кусками по d: буфер BK x DC вместо BK x D. Профиль показал упор в 1 блок/SM
  // (smem 59.4 КБ), а инструкций после рукописного мейнлупа стало в 7 раз меньше — прятать задержку
  // нечем, и занятость СТАЛА рычагом (в отличие от cutlass, где лишние варпы вредят). Порог двух
  // блоков — 48 КБ. Цена: D/DC дополнительных барьеров, у нас их было 5 против ~20 у cutlass.
  // [ПЛОТНОСТЬ, шаг 2] При SWZ строка Q без дополнения (степень двойки) и с XOR-перемешиванием:
  // адрес кратен 16 Б, и ОДНА LDS.128 покрывает ДВА шага k. В backward это проиграло регистрам
  // (255 при разливе), здесь бюджет есть: 123 регистра, разлив ноль (docs §16, §27).
  constexpr int LDQ = SWZ ? D : (D + PAD), LDP = BK + PAD, LDK = DC + PAD, LDV = D + PAD;
  constexpr int SM = BQ / WM, SN = BK / WN;     // плитка варпа для S
  constexpr int OM = BQ / WM, ON = D / WN;      // плитка варпа для O
  constexpr int SMB = SM / 16, SNB = SN / 16;   // блоков 16x16 в плитке S
  static_assert(SM % 16 == 0 && SN % 16 == 0 && OM % 16 == 0 && ON % 16 == 0, "плитки кратны 16");
  static_assert(OM == SM, "O и S делят строки: одна и та же сетка варпов по m");

  extern __shared__ __half smem[];
  __half* sQ = smem;                    // [BQ][LDQ] резидентно
  __half* sP = sQ + BQ * LDQ;           // [BQ][LDP] — P между двумя gemm
  // [ГЛУБИНА 2] при PIPE==2 K/V двойной буфер: пока считается плитка i, в полёте загрузки i+2,
  // а плитка i+1 уже лежит в smem. Цена — вторая пара буферов; занятость измеренно не рычаг (§25,§27).
  // ВНИМАНИЕ: PIPE==2 (глубина 2) НЕ РАБОТАЕТ (relL2 nan) И измеренно МЕДЛЕННЕЕ глубины 1
  // (0.219 против 0.189; 0.239 против 0.205): удвоение smem стоит дороже дополнительного перекрытия.
  // Оставлено только как след эксперимента — НЕ включать без переделки.
  constexpr int kBuf = (PIPE == 2) ? 2 : 1;
  __half* sK = sP + BQ * LDP;                        // [kBuf][BK][LDK]
  __half* sV = sK + kBuf * BK * LDK;                 // [kBuf][BK][LDV]
  // [ГОНКА] Один буфер использовался ДВАЖДЫ за итерацию: сначала под максимумы, затем под суммы —
  // а барьера между ЧТЕНИЕМ максимумов и ЗАПИСЬЮ сумм не было. Ушедший вперёд варп затирал максимумы,
  // которые отставший ещё не прочитал. Проявлялось ТОЛЬКО при 2 блоках на SM (S=2048 И H=4): там
  // планировщик разводит варпы сильнее. Два раздельных буфера дешевле лишнего барьера (512 Б).
  __shared__ float sRedM[BQ * WN];       // максимумы по строке
  __shared__ float sRedS[BQ * WN];       // суммы по строке

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  // [ВОЛНЫ] При причинной маске блок q0=0 считает ОДНУ плитку ключей, блок q0=Sq-BQ -- все Sq/BK.
  // Диспетчер раздаёт блоки по возрастанию blockIdx, поэтому в ОГРЫЗОК последней волны попадают самые
  // ТЯЖЁЛЫЕ блоки: 256 блоков на 240 слотов -> хвост из 16 блоков с максимальной работой на пустой
  // машине (эфф волн 53%). REV=1 разворачивает порядок: тяжёлые уходят в первую полную волну,
  // в огрызке остаются самые лёгкие.
  const int q0 = (REV ? (int)(gridDim.x - 1 - blockIdx.x) : (int)blockIdx.x) * BQ;
  {
    const long h = blockIdx.y;
    Q += h * (long)Sq * D; Kk += h * (long)Sk * D; Vv += h * (long)Sk * D;
    O += h * (long)Sq * D; Lse += h * (long)Sq;
  }
#pragma unroll
  for (int i = tid; i < BQ * D / 8; i += kWarps * 32) {          // Q один раз, 16 Б за доступ
    const int r = i / (D / 8), c = (i % (D / 8)) * 8;
    const uint4 g = *reinterpret_cast<const uint4*>(&Q[(long)(q0 + r) * D + c]);
    const int cs = SWZ ? ((((c >> 3) ^ (r & 15)) << 3) | (c & 7)) : c;   // перемешивание вместо дополнения
    __half* d = &sQ[r * LDQ + cs];
    *reinterpret_cast<uint2*>(d) = make_uint2(g.x, g.y);
    *reinterpret_cast<uint2*>(d + 4) = make_uint2(g.z, g.w);
  }

  WarpMma<OM / 16, ON / 16, BLayout::kNMajor> o; o.clear();      // O += P*V, V это [k][n] -> n-мажорный
  // Онлайн-состояние: у потока 2 строки на каждый m-блок плитки S.
  float m_run[SMB][2], l_run[SMB][2];
#pragma unroll
  for (int i = 0; i < SMB; ++i) { m_run[i][0] = m_run[i][1] = -CUDART_INF_F; l_run[i][0] = l_run[i][1] = 0.f; }

  // Конвейер грузит K/V ЦЕЛИКОМ по d (иначе нечего держать в регистрах между итерациями), поэтому
  // он несовместим с подачей K кусками. Раньше сочетание молча портило память: буфер на DC+PAD,
  // а запись на D. Ловим на этапе компиляции.
  static_assert(!(PIPE == 1 || PIPE == 2) || DC == D, "конвейер по kb требует DC == D");
  static_assert(PIPE != 3 || (BK * DC / 8) % (WM * WN * 32) == 0, "кусок должен делиться на нити нацело");
  // XOR-свизл меняет РАСКЛАДКУ sQ, а пути предзагрузки (QREG/QPRE) и конвейера читают её обычной
  // адресацией. Сочетание молча даёт неверный результат (relL2 1e-01). Третий случай подряд, когда
  // несовместимость шаблонных параметров проявляется как «геометрия не работает» — запрещаем статически.
  // [11.09] свизл Q (в shared) совместим с конвейером K (PIPE): они про разные операнды.
  // Несовместим лишь с Q-в-регистрах (QREG/QPRE).
  static_assert(!SWZ || (!QREG && !QPRE), "свизл несовместим с предзагрузкой Q в регистры");
  constexpr int kStage = PIPE ? (BK * D / 8) / (WM * WN * 32) : 1;
  // [ДОЗИРОВАННЫЙ КОНВЕЙЕР] Конвейер был «всё или ничего»: либо ВСЕ kStage векторов плитки идут через
  // регистры, либо ни один. Замер налога: BQ=32/4 варпа 124 -> 168 (+44), BQ=16/2 варпа 124 -> 234
  // (+110) -- налог обратно пропорционален числу нитей, потому что на нить приходится (BK*D/8)/нитей
  // векторов. Но перекрытие не обязано быть полным: PS векторов ведём через регистры (LDG рано),
  // остальные грузим и кладём на месте в поздней записи. Освободившиеся регистры идут в QPRE --
  // единственный измеренно работающий рычаг (+16%), который упирался ровно в регистры (обрыв 28->32).
  constexpr int kPS = (PS && PS < kStage) ? PS : kStage;
  uint4 stK[kPS], stV[kPS];
  // [ПРЯМАЯ АДРЕСАЦИЯ] SASS показал IMAD 185 из 1384 инструкций (13.4%, больше всего после HMMA) —
  // это 64-битное умножение `(long)(kb + r) * D` на КАЖДОЙ итерации загрузки. Считаем базу ОДИН раз
  // и двигаем её на константу BK*D за итерацию: ни одного умножения в цикле.
  const __half* pKs[kStage]; const __half* pVs[kStage];
  if constexpr (PIPE == 1 || PIPE == 2) {
#pragma unroll
    for (int u = 0; u < kStage; ++u) {
      const int i = tid + u * kWarps * 32;
      const int r = i / (D / 8), c = (i % (D / 8)) * 8;
      pKs[u] = Kk + (long)r * D + c;      // база: считается ОДИН раз на поток
      pVs[u] = Vv + (long)r * D + c;
    }
  }
  const int kEnd = CAUSAL ? ((q0 + BQ < Sk) ? q0 + BQ : Sk) : Sk;
  if constexpr (PIPE == 1 || PIPE == 2) {      // пролог: первая плитка ключей в smem (буфер 0)
#pragma unroll
    for (int i = tid; i < BK * D / 8; i += kWarps * 32) {
      const int r = i / (D / 8), c = (i % (D / 8)) * 8;
      const uint4 gk = *reinterpret_cast<const uint4*>(&Kk[(long)r * D + c]);
      const uint4 gv = *reinterpret_cast<const uint4*>(&Vv[(long)r * D + c]);   // пролог — вне цикла
      __half* pk = &sK[r * LDK + c]; __half* pv = &sV[r * LDV + c];
      *reinterpret_cast<uint2*>(pk) = make_uint2(gk.x, gk.y);
      *reinterpret_cast<uint2*>(pk + 4) = make_uint2(gk.z, gk.w);
      *reinterpret_cast<uint2*>(pv) = make_uint2(gv.x, gv.y);
      *reinterpret_cast<uint2*>(pv + 4) = make_uint2(gv.z, gv.w);
    }
  }
  // [Q В РЕГИСТРАХ] фрагменты A грузятся ОДИН раз на весь блок запросов
  WarpMma<SMB, SNB, BLayout::kKMajor> sTpl;
  uint32_t qfrag[QREG ? (D / 4) : (QPRE ? QPRE : 1)][SMB][2];
  if constexpr (QREG) { __syncthreads(); sTpl.template preload_a<D>(qfrag, sQ, LDQ, wm * SM, lane); }
  else if constexpr (QPRE) { __syncthreads(); sTpl.template preload_n<QPRE>(qfrag, sQ, LDQ, wm * SM, lane); }
  int buf = 0;
  for (int kb = 0; kb < kEnd; kb += BK) {
    __syncthreads();
    __half* sKc = sK + (PIPE == 2 ? buf : 0) * BK * LDK;          // текущий буфер
    __half* sVc = sV + (PIPE == 2 ? buf : 0) * BK * LDV;
    __half* sKn = sK + (PIPE == 2 ? (buf ^ 1) : 0) * BK * LDK;    // куда писать следующую
    __half* sVn = sV + (PIPE == 2 ? (buf ^ 1) : 0) * BK * LDV;
    if constexpr (PIPE == 0 || PIPE == 3) {    // PIPE==3 тоже грузит V здесь: конвейер у него по КУСКАМ K
#pragma unroll
      for (int i = tid; i < BK * D / 8; i += kWarps * 32) {      // V целиком, векторно
        const int r = i / (D / 8), c = (i % (D / 8)) * 8;
        const uint4 gv = *reinterpret_cast<const uint4*>(&Vv[(long)(kb + r) * D + c]);
        __half* pv = &sV[r * LDV + c];
        *reinterpret_cast<uint2*>(pv) = make_uint2(gv.x, gv.y);
        *reinterpret_cast<uint2*>(pv + 4) = make_uint2(gv.z, gv.w);
      }
    }
    WarpMma<SMB, SNB, BLayout::kKMajor> s; s.clear();            // S = Q*K^T, K это [n][k] -> k-мажорный
    if constexpr (PIPE == 1 || PIPE == 2) {
      // [ДВЕ ЦЕПОЧКИ] Загрузки СЛЕДУЮЩЕЙ плитки ключей выдаются ЗДЕСЬ, в регистры, до всякого счёта
      // по текущей. Запись в smem — в конце итерации. Это разнесение LDG и STS: в backward именно
      // оно дало +20-46%, тогда как размазывание выдачи по слотам — ноль (docs §13).
      if constexpr (PIPE == 2) {
        // Порядок обязателен: СНАЧАЛА выгрузить плитку i+1 из регистров в ДРУГОЙ буфер (со счётом по
        // текущему она не конфликтует, барьер не нужен), и только ПОТОМ выдавать загрузки плитки i+2 —
        // иначе i+2 затирает ещё не записанную i+1. Первая сборка падала именно на этом (relL2 nan).
        if (kb + BK < kEnd) {
#pragma unroll
          for (int u = 0; u < kStage; ++u) {
            const int i = tid + u * kWarps * 32;
            const int r = i / (D / 8), c = (i % (D / 8)) * 8;
            __half* pk = &sKn[r * LDK + c]; __half* pv = &sVn[r * LDV + c];
            *reinterpret_cast<uint2*>(pk) = make_uint2(stK[u].x, stK[u].y);
            *reinterpret_cast<uint2*>(pk + 4) = make_uint2(stK[u].z, stK[u].w);
            *reinterpret_cast<uint2*>(pv) = make_uint2(stV[u].x, stV[u].y);
            *reinterpret_cast<uint2*>(pv + 4) = make_uint2(stV[u].z, stV[u].w);
          }
        }
      }
      const int nb = kb + (PIPE == 2 ? 2 * BK : BK);
      if (nb < kEnd) {
#pragma unroll
        for (int u = 0; u < kPS; ++u) {             // адрес = база + СДВИГ, без умножения
          stK[u] = *reinterpret_cast<const uint4*>(pKs[u] + (long)nb * D);
          stV[u] = *reinterpret_cast<const uint4*>(pVs[u] + (long)nb * D);
        }
      }
      if constexpr (QREG)      s.template accumulate_pre<D>(qfrag, sKc, LDK, wn * SN, lane);
      else if constexpr (QPRE) s.template accumulate_part<D, QPRE>(qfrag, sQ, LDQ, wm * SM, sKc, LDK, wn * SN, lane);
      else                     s.template accumulate<D>(sQ, LDQ, wm * SM, sKc, LDK, wn * SN, lane);
    } else if constexpr (PIPE == 3) {
      // [softDMA НА ОСИ КУСКОВ] Прежний цикл кусков чередовал строго: sync -> LDG+STS -> sync -> mma.
      // Перекрытия нет вообще, поэтому 2 куска и 4 куска давали ОДНО время (0.265/0.266): всё теряется
      // на первом же дроблении. Здесь тот же приём, что дал +20-46% в backward (docs §13): LDG следующего
      // куска выдаётся ДО mma текущего и живёт в регистрах, STS -- поздно. Цена: kCh регистров uint4.
      constexpr int kCh = (BK * DC / 8) / (WM * WN * 32);
      uint4 stC[kCh];
#pragma unroll
      for (int u = 0; u < kCh; ++u) {                              // пролог: кусок 0 -> регистры
        const int i = tid + u * kWarps * 32, r = i / (DC / 8), c = (i % (DC / 8)) * 8;
        stC[u] = *reinterpret_cast<const uint4*>(&Kk[(long)(kb + r) * D + c]);
      }
#pragma unroll
      for (int dc = 0; dc < D; dc += DC) {
        __syncthreads();                                           // sK прочитана всеми -- можно писать
#pragma unroll
        for (int u = 0; u < kCh; ++u) {                            // STS поздно: регистры -> smem
          const int i = tid + u * kWarps * 32, r = i / (DC / 8), c = (i % (DC / 8)) * 8;
          __half* pk = &sK[r * LDK + c];
          *reinterpret_cast<uint2*>(pk) = make_uint2(stC[u].x, stC[u].y);
          *reinterpret_cast<uint2*>(pk + 4) = make_uint2(stC[u].z, stC[u].w);
        }
        __syncthreads();
        if (dc + DC < D) {                                         // LDG рано: следующий кусок в полёте
#pragma unroll
          for (int u = 0; u < kCh; ++u) {
            const int i = tid + u * kWarps * 32, r = i / (DC / 8), c = (i % (DC / 8)) * 8;
            stC[u] = *reinterpret_cast<const uint4*>(&Kk[(long)(kb + r) * D + dc + DC + c]);
          }
        }
        // QPRE здесь НЕ применим: qfrag держит k-шаги 0..QPRE-1 ПОЛНОГО D, а не текущего куска dc.
        if constexpr (SWZ)
          s.template accumulate_swzA_dc<DC>(sQ, LDQ, wm * SM, dc, sK, LDK, wn * SN, lane);
        else
          s.template accumulate<DC>(sQ + dc, LDQ, wm * SM, sK, LDK, wn * SN, lane);
      }
    } else {
#pragma unroll
      for (int dc = 0; dc < D; dc += DC) {                        // K кусками по d
        __syncthreads();
#pragma unroll
        for (int i = tid; i < BK * DC / 8; i += kWarps * 32) {
          const int r = i / (DC / 8), c = (i % (DC / 8)) * 8;
          const uint4 gk = *reinterpret_cast<const uint4*>(&Kk[(long)(kb + r) * D + dc + c]);
          __half* pk = &sK[r * LDK + c];
          *reinterpret_cast<uint2*>(pk) = make_uint2(gk.x, gk.y);
          *reinterpret_cast<uint2*>(pk + 4) = make_uint2(gk.z, gk.w);
        }
        __syncthreads();
        if constexpr (SWZ)
          s.template accumulate_swzA_dc<DC>(sQ, LDQ, wm * SM, dc, sK, LDK, wn * SN, lane);
        else
          s.template accumulate<DC>(sQ + dc, LDQ, wm * SM, sK, LDK, wn * SN, lane);
      }
    }

    const int r0 = wm * SM, c0 = wn * SN;
    const int rowbase = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);   // строка внутри 16-блока
    float mloc[SMB][2], lloc[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i) { mloc[i][0] = mloc[i][1] = -CUDART_INF_F; lloc[i][0] = lloc[i][1] = 0.f; }

    s.visit(lane, [&](float& v, int r, int c) {                   // масштаб и причинная маска
      v *= qk_scale;   // qk_scale уже = scale*log2(e), см. запуск: экспонента считается exp2
      if (CAUSAL && (q0 + r0 + r) < (kb + c0 + c)) v = -CUDART_INF_F;
      const int mb = r >> 4, sub = ((r & 15) - rowbase) >> 1;      // 0 или 1 (смещение 0 / 2)
      if (v > mloc[mb][sub]) mloc[mb][sub] = v;
    });
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {                               // строку держат 4 полосы: биты 1 и 2
        float x = mloc[i][t];
        float y = __shfl_xor_sync(0xffffffffu, x, 2); x = x > y ? x : y;
        y = __shfl_xor_sync(0xffffffffu, x, 4);       x = x > y ? x : y;
        mloc[i][t] = x;
        if ((lane & 6) == 0) sRedM[(wm * SM + i * 16 + rowbase + 2 * t) * WN + wn] = x;
      }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {                               // и по варпам вдоль n
        float x = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < WN; ++w) { float y = sRedM[(wm * SM + i * 16 + rowbase + 2 * t) * WN + w]; x = x > y ? x : y; }
        mloc[i][t] = x;
      }

    // [БЕЗ ЛИШНЕГО НАКОПИТЕЛЯ] m_new и масштабы известны УЖЕ ЗДЕСЬ: они зависят от максимумов, а не
    // от сумм. Значит cn можно вложить в саму P при записи и копить P*V ПРЯМО в o — отдельный
    // накопитель add (32 регистра) не нужен. Освобождённые регистры уходят на рычаги, которые до сих
    // пор упирались в бюджет (см. §33: пять из шести попыток похоронили именно регистры).
    float corr2[SMB][2], cnew[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const float mo = m_run[i][t], mn = (mo > mloc[i][t]) ? mo : mloc[i][t];
        corr2[i][t] = (mo == -CUDART_INF_F) ? 0.f : exp2f(mo - mn);
        cnew[i][t]  = (mloc[i][t] == -CUDART_INF_F) ? 0.f : exp2f(mloc[i][t] - mn);
      }
    // [ПЛОТНОСТЬ] P пишется в smem ПАРАМИ по 4 Б, а не по 2. Карта аккумулятора даёт это даром:
    //   acc_col(lane,reg) = base + (reg&1) + 4*((reg>>2)&1),  acc_row = rowbase + 2*((reg>>1)&1)
    // значит регистры (0,1), (2,3), (4,5), (6,7) — это ПАРЫ СОСЕДНИХ СТОЛБЦОВ одной строки.
    // Вдвое меньше инструкций записи; тот же приём, что в backward дал x1.265 на перезаливке.
    {
      const int cbase = (lane & 2) | ((lane & 4) << 1);
#pragma unroll
      for (int i = 0; i < SMB; ++i)
#pragma unroll
        for (int j = 0; j < SNB; ++j)
#pragma unroll
          for (int pr = 0; pr < 4; ++pr) {                       // пары регистров
            const int rg = (pr & 1) ? (4 + ((pr >> 1) << 1)) : ((pr >> 1) << 1);
            const int reg0 = rg, reg1 = rg + 1;
            const int row = i * 16 + rowbase + 2 * ((reg0 >> 1) & 1);
            const int col = j * 16 + cbase + 4 * ((reg0 >> 2) & 1);
            const int mb = i, sub = (reg0 >> 1) & 1;   // константы компиляции
            const float mm = mloc[mb][sub];
            const float v0 = s.acc[i][j][reg0], v1 = s.acc[i][j][reg1];
            const float p0 = (v0 == -CUDART_INF_F || mm == -CUDART_INF_F) ? 0.f : exp2f(v0 - mm);
            const float p1 = (v1 == -CUDART_INF_F || mm == -CUDART_INF_F) ? 0.f : exp2f(v1 - mm);
            lloc[mb][sub] += p0 + p1;                       // сумма — по НЕмасштабированной p
            const float sc2 = cnew[mb][sub];                  // масштаб нового вклада вложен в P
            *reinterpret_cast<__half2*>(&sP[(r0 + row) * LDP + c0 + col]) = __floats2half2_rn(p0 * sc2, p1 * sc2);
          }
    }
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        float x = lloc[i][t];
        x += __shfl_xor_sync(0xffffffffu, x, 2);
        x += __shfl_xor_sync(0xffffffffu, x, 4);
        if ((lane & 6) == 0) sRedS[(wm * SM + i * 16 + rowbase + 2 * t) * WN + wn] = x;
      }
    __syncthreads();
    float corr[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        float ss = 0.f;
#pragma unroll
        for (int w = 0; w < WN; ++w) ss += sRedS[(wm * SM + i * 16 + rowbase + 2 * t) * WN + w];
        const float mo = m_run[i][t], mn = (mo > mloc[i][t]) ? mo : mloc[i][t];
        l_run[i][t] = l_run[i][t] * corr2[i][t] + ss * cnew[i][t];
        m_run[i][t] = mn; corr[i][t] = corr2[i][t];
      }
    __syncthreads();                                              // sP готова для всех варпов
    {                                    // индексы поправки — константы, а не runtime
      constexpr int OMB = OM / 16, ONB = ON / 16;
#pragma unroll
      for (int i = 0; i < OMB; ++i)
#pragma unroll
        for (int j = 0; j < ONB; ++j)
#pragma unroll
          for (int rg = 0; rg < 8; ++rg) o.acc[i][j][rg] *= corr[i][(rg >> 1) & 1];
    }
    o.template accumulate<BK>(sP, LDP, wm * OM, sVc, LDV, wn * ON, lane);   // ПРЯМО в o

    if constexpr (PIPE == 1) {                  // ПОЗДНЯЯ запись (глубина 1): счёт уже позади
      const int nb = kb + BK;
      if (nb < kEnd) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < kStage; ++u) {
          const int i = tid + u * kWarps * 32;
          const int r = i / (D / 8), c = (i % (D / 8)) * 8;
          __half* pk = &sKn[r * LDK + c]; __half* pv = &sVn[r * LDV + c];
          // u < kPS -- вектор ждал в регистрах (LDG выдан до счёта); иначе грузим прямо здесь
          const uint4 gk = (u < kPS) ? stK[u] : *reinterpret_cast<const uint4*>(pKs[u] + (long)nb * D);
          const uint4 gv = (u < kPS) ? stV[u] : *reinterpret_cast<const uint4*>(pVs[u] + (long)nb * D);
          *reinterpret_cast<uint2*>(pk) = make_uint2(gk.x, gk.y);
          *reinterpret_cast<uint2*>(pk + 4) = make_uint2(gk.z, gk.w);
          *reinterpret_cast<uint2*>(pv) = make_uint2(gv.x, gv.y);
          *reinterpret_cast<uint2*>(pv + 4) = make_uint2(gv.z, gv.w);
        }
      }
    }
    if constexpr (PIPE == 2) buf ^= 1;
  }
  // ---- эпилог: O = acc / l, LSE = m + log(l). Строки принадлежат этому варпу, обмена не нужно.
  {
    const int r0 = wm * OM;
    const int rowbase = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);
    float inv[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) inv[i][t] = (l_run[i][t] > 0.f) ? __frcp_rn(l_run[i][t]) : 0.f;
    {
      constexpr int OMB = OM / 16, ONB = ON / 16;
      const int cbase0 = (lane & 2) | ((lane & 4) << 1);
#pragma unroll
      for (int i = 0; i < OMB; ++i)
#pragma unroll
        for (int j = 0; j < ONB; ++j)
#pragma unroll
          for (int rg = 0; rg < 8; ++rg) {
            const int sub = (rg >> 1) & 1;
            const int row = i * 16 + rowbase + 2 * sub;
            const int col = j * 16 + cbase0 + (rg & 1) + 4 * ((rg >> 2) & 1);
            O[(long)(q0 + r0 + row) * D + wn * ON + col] = __float2half(o.acc[i][j][rg] * inv[i][sub]);
          }
    }
    if ((lane & 6) == 0 && wn == 0) {                       // LSE пишет один варп на строку
#pragma unroll
      for (int i = 0; i < SMB; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          const int row = r0 + i * 16 + rowbase + 2 * t;
          Lse[q0 + row] = (l_run[i][t] > 0.f) ? (m_run[i][t] * 0.6931471805599453f + __logf(l_run[i][t])) : -CUDART_INF_F;
        }
    }
  }
}


// ---------------------------------------------------------------------------------------------
template <int BQ, int BK, int D, int WM, int WN, bool CAUSAL, int DC = D, bool SWZ = false, int PIPE = 0, bool QREG = false, int MINB = 0, int QPRE = 0, int PADB = 0, int REV = 0, int PADH = 0, int WIDE = 0, int PS = 0>
static bool run(int Sq, int Sk, int H, bool bench) {
  std::vector<__half> hQ((size_t)Sq*D*H), hK((size_t)Sk*D*H), hV((size_t)Sk*D*H);
  for (auto& x : hQ) x = __float2half((rand()/(float)RAND_MAX - .5f));
  for (auto& x : hK) x = __float2half((rand()/(float)RAND_MAX - .5f));
  for (auto& x : hV) x = __float2half((rand()/(float)RAND_MAX - .5f));
  __half *dQ,*dK,*dV,*dO; float* dL;
  CK(cudaMalloc(&dQ,hQ.size()*2)); CK(cudaMalloc(&dK,hK.size()*2)); CK(cudaMalloc(&dV,hV.size()*2));
  CK(cudaMalloc(&dO,(size_t)Sq*D*H*2)); CK(cudaMalloc(&dL,(size_t)Sq*H*4));
  CK(cudaMemcpy(dQ,hQ.data(),hQ.size()*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK,hK.data(),hK.size()*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV,hV.data(),hV.size()*2,cudaMemcpyHostToDevice));
  constexpr int PAD=PADH?PADH:4, LDQ=D+PAD, LDP=BK+PAD;
  constexpr int kBuf=(PIPE==2)?2:1;
  const size_t sh=(size_t)(BQ*(SWZ?D:LDQ) + BQ*LDP + kBuf*BK*(DC+PAD) + kBuf*BK*(D+PAD))*sizeof(__half) + PADB;
  if (sh > 96*1024) { printf("  BQ=%-3d BK=%-3d D=%-4d %dx%-2d ПРОПУСК smem %zu\n",BQ,BK,D,WM,WN,sh); return true; }
  auto kern = k_fwd<BQ,BK,D,WM,WN,CAUSAL,DC,SWZ,PIPE,QREG,MINB,QPRE,REV,PADH,WIDE,PS>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  const float sc_ref = 1.f/sqrtf((float)D);                        // эталон — в натуральных единицах
  const float sc = sc_ref * 1.4426950408889634f;                   // ядру — масштаб СЛИТ с log2(e)
  auto go=[&]{ kern<<<dim3(Sq/BQ,H),WM*WN*32,sh>>>(dQ,dK,dV,dO,dL,Sq,Sk,sc); };
  go(); CK(cudaDeviceSynchronize());
  std::vector<__half> hO((size_t)Sq*D);
  CK(cudaMemcpy(hO.data(),dO,(size_t)Sq*D*2,cudaMemcpyDeviceToHost));
  double num=0,den=0;
  for (int q=0;q<Sq;++q){
    std::vector<double> sv(Sk); double mx=-1e30;
    int kmax = CAUSAL ? q+1 : Sk;
    for (int k=0;k<kmax;++k){ double a=0; for(int t=0;t<D;++t) a+=(double)__half2float(hQ[(size_t)q*D+t])*__half2float(hK[(size_t)k*D+t]);
      sv[k]=a*sc_ref; if(sv[k]>mx) mx=sv[k]; }
    double l=0; for(int k=0;k<kmax;++k){ sv[k]=exp(sv[k]-mx); l+=sv[k]; }
    for(int t=0;t<D;++t){ double a=0; for(int k=0;k<kmax;++k) a+=sv[k]*__half2float(hV[(size_t)k*D+t]);
      a/=l; double e=a-__half2float(hO[(size_t)q*D+t]); num+=e*e; den+=a*a; }
  }
  const double rel=sqrt(num/(den+1e-30));
  double ms=0;
  if (bench){ for(int i=0;i<5;++i) go(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for(int i=0;i<20;++i) go(); cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float t; cudaEventElapsedTime(&t,a,b); ms=t/20; }
  cudaFuncAttributes at; cudaFuncGetAttributes(&at,(const void*)kern);
  int occ=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,(const void*)kern,WM*WN*32,sh);
  int nsm=0; cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0);
  const int nblk=(Sq/BQ)*H, slots=occ*nsm;
  const double waves=(double)nblk/slots, eff=waves/ceil(waves);       // потери на квантовании волн
  printf("      [занятость] ЧЕСТНО %d блок/SM (мой счёт по smem: %d) | SM=%d слотов=%d блоков=%d волн=%.2f эфф=%.0f%% | варпов/SM=%d\n",
         occ, (int)(98304/(sh+2048)), nsm, slots, nblk, waves, 100*eff, occ*WM*WN);
  printf("  BQ=%-3d BK=%-3d %dx%-2d плитка %dx%-2d загр/mma %.2f %-5s%-6s мб=%d блок/SM=%d | relL2 %.1e | рег %3d разлив %4d smem %5zu | %.3f мс %s\n",
         BQ,BK,WM,WN,BQ/WM,BK/WN,(double)((BQ/WM)/16+(BK/WN)/16)/(((BQ/WM)/16)*((BK/WN)/16)),PIPE?"КОНВ":(SWZ?"СВИЗЛ":"пад"),WIDE?"+ШИР128":(QREG?"+Qрег":(QPRE?(REV?"+Qч+РЕВ":"+Qчаст"):(REV?"+РЕВ":""))),MINB,(int)(98304/(sh+2048)),rel,at.numRegs,(int)at.localSizeBytes,sh,ms, rel<3e-3?"ОК":"ОТКАЗ");
  cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dL);
  return rel<3e-3;
}
int main(){ CK(cudaSetDevice(0)); srand(7);
  printf("=== динамические индексы убраны: массивы должны остаться в регистрах\n");
  bool ok=true;
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24>(512,512,1,false); ok &= run<32,32,128,2,2,true,128,false,1,false,3,24>(2048,2048,4,true);
  ok &= run<64,64,128,2,2,true,128,false,0,false,1,0>(512,512,1,false);  ok &= run<64,64,128,2,2,true,128,false,0,false,1,0>(2048,2048,4,true);
  ok &= run<64,64,128,2,2,true,128,false,1,false,1,0>(2048,2048,4,true);
  ok &= run<64,64,128,4,2,true,128,false,1,false,1,8>(2048,2048,4,true);
  printf("=== 16 ВАРПОВ В БЛОКЕ: плитка 64x64, варп 16x16, 512 нитей (потолок 128 рег)\n");
  ok &= run<64,64,128,4,4,true,128,false,0,false,0,0>(512,512,1,false);  ok &= run<64,64,128,4,4,true,128,false,0,false,0,0>(2048,2048,4,true);
  ok &= run<64,64,128,4,4,true,128,false,0,false,1,0>(2048,2048,4,true);
  ok &= run<64,64,128,4,4,true,128,false,1,false,1,0>(2048,2048,4,true);
  ok &= run<64,64,128,4,4,true,128,false,1,false,1,8>(2048,2048,4,true);
  printf("=== ГИПОТЕЗА: 16 варпов/SM в ДВУХ независимых блоках (64x32, варп 16x16)\n");
  ok &= run<64,32,128,4,2,true,128,false,0,false,0,0>(512,512,1,false);  ok &= run<64,32,128,4,2,true,128,false,0,false,0,0>(2048,2048,4,true);
  ok &= run<64,32,128,4,2,true,128,false,0,false,2,0>(2048,2048,4,true);
  ok &= run<64,32,128,4,2,true,128,false,1,false,2,0>(2048,2048,4,true);
  ok &= run<64,32,128,4,2,true,128,false,1,false,2,8>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,2,24>(2048,2048,4,true);
  printf("=== ФАЛЬСИФИКАТОР: чемпион, принудительно 1 блок/SM (smem раздута балластом)\n");
  ok &= run<32,32,128,2,2,true,128,false,1,false,1,24,40000>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,1,24,16000>(2048,2048,4,true);
  printf("=== ЦЕЛЬ 4 БЛОКА/SM: резать smem подачей K кусками (потолок 128 рег, smem<=24576)\n");
  ok &= run<32,32,128,2,2,true,64,false,0,false,4,24>(512,512,1,false);  ok &= run<32,32,128,2,2,true,64,false,0,false,4,24>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,64,false,0,false,4,0>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,32,false,0,false,4,24>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,32,false,0,false,4,0>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,64,false,0,false,3,24>(2048,2048,4,true);
  printf("=== КОНТРОЛЬ: развести две причины -- конвейер отдельно, куски K отдельно\n");
  ok &= run<32,32,128,2,2,true,128,false,0,false,3,24>(2048,2048,4,true);   // 3 блока, БЕЗ конвейера, DC=D
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24>(2048,2048,4,true);   // 3 блока, С конвейером, DC=D
  printf("=== ДОЗИРОВАННЫЙ КОНВЕЙЕР: PS векторов через регистры, остаток на месте -> регистры в QPRE\n");
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,28,0,1,0,0,3>(512,512,1,false);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,28,0,1,0,0,0>(4096,4096,4,true);  // доза 4/4 (как было)
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,28,0,1,0,0,3>(4096,4096,4,true);  // доза 3/4
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,28,0,1,0,0,2>(4096,4096,4,true);  // доза 2/4
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,32,0,1,0,0,3>(4096,4096,4,true);  // доза 3/4 + ВЕСЬ Q
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,32,0,1,0,0,2>(4096,4096,4,true);  // доза 2/4 + ВЕСЬ Q
  ok &= run<32,32,128,2,2,true,128,false,1,true, 3,0, 0,1,0,0,2>(4096,4096,4,true);  // доза 2/4 + QREG
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,30,0,1,0,0,3>(4096,4096,4,true);
  printf("=== МЕХАНИЗМ: регистровый налог конвейера = (BK*D/8)/нитей на нить\n");
  ok &= run<16,32,128,1,2,true,128,false,1,true,3,0,0,1>(4096,4096,4,true);   // 2 варпа, конвейер
  ok &= run<16,32,128,1,2,true,128,false,0,true,3,0,0,1>(4096,4096,4,true);   // 2 варпа, БЕЗ конвейера
  ok &= run<32,32,128,2,2,true,128,false,1,true,3,0,0,1>(4096,4096,4,true);   // 4 варпа, конвейер
  ok &= run<32,32,128,2,2,true,128,false,0,true,3,0,0,1>(4096,4096,4,true);   // 4 варпа, БЕЗ конвейера
  printf("=== ИДЕЯ Q ДО КОНЦА: BQ=16 -> MB=1 -> ВЕСЬ Q влезает в 64 регистра\n");
  ok &= run<16,32,128,1,2,true,128,false,1,true, 3,0,0,1>(512,512,1,false);
  ok &= run<16,32,128,1,2,true,128,false,1,true, 3,0,0,1>(4096,4096,4,true);   // BQ=16, полный QREG
  ok &= run<16,32,128,1,2,true,128,false,1,false,3,28,0,1>(4096,4096,4,true);  // BQ=16, QPRE=28
  ok &= run<16,64,128,1,4,true,128,false,1,true, 3,0,0,1>(4096,4096,4,true);
  ok &= run<16,32,128,1,2,true,128,false,1,true, 0,0,0,1>(4096,4096,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,28,0,1>(4096,4096,4,true);  // чемпион QPRE=28
  printf("=== ТОЧКА QPRE: сколько групп k держать в регистрах (32 = весь Q)\n");
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,16,0,1>(4096,4096,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,20,0,1>(4096,4096,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,1>(4096,4096,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,28,0,1>(4096,4096,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,32,0,1>(4096,4096,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,true, 3,0, 0,1>(4096,4096,4,true);   // полный QREG
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,26,0,1>(4096,4096,4,true);
  printf("=== СИНТЕЗ: ПЛОТНЫЙ варповый тайл 32x32 (1.00 загр/mma) + упреждение по кускам -> вернуть блоки\n");
  ok &= run<64,64,128,2,2,true,32,false,3,false,2,0,0,1>(512,512,1,false);
  ok &= run<64,64,128,2,2,true,32,false,3,false,2,0,0,1>(4096,4096,4,true);   // DC=32, 2 блока?
  ok &= run<64,64,128,2,2,true,64,false,3,false,2,0,0,1>(4096,4096,4,true);   // DC=64
  ok &= run<64,64,128,2,2,true,32,false,3,false,0,0,0,1>(4096,4096,4,true);
  ok &= run<64,64,128,2,2,true,128,false,1,false,1,0,0,1>(4096,4096,4,true);  // плотный, конвейер по kb
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,1>(4096,4096,4,true); // чемпион для сверки
  printf("=== ШИРОКАЯ ЗАГРУЗКА 128 бит (PAD=8) против узкой 64 бит (PAD=4) -- цель: mio_throttle\n");
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,1,0,0>(4096,4096,4,true);  // чемпион: QPRE, узкая
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,0, 0,1,0,0>(4096,4096,4,true);  // без QPRE, узкая PAD=4
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,0, 0,1,8,0>(4096,4096,4,true);  // без QPRE, узкая PAD=8
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,0, 0,1,8,1>(512,512,1,false);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,0, 0,1,8,1>(4096,4096,4,true);  // ШИРОКАЯ PAD=8
  ok &= run<32,32,128,2,2,true,128,false,1,false,0,0, 0,1,8,1>(4096,4096,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,4,0, 0,1,8,1>(4096,4096,4,true);
  printf("=== РАЗВОРОТ ПОРЯДКА БЛОКОВ: тяжёлые в первую волну, лёгкие в огрызок\n");
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,1>(512,512,1,false);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,0>(2048,2048,4,true);   // чемпион, прямой порядок
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,1>(2048,2048,4,true);   // он же, РАЗВЁРНУТЫЙ
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,0>(4096,4096,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,1>(4096,4096,4,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,0>(1024,1024,8,true);
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24,0,1>(1024,1024,8,true);
  printf("=== ЧЕСТНАЯ ЗАНЯТОСТЬ: что компилятор ДЕЙСТВИТЕЛЬНО дал, на 80 SM\n");
  ok &= run<32,32,128,2,2,true,128,false,1,false,3,24>(2048,2048,4,true);   // чемпион
  ok &= run<32,32,128,2,2,true,128,false,1,false,2,24>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,128,false,0,false,3,24>(2048,2048,4,true);
  ok &= run<64,64,128,4,4,true,128,false,1,false,1,8>(2048,2048,4,true);
  ok &= run<64,32,128,4,2,true,128,false,1,false,2,8>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,64,false,3,false,4,0>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,32,false,3,false,4,0>(2048,2048,4,true);
  printf("=== softDMA НА ОСИ КУСКОВ (PIPE=3): 4 блока/SM И перекрытие\n");
  ok &= run<32,32,128,2,2,true,64,false,3,false,4,0>(512,512,1,false);  ok &= run<32,32,128,2,2,true,64,false,3,false,4,0>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,32,false,3,false,4,0>(512,512,1,false);  ok &= run<32,32,128,2,2,true,32,false,3,false,4,0>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,64,false,3,false,3,0>(2048,2048,4,true);
  ok &= run<32,32,128,2,2,true,32,false,3,false,0,0>(2048,2048,4,true);
  printf("%s\n", ok?"ПРОЙДЕНО":"ЕСТЬ ОТКАЗЫ"); return ok?0:1; }
