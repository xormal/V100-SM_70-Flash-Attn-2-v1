"""ОПИСЬ ПУТИ ПО ЯДРАМ: что реально шло на карте, чьё оно и где машина ПРОСТАИВАЛА.

ЗАЧЕМ ИМЕННО ТАК. Цель -- «вход | слитое ядро | выход», то есть переукладка ЛЕСЕНКИ тел, а не
ускорение отдельного тела. Основание для переукладки -- расписание: какие тела идут, сколько
занимают и СКОЛЬКО МЕЖДУ НИМИ ПУСТОТЫ. Опись по владению («наше/чужое») без времени бесполезна:
чужое тело на 0.1 % не долг, а наше на 40 % с провалом рядом -- главный ход.

ЧИТАТЬ ТАК:
  * `своё/чужое` -- доля ВРЕМЕНИ КАРТЫ, а не число ядер;
  * `простой` -- сумма разрывов между ядрами внутри окна. Это верхняя оценка того, что можно
    отыграть УКЛАДКОЙ, ничего не ускоряя;
  * `запусков` -- сколько раз тело стартовало: тысяча коротких пусков это и есть та лесенка,
    которую надо сливать.

    python tools/opis_puti.py <файл.pt.trace.json> [--топ 20]
"""
import argparse
import gzip
import json
import re
import sys
from collections import defaultdict

# Признаки владения. cuBLAS на Volta называется volta_h884gemm/volta_sgemm -- то есть начинается
# с "volta_", как и НАШИ тела. Поэтому чужое проверяется ПЕРВЫМ, иначе опись припишет нам cuBLAS.
ЧУЖИЕ = (
    (r"^volta_\w*gemm|^volta_\w*_nn|^volta_\w*_nt|^volta_\w*_tn|sgemm|h884|hgemm|^sm\d+_xmma|cutlass",
     "cuBLAS/cutlass"),
    (r"^triton_|^_triton|triton_poi|triton_red|triton_per", "Triton"),
    (r"at::native|^void at::|elementwise_kernel|vectorized_elementwise|reduce_kernel|CatArrayBatched"
     r"|indexSelect|index_elementwise|unrolled_elementwise", "torch"),
    (r"ncclDevKernel|ncclKernel|AllReduce|all_reduce", "NCCL"),
    (r"vllm::gptq|reconstruct_exllama|gemm_half_q_half", "vLLM GPTQ"),
    (r"causal_conv1d|l2norm_fwd|turbomind", "прочее чужое"),
    # Скан GDN у движка СВОЙ (Triton), наши ядра скана в декоде НЕ ПОДКЛЮЧЕНЫ -- см.
    # память "Ядра скана GDN НЕ подключены". Называем это явно, иначе оно немое.
    (r"fused_sigmoid_gating_delta_rule|fused_recurrent_gated_delta", "чужое: скан GDN"),
    (r"gemv2T_kernel|dot_kernel|splitKreduce", "cuBLAS/cutlass"),
    (r"^Memcpy|^Memset", "копии/обнуления"),
)
СВОИ = (
    (r"volta_gdn|gdn_|chunk_h|chunk_o|solve_tril|kkt", "наше: GDN"),
    (r"w12a16|w12_dequant|w8a16", "наше: веса int-N"),
    (r"pack_i12|unpack_add_i12|unpack_i12|exch", "наше: обмен int12"),
    # ИМЕНА БЕРУТСЯ ИЗ ЛЕНТЫ, А НЕ ИЗ ПАМЯТИ. `split_defer_mqa_kernel`, `reduce_defer*`,
    # `dense_gemv/dense_norm`, `gather_paged_kv*` -- НАШИ тела из split_decode.cu, но под шаблон
    # "split_decode" не подходили и падали в НЕОПОЗНАНО. Опись, теряющая собственные ядра, даёт не
    # "меньше данных", а ДРУГОЙ ОТВЕТ: 0.8 % нашего внимания читались как чужие. Проверено 16.08 на
    # ленте декода Qwen3.8 -- НЕОПОЗНАНО упало с 5.2 %.
    (r"fa2_sm70|fmha|volta_fwd|volta_bwd|split_decode|split_defer|reduce_defer|decode_paged"
     r"|flash_attn|attn_fwd|attn_bwd|reshape_and_cache|quantize_kv|gather_paged_kv|dense_gemv"
     r"|dense_norm|from_e4m3|f2e4m3", "наше: внимание/KV"),
)


def чей(имя):
    for об, метка in ЧУЖИЕ:
        if re.search(об, имя, re.I):
            return метка
    for об, метка in СВОИ:
        if re.search(об, имя, re.I):
            return метка
    return "НЕОПОЗНАНО"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("лента")
    ap.add_argument("--топ", type=int, default=20)
    a = ap.parse_args()

    # Лента приходит СЖАТОЙ (.json.gz) -- открывать по расширению, а не по надежде.
    открыть = gzip.open if a.лента.endswith(".gz") else open
    с = json.load(открыть(a.лента, "rt"))
    соб = с["traceEvents"]
    ядра = [e for e in соб
            if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_memcpy",
                                                                       "gpu_memset")]
    if not ядра:
        print("в ленте нет событий ядер"); return 2
    ядра.sort(key=lambda e: e["ts"])
    t0, t1 = ядра[0]["ts"], max(e["ts"] + e["dur"] for e in ядра)
    окно = t1 - t0

    # ПРОСТОЙ СЧИТАЕТСЯ ПО ОБЪЕДИНЕНИЮ ОТРЕЗКОВ, а не по сумме длительностей: ядра идут на
    # нескольких потоках и перекрываются, и наивная сумма даёт занятость больше 100 %.
    отр = sorted((e["ts"], e["ts"] + e["dur"]) for e in ядра)
    занято, кон = 0.0, None
    нач = None
    for s, e in отр:
        if кон is None:
            нач, кон = s, e
        elif s > кон:
            занято += кон - нач
            нач, кон = s, e
        else:
            кон = max(кон, e)
    занято += (кон - нач) if кон is not None else 0

    по_вл, по_имени = defaultdict(float), defaultdict(lambda: [0.0, 0])
    for e in ядра:
        имя = e["name"]
        по_вл[чей(имя)] += e["dur"]
        по_имени[имя][0] += e["dur"]
        по_имени[имя][1] += 1
    сумма = sum(по_вл.values())

    print(f"окно {окно/1000:.1f} мс, ядер {len(ядра)}, суммарно на карте {сумма/1000:.1f} мс")
    print(f"КАРТА ЗАНЯТА {занято/окно:.1%} окна; ПРОСТОЙ {(окно-занято)/1000:.1f} мс "
          f"({(окно-занято)/окно:.1%}) -- это верхняя оценка того, что берётся УКЛАДКОЙ\n")

    print(f"{'владение':<22s} {'мс':>9s} {'доля':>7s}")
    for k, v in sorted(по_вл.items(), key=lambda x: -x[1]):
        print(f"{k:<22s} {v/1000:>9.1f} {v/сумма:>7.1%}")
    наше = sum(v for k, v in по_вл.items() if k.startswith("наше"))
    print(f"{'-'*40}\n{'ИТОГО НАШЕ':<22s} {наше/1000:>9.1f} {наше/сумма:>7.1%}\n")

    print(f"{'тело':<62s} {'мс':>8s} {'доля':>6s} {'пусков':>7s} {'мкс/пуск':>9s}  владение")
    for имя, (д, n) in sorted(по_имени.items(), key=lambda x: -x[1][0])[:a.топ]:
        print(f"{имя[:62]:<62s} {д/1000:>8.1f} {д/сумма:>6.1%} {n:>7d} {д/n:>9.1f}  {чей(имя)}")

    неоп = [(и, д) for и, (д, _) in по_имени.items() if чей(и) == "НЕОПОЗНАНО"]
    if неоп:
        неоп.sort(key=lambda x: -x[1])
        print(f"\nНЕОПОЗНАНО {sum(д for _, д in неоп)/1000:.1f} мс -- опись НЕПОЛНА, "
              f"вердикт по ней недействителен, пока это не разобрано:")
        for и, д in неоп[:8]:
            print(f"   {д/1000:>8.1f} мс  {и[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
