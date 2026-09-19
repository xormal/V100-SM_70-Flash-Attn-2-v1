"""КАНАЛ ИЛИ ЦЕПЬ: что связывает тело мейнлупа -- и, значит, ЧТО с ним делать.

ЗАЧЕМ ИМЕННО ЭТОТ ВОПРОС. Есть ровно два разных лечения, и они не взаимозаменяемы:

  * связывает КАНАЛ (ResMII: сумма отпечатков по ресурсу / его пропускная способность) --
    помогает ТОЛЬКО СНЯТИЕ РАБОТЫ с этого канала: специализация, другой примитив, другая
    раскладка. Добавление стадий конвейера НЕ ПОМОЖЕТ: канал занят одинаково при любой укладке;
  * связывает ЦЕПЬ (RecMII: максимум по циклам графа зависимостей отношения длина/обороты) --
    помогает ТОЛЬКО РАЗНЕСЕНИЕ ПО СТАДИЯМ: больше стадий, перекос по плиткам, роль на звено.
    Снятие команд НЕ ПОМОЖЕТ: цепь той же длины.

Обе границы считаются ПО МАШИННОМУ КОДУ и БЕЗ ЕДИНОЙ СБОРКИ вариантов -- это дёшево, в отличие
от полного поиска расписания (он NP-труден и на теле нашего форварда не сходится за 10 минут).
Здесь ТОЛЬКО границы и опись по каналам: их достаточно, чтобы отсеять неверный КЛАСС правки.

Ядро укладчика взято из `/mnt/d1/alex/tempo` (tools/schedule.py, плагин sm70) -- заводить свой
разборщик SASS означало бы получить ВТОРОЙ источник уверенных неверных ответов.

ЧЕСТНОСТЬ ОХВАТА. Инструмент, нашедший меньше сущностей, чем их есть, даёт не «меньше данных», а
ДРУГОЙ ОТВЕТ. Поэтому тела, у которых мейнлуп не опознан, считаются и печатаются отдельной
строкой: сплошной прогон без этой строки читать нельзя.

!!! ГЛАВНОЕ ОГРАНИЧЕНИЕ, НА КОТОРОЕ Я НАСТУПИЛ В ДЕНЬ НАПИСАНИЯ ИНСТРУМЕНТА !!!
**Здесь считается ПОТЕНЦИАЛ тела, а не его ДОЛЯ в проходе.** Тело с запасом канала x5, занимающее
0.1 % шага, стоит РОВНО НОЛЬ, и таблица этого не покажет -- она отсортирует его наверх. Ровно так
я и вынес вердикт «шесть макрофаз декода -- вот конструкция мегаядра», не проверив, что декодное
внимание на боевом маршруте это НОЛЬ шага (веса 96-99 % своего потолка, обмен и хвост -- вот шаг).
Поэтому: **`--doli <файл.pt.trace.json>` обязателен для любого вердикта о порядке работ**; без
него инструмент печатает предупреждение и отказывается ранжировать. Ранжирование по потенциалу
годится только внутри УЖЕ ВЫБРАННОГО по доле тела. Опись по времени -- `tools/opis_puti.py`.

    python tools/kanal_ili_cep.py <файл.so> [ещё.so ...] [--kernel регексп] [--warps 8] [--top 40]
    python tools/kanal_ili_cep.py --vse            # все собранные расширения из FA2SM70_BUILD_DIR
"""
import argparse
import collections
import glob
import os
import re
import sys

ТЕМПО = os.environ.get("TEMPO_ROOT", "/mnt/d1/alex/tempo")
sys.path.insert(0, os.path.join(ТЕМПО, "tools"))
sys.path.insert(0, os.path.join(ТЕМПО, "tempo", "plugins", "sm70"))
try:
    import tempo as T            # noqa: E402
    import isa_sass as IS        # noqa: E402
    import schedule as S         # noqa: E402
except ImportError as e:         # noqa: BLE001
    print(f"укладчик не найден в {ТЕМПО}: {e}\n"
          "путь задаётся TEMPO_ROOT; без него вердикта нет -- это НЕ повод считать на глаз")
    raise SystemExit(2)

# Сколько варпов РЕАЛЬНО делит планировщики у каждого семейства тел. Число входит в ResMII прямо
# (нагрузка на канал делится на варпы), поэтому одно значение на все ядра дало бы ранжирование,
# верное только для одного из них.
ВАРПОВ = (
    (r"volta_fwd_ws", 8, "форвард: роли, считающих 8 из 12"),
    (r"volta_fwd_i8|volta_fwd_block|fmha", 8, "форвард: 8 варпов"),
    (r"volta_bwd|kernel_backward|attn_bwd", 16, "обратный проход: 16 варпов"),
    (r"chunk_h|chunk_o|gdn|solve_tril", 8, "скан GDN"),
    (r"split_decode|decode_paged|split_kv", 4, "декод: 4 варпа"),
    (r"w12a16|w8a16|gemv|fused_norm", 4, "гемв: 4 варпа"),
    (r"fused_mlp|mega", 8, "мегаядро MLP"),
)


def варпов(имя, по_умолчанию):
    for об, w, _ in ВАРПОВ:
        if re.search(об, имя, re.I):
            return w
    return по_умолчанию


def разбор(имя, тело, warps):
    fps = [T.footprint(it, safe=False) for it in тело]
    edges = T.dep_graph(тело)
    res = S._resmii(fps, warps)
    rec = T.rec_mii(len(тело), edges)
    груз, счёт = collections.Counter(), collections.Counter()
    for fp in fps:
        for ch, c in fp.items():
            груз[ch] += c
            счёт[ch] += 1
    wps = warps / float(T.SCHEDULERS)
    такты = {ch: (v * warps / T.MIO_BYTES_PER_CYCLE if ch == T.CH_MIO else v * wps)
             for ch, v in груз.items()}
    низ = max(res, rec)
    главный = max(такты, key=lambda c: такты[c]) if такты else None
    return {"имя": имя, "n": len(тело), "res": res, "rec": rec, "низ": низ, "warps": warps,
            "такты": такты, "счёт": счёт, "главный": главный}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", nargs="*")
    ap.add_argument("--kernel", default="")
    ap.add_argument("--warps", type=int, default=0, help="0 -- взять из таблицы ВАРПОВ по имени")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--vse", action="store_true", help="все .so из FA2SM70_BUILD_DIR")
    ap.add_argument("--kanaly", action="store_true", help="печатать опись по каналам для каждого тела")
    ap.add_argument("--doli", help="лента профилировщика: БЕЗ НЕЁ ранжирование недействительно")
    a = ap.parse_args()

    # ДОЛЯ ВРЕМЕНИ ИЗ ЛЕНТЫ. Она умножается на потенциал: тело, которого на маршруте нет, обязано
    # уйти вниз таблицы, а не наверх. Без ленты ранжирования не будет вовсе -- только предупреждение.
    доли, окно = {}, 0.0
    if a.doli:
        import json
        соб = json.load(open(a.doli))["traceEvents"]
        for e in соб:
            if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel",):
                доли[e["name"]] = доли.get(e["name"], 0.0) + e["dur"]
        окно = sum(доли.values()) or 1.0

    def доля(имя):
        """Имя в ленте -- ДЕМАНГЛИРОВАННОЕ, в cubin -- манглированное. Сопоставляем по подстроке
        опознавательного корня; не нашли -- возвращаем None, а не ноль (отсутствие в ленте и
        нулевая доля -- РАЗНЫЕ утверждения, и путать их значит снова врать в пользу вердикта)."""
        if not доли:
            return None
        корень = re.sub(r"^_Z\d*|^_ZN\d*", "", имя)[:20]
        s = sum(v for k, v in доли.items() if корень[:12] and корень[:12] in k)
        return s / окно if s else 0.0

    файлы = list(a.obj)
    if a.vse:
        корень = os.environ.get("FA2SM70_BUILD_DIR",
                                os.path.join(os.path.dirname(os.path.dirname(
                                    os.path.abspath(__file__))), "_build"))
        файлы += sorted(glob.glob(os.path.join(корень, "*", "*.so")))
    if not файлы:
        print("нечего разбирать: дайте .so или --vse"); return 2

    строки, немых, всего = [], [], 0
    for ф in файлы:
        try:
            ядра = T.load_kernels(ф)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(ф)}: не читается ({str(e)[:60]})"); continue
        for k in ядра:
            if a.kernel and not re.search(a.kernel, k):
                continue
            всего += 1
            try:
                cfg = IS.CFG(ядра[k])
                тело, _ = T.mainloop_ins(cfg)
            except Exception:  # noqa: BLE001
                тело = None
            if not тело:
                немых.append(k)
                continue
            строки.append(разбор(k, тело, a.warps or варпов(k, 8)))

    for r in строки:
        r["доля"] = доля(r["имя"])
    # СОРТИРОВКА ПО ДОЛЕ, ЕСЛИ ОНА ЕСТЬ. Потенциал -- только вторичный ключ.
    if доли:
        строки.sort(key=lambda r: (-(r["доля"] or 0.0), -r["низ"]))
    else:
        строки.sort(key=lambda r: -r["низ"])
        print("\n!!! ЛЕНТЫ НЕТ (--doli не задан): ЭТО НЕ ПОРЯДОК РАБОТ !!!\n"
              "Таблица ниже ранжирует по ПОТЕНЦИАЛУ. Тело с запасом канала x5, занимающее 0.1 %\n"
              "прохода, встанет наверх и будет стоить ноль. Прежде чем что-то по ней делать --\n"
              "снять долю времени (`tools/opis_puti.py` по ленте профилировщика движка).")
    print(f"\nтел разобрано {len(строки)} из {всего}; мейнлуп не опознан у {len(немых)}\n")
    print(f"{'тело':<44s} {'вар':>4s} {'команд':>7s} {'ResMII':>8s} {'RecMII':>8s} "
          f"{'связывает':<8s} {'канал':<8s} {'занят':>6s} {'ДОЛЯ':>7s}")
    for r in строки[:a.top]:
        кто = "КАНАЛ" if r["res"] >= r["rec"] else "ЦЕПЬ"
        д = r["такты"][r["главный"]] / r["низ"] if r["низ"] else 0
        дл = "  нет" if r["доля"] is None else f"{r['доля']:>6.2%}"
        print(f"{r['имя'][:44]:<44s} {r['warps']:>4d} {r['n']:>7d} {r['res']:>8.1f} "
              f"{r['rec']:>8.1f} {кто:<8s} {str(r['главный']):<8s} {д:>5.1%} {дл:>7s}")
        if a.kanaly:
            for ch, t in sorted(r["такты"].items(), key=lambda x: -x[1]):
                print(f"        {str(ch):<12s} команд {r['счёт'][ch]:>5d}  тактов {t:>9.1f}  "
                      f"{t / r['низ']:>6.1%}")

    цепи = [r for r in строки if r["rec"] > r["res"]]
    print(f"\nСВЯЗЫВАЕТ ЦЕПЬ (лечится СТАДИЯМИ): {len(цепи)} тел из {len(строки)}")
    for r in цепи[:12]:
        print(f"    {r['имя'][:60]:<60s} RecMII {r['rec']:.1f} против ResMII {r['res']:.1f}")
    if немых:
        print(f"\nМЕЙНЛУП НЕ ОПОЗНАН у {len(немых)} тел -- по ним вердикта НЕТ, и опись НЕПОЛНА:")
        for k in немых[:12]:
            print(f"    {k[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
