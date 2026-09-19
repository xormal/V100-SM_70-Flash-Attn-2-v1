"""ПЕРЕУКЛАДЧИК ТЕЛА (задача 104): комбинаторный поиск порядка команд, а не один проход.

ПОСТАНОВКА (поправка владельца 19.08). Переукладка -- комбинаторная задача с разменами
(регистры <-> тени <-> длина расписания <-> занятость); жадные одиночные ходы её не решают:
период = max по каналам, и размены съедают одиночный ход. Здесь:

  * ИНВАРИАНТЫ перестановки считаются один раз: ResMII (каналы) и RecMII (цепи) от порядка
    команд НЕ зависят -- их лечат другие классы правок (снятие работы / стадии);
  * ДВИЖЕТСЯ перестановкой ровно ДЛИНА ФАКТИЧЕСКОГО РАСПИСАНИЯ T1 (наш закон R6: при дефиците
    теней связывает она) и РЕГИСТРОВОЕ ДАВЛЕНИЕ (пик живых значений; tm8 упёрся в 255);
  * ДВА движка: точный CP-SAT (makespan при precedence-дугах и слоте выдачи 1/такт) и
    отжиг по перестановкам с тем же оценщиком -- CP-SAT даёт и НИЖНЮЮ ГРАНИЦУ,
    отжиг -- решение, когда точному не хватает времени;
  * выход -- эталонный порядок + КАРТА СДВИГОВ по классам команд (какие классы двигать в
    исходнике), потому что порядок в железо несёт ptxas, а не мы: перенос обязан идти петлёй
    исходник -> сборка -> SASS-сверка -> секундомер (граница познаваемого 0.747-1.254, №144).

Ядро разбора -- из /mnt/d1/alex/tempo (isa_sass, tempo.dep_graph/footprint/LATENCY):
второй разборщик SASS был бы вторым источником уверенных неверных ответов.

    python tools/pereukladchik.py <файл.sass|.so> [--kernel regex] [--time-limit 120]
                                  [--warps 4] [--no-cpsat]
"""
import argparse
import bisect
import collections
import os
import random
import re
import subprocess
import sys

ТЕМПО = os.environ.get("TEMPO_ROOT", "/mnt/d1/alex/tempo")
sys.path.insert(0, os.path.join(ТЕМПО, "tools"))
try:
    import tempo as T          # noqa: E402
    import isa_sass as IS      # noqa: E402
except ImportError as e:       # noqa: BLE001
    print(f"ядро укладчика не найдено в {ТЕМПО}: {e}")
    raise SystemExit(2)


# ------------------------------------------------------------------ извлечение горячего цикла
def читать_sass(путь, kernel_re):
    if путь.endswith(".sass"):
        текст = open(путь).read()
    else:
        cu = IS.find_cuobjdump()
        текст = subprocess.check_output([cu, "-sass", путь], text=True)
    if "Function :" not in текст:                       # голый дамп одного ядра
        текст = "Function : <без имени>\n" + текст
    тела = IS.parse_sass(текст)                          # OrderedDict{имя: [Ins]}
    if kernel_re:
        тела = {k: v for k, v in тела.items() if re.search(kernel_re, k)}
    if not тела:
        raise SystemExit("ядро по образцу не найдено")
    имя = max(тела, key=lambda k: sum(1 for it in тела[k] if "HMMA" in it.op))
    return имя, тела[имя]


def горячий_цикл(body):
    """Самый тензорно-плотный естественный цикл; если циклов нет -- отказ, не догадка."""
    cfg = IS.CFG(body)
    петли = cfg.natural_loops()          # [(header, set(block_addrs), backedge)]
    if not петли:
        raise SystemExit("естественных циклов нет -- мейнлуп не опознан (это отказ, не ноль)")
    лучший, счёт = None, -1
    for головной, узлы, _хвост in петли:
        ins = [i for a in sorted(узлы) for i in cfg.blocks[a]]
        c = sum(1 for it in ins if "HMMA" in it.op)
        if c > счёт or (c == счёт and лучший is not None and len(ins) < len(лучший)):
            счёт, лучший = c, ins
    if счёт <= 0:
        # Тела без тензорных команд (гемвы декода, сборщики): горячий цикл -- максимальный по
        # числу команд. Пометка в отчёте обязательна: выбор эвристический, а не по HMMA.
        лучший = max((sorted(узлы) for _г, узлы, _х in петли),
                     key=lambda уз: sum(len(cfg.blocks[a]) for a in уз))
        лучший = [i for a in лучший for i in cfg.blocks[a]]
    return лучший


# ------------------------------------------------------------------ оценщик: T1 и давление
def дуги(тело):
    """(i, j, lat) внутри итерации; межоборотные отдаются отдельно (для отчёта)."""
    внутр, меж = [], []
    for i, j, lat, обороты in T.dep_graph(тело):
        (внутр if обороты == 0 else меж).append((i, j, max(1, int(lat))))
    return внутр, меж


def t1_расписание(порядок, внутр_дуги_по_j):
    """Длина фактического расписания ОДНОГО варпа: выдача 1/такт по порядку, команда ждёт
    готовности операндов (scoreboard). Ровно то, что двигается перестановкой (R6)."""
    поз = {v: k for k, v in enumerate(порядок)}
    готов = [0] * len(порядок)
    такт = 0
    for p, i in enumerate(порядок):
        t = такт
        for (a, lat) in внутр_дуги_по_j.get(i, ()):
            t = max(t, готов[a])
        такт = t + 1
        готов[i] = t + внутр_дуги_по_j.get(("lat", i), 1)
    return такт


def оценщик(тело, внутр):
    """Возвращает функцию порядка -> (T1, пик_регистров)."""
    n = len(тело)
    lat_i = [max(1, int(T.latency(it))) for it in тело]
    по_j = collections.defaultdict(list)
    for (a, b, lat) in внутр:
        по_j[b].append((a, lat))
    # def-use интервалы для давления: производитель a жив до последнего потребителя
    посл_потр = {}
    for (a, b, _l) in внутр:
        посл_потр[a] = max(посл_потр.get(a, a), b)

    def оценка(порядок):
        поз = [0] * n
        for p, i in enumerate(порядок):
            поз[i] = p
        # T1: выдача в порядке, ожидание готовности источников
        готов = [0] * n
        такт = 0
        for i in (порядок):
            t = такт
            for (a, lat) in по_j[i]:
                t = max(t, готов[a])
            такт = t + 1
            готов[i] = t + lat_i[i]
        T1 = такт
        # давление: сколько интервалов (поз[a], поз[последний потребитель]) накрывают точку
        события = []
        for a, b in посл_потр.items():
            события.append((поз[a], 1))
            события.append((поз[b], -1))
        события.sort()
        пик = тек = 0
        for _, d in события:
            тек += d
            пик = max(пик, тек)
        return T1, пик

    return оценка


# ------------------------------------------------------------------ поиск: отжиг
def отжиг(тело, внутр, шаги=200000, seed=104, w_press=0.05):
    n = len(тело)
    оц = оценщик(тело, внутр)
    преды = collections.defaultdict(set)
    потомки = collections.defaultdict(set)
    for (a, b, _l) in внутр:
        преды[b].add(a)
        потомки[a].add(b)
    порядок = list(range(n))                     # фактический порядок SASS -- стартовая точка
    T1, пик = оц(порядок)
    базT1, базПик = T1, пик
    бест = (T1 + w_press * пик, list(порядок), T1, пик)
    цена = бест[0]
    rnd = random.Random(seed)
    поз = list(range(n))
    for шаг in range(шаги):
        # ход: взять команду и сдвинуть в СЛУЧАЙНУЮ допустимую позицию её окна
        p = rnd.randrange(n)
        i = порядок[p]
        lo = 0
        for a in преды[i]:
            lo = max(lo, поз[a] + 1)
        hi = n - 1
        for b in потомки[i]:
            hi = min(hi, поз[b] - 1)
        if hi <= lo:
            continue
        q = rnd.randint(lo, hi)
        if q == p:
            continue
        порядок.pop(p)
        порядок.insert(q, i)
        lo2, hi2 = (min(p, q), max(p, q))
        for pp in range(lo2, hi2 + 1):
            поз[порядок[pp]] = pp
        T1n, пикn = оц(порядок)
        ценаn = T1n + w_press * пикn
        темп = max(0.02, 3.0 * (1.0 - шаг / шаги))
        if ценаn <= цена or rnd.random() < pow(2.718, -(ценаn - цена) / темп):
            цена = ценаn
            if ценаn < бест[0]:
                бест = (ценаn, list(порядок), T1n, пикn)
        else:
            порядок.pop(q)
            порядок.insert(p, i)
            for pp in range(lo2, hi2 + 1):
                поз[порядок[pp]] = pp
    return базT1, базПик, бест


# ------------------------------------------------------------------ точный слой: CP-SAT
def cpsat(тело, внутр, лимит_с=120, w_press=0.05):
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None
    n = len(тело)
    lat_i = [max(1, int(T.latency(it))) for it in тело]
    H = n + sum(l for (_a, _b, l) in внутр) // max(1, n) * 4 + max(lat_i) * 4 + n
    m = cp_model.CpModel()
    s = [m.NewIntVar(0, H, f"s{i}") for i in range(n)]
    # слот выдачи: 1 команда на такт
    m.AddAllDifferent(s)
    for (a, b, lat) in внутр:
        m.Add(s[b] >= s[a] + lat)
    mk = m.NewIntVar(0, H, "mk")
    m.AddMaxEquality(mk, [s[i] + 1 for i in range(n)])
    # давление v1 -- в целевой не входит (пост-проверка оценщиком); цель -- makespan
    m.Minimize(mk)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(лимит_с)
    solver.parameters.num_search_workers = 8
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    порядок = sorted(range(n), key=lambda i: solver.Value(s[i]))
    return {"статус": "OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE",
            "нижняя": int(solver.BestObjectiveBound()),
            "makespan": int(solver.Value(mk)),
            "порядок": порядок}


# ------------------------------------------------------------------ точный слой 2: MODULO
def cpsat_modulo(тело, внутр, меж, II, лимит_с=60):
    """Стационарное расписание с периодом II: intra s_b >= s_a + lat; межоборотные
    s_b + II >= s_a + lat; слот выдачи -- AllDifferent по s mod II. Feasibility при фикс. II."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None
    n = len(тело)
    H = II * 3
    m = cp_model.CpModel()
    s = [m.NewIntVar(0, H, f"s{i}") for i in range(n)]
    slot = [m.NewIntVar(0, II - 1, f"m{i}") for i in range(n)]
    for i in range(n):
        m.AddModuloEquality(slot[i], s[i], II)
    m.AddAllDifferent(slot)
    for (a, b, lat) in внутр:
        m.Add(s[b] >= s[a] + lat)
    for (a, b, lat) in меж:
        m.Add(s[b] + II >= s[a] + lat)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(лимит_с)
    solver.parameters.num_search_workers = 8
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return sorted(range(n), key=lambda i: solver.Value(s[i]))


def modulo_binsearch(тело, внутр, меж, lo, hi, лимит_с=60):
    """Минимальный II, при котором стационарное расписание существует (док-во солвером)."""
    best = None
    while lo < hi:
        mid = (lo + hi) // 2
        r = cpsat_modulo(тело, внутр, меж, mid, лимит_с)
        if r is not None:
            best, hi = (mid, r), mid
        else:
            lo = mid + 1
    if best is None:
        r = cpsat_modulo(тело, внутр, меж, hi, лимит_с)
        best = (hi, r) if r is not None else None
    return best


# ------------------------------------------------------------------ карта сдвигов по классам
def карта_сдвигов(тело, было, стало, top=14):
    поз_б = {v: p for p, v in enumerate(было)}
    поз_с = {v: p for p, v in enumerate(стало)}
    сдвиги = collections.defaultdict(list)
    for i, it in enumerate(тело):
        кл = IS.classify(it.op) if hasattr(IS, "classify") else T.classify(it.op)
        сдвиги[f"{кл}:{it.op.split('.')[0]}"].append(поз_с[i] - поз_б[i])
    строки = []
    for кл, xs in sorted(сдвиги.items(), key=lambda kv: -abs(sum(kv[1]) / len(kv[1]))):
        ср = sum(xs) / len(xs)
        строки.append(f"  {кл:28s} n={len(xs):4d} сдвиг ср={ср:+7.1f} мин={min(xs):+5d} макс={max(xs):+5d}")
    return "\n".join(строки[:top])


def пакет(so_dir, лимит_с, steps, warps_dflt, max_n=900):
    import glob as _g
    import schedule as S
    строки = []
    for so in sorted(_g.glob(os.path.join(so_dir, "*", "*.so"))):
        cu = IS.find_cuobjdump()
        try:
            текст = subprocess.check_output([cu, "-sass", so], text=True)
        except subprocess.CalledProcessError:
            строки.append((os.path.basename(so), "-", "SASS не снялся"))
            continue
        тела = IS.parse_sass(текст)
        for имя, body in тела.items():
            try:
                цикл = горячий_цикл(body)
            except SystemExit:
                continue
            if len(цикл) < 24:
                continue
            кор = имя if len(имя) < 40 else re.sub(r"^_ZN.*?(volta_\w+|chunk_\w+|solve_\w+|split_\w+|gemm_kernel|fused_\w+|w12\w*|w8a16\w*|exch\w*|gather\w+|attn\w+|[A-Za-z_]\w{5,})", r"\1", имя)[:44]
            if len(цикл) > max_n:
                строки.append((кор, len(цикл), "цикл велик -- пропуск (честный отказ)"))
                continue
            внутр, меж = дуги(цикл)
            fps = [T.footprint(it, safe=False) for it in цикл]
            res = S._resmii(fps, warps_dflt)
            rec = T.rec_mii(len(цикл), [(a, b, l, 0) for (a, b, l) in внутр] +
                            [(a, b, l, 1) for (a, b, l) in меж], limit=1000)
            оц = оценщик(цикл, внутр)
            T1f, пикf = оц(list(range(len(цикл))))
            r = cpsat(цикл, внутр, лимит_с=лимит_с)
            if r:
                T1o, пикo = оц(r["порядок"])
                статус = r["статус"]
            else:
                _b1, _b2, (цена, поряд, T1o, пикo) = отжиг(цикл, внутр, шаги=steps)
                статус = "отжиг"
            фактП = max(res, rec, T1f)
            планП = max(res, rec, T1o)
            зазор = (фактП - планП) / фактП * 100 if фактП else 0.0
            hm = sum(1 for it in цикл if "HMMA" in it.op)
            строки.append((кор, len(цикл),
                           f"HMMA={hm:3d} Res={res:5.0f} Rec={rec:4.0f} T1 факт={T1f:5d} "
                           f"опт={T1o:5d}({статус}) период {фактП:5.0f}->{планП:5.0f} "
                           f"ЗАЗОР {зазор:4.1f}%"))
    print("\n=== ПАКЕТ: тела и зазоры укладки (период = max(Res,Rec,T1)) ===")
    for кор, n, s in строки:
        print(f"{кор:44s} n={n:4} {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("--kernel", default="")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--no-cpsat", action="store_true")
    ap.add_argument("--batch", action="store_true", help="obj -- каталог сборки: пройти все .so")
    a = ap.parse_args()

    if a.batch:
        пакет(a.obj, a.time_limit, a.steps, a.warps)
        return

    имя, body = читать_sass(a.obj, a.kernel)
    цикл = горячий_цикл(body)
    print(f"тело: {имя}\nгорячий цикл: {len(цикл)} команд, HMMA={sum(1 for i in цикл if 'HMMA' in i.op)}")

    внутр, меж = дуги(цикл)
    fps = [T.footprint(it, safe=False) for it in цикл]
    import schedule as S
    res = S._resmii(fps, a.warps)
    rec = T.rec_mii(len(цикл), [(i, j, l, t) for (i, j, l) in внутр for t in (0,)] +
                    [(i, j, l, 1) for (i, j, l) in меж], limit=1000)
    print(f"ИНВАРИАНТЫ (порядок их не двигает): ResMII={res:.0f}  RecMII={rec:.0f}  дуг внутри={len(внутр)} через оборот={len(меж)}")

    базT1, базПик, (цена, поряд, T1, пик) = отжиг(цикл, внутр, шаги=a.steps)
    print(f"ФАКТ (порядок ptxas):   T1={базT1}  пик живых={базПик}")
    print(f"ОТЖИГ ({a.steps} шагов): T1={T1}  пик живых={пик}  (T1 {'-' if T1<=базT1 else '+'}{abs(базT1-T1)} тактов, {100*(базT1-T1)/max(1,базT1):+.1f}%)")

    if not a.no_cpsat:
        r = cpsat(цикл, внутр, лимит_с=a.time_limit)
        if r:
            print(f"CP-SAT: {r['статус']}  makespan={r['makespan']}  нижняя граница={r['нижняя']}")
            оц = оценщик(цикл, внутр)
            T1c, пикc = оц(r["порядок"])
            print(f"CP-SAT порядок оценщиком: T1={T1c}  пик живых={пикc}")
            if T1c < T1:
                поряд, T1, пик = r["порядок"], T1c, пикc
        else:
            print("CP-SAT: не сошёлся за лимит (отжиг остаётся решением)")

    # MODULO: честный период на стыке итераций (межоборотные дуги входят)
    if not a.no_cpsat:
        lo = int(max(res, rec, 1))
        mb = modulo_binsearch(цикл, внутр, меж, lo, max(lo, базT1), лимит_с=a.time_limit // 2 or 30)
        if mb:
            II, поряд_mod = mb
            оц = оценщик(цикл, внутр)
            T1m, пикm = оц(поряд_mod)
            print(f"MODULO: минимальный доказанный период II={II} (нижняя {lo}); "
                  f"порядок этого расписания: T1={T1m} пик живых={пикm}")
            if II <= max(res, rec, T1):
                поряд = поряд_mod
        else:
            print("MODULO: стационарное расписание в лимит не нашлось (остаётся makespan-порядок)")

    print(f"\nПЕРИОД (модель, {a.warps} варпа): факт=max({res:.0f},{rec:.0f},{базT1})={max(res,rec,базT1):.0f}"
          f" -> план=max({res:.0f},{rec:.0f},{T1})={max(res,rec,T1):.0f}")
    print("\nКАРТА СДВИГОВ ПО КЛАССАМ (перенос в исходник -- вот этими классами):")
    print(карта_сдвигов(цикл, list(range(len(цикл))), поряд))
    print("\nПЕРЕНОС: порядок в железо кладёт ptxas -- правку исходника гейтить SASS-сверкой и"
          " секундомером (полоса переноса 0.747-1.254, задача 144).")


if __name__ == "__main__":
    main()
