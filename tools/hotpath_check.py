#!/usr/bin/env python3
"""САМОПРОВЕРКА ГОРЯЧЕГО ПУТИ: не спрашивает «помнишь ли», а ПЕРЕМЕРЯЕТ.

Зачем. Подходы проекта лежат за пределами обученного знания, а контекст сессии сжимается и
теряет числа. Решение, принятое на выветрившемся знании, выглядит разумно и уводит в сторону.
Этот файл ловит ТРИ разных дрейфа, и каждый — своим способом:

  1. ДРЕЙФ МАШИНЫ    — состояние железа и процессов больше не то, при котором снимались числа
                       (частоты отпущены, боевой не там, тело пересобрано);
  2. ДРЕЙФ ДОКУМЕНТА — число в docs/HOTPATH.md разошлось с ИСТОЧНИКОМ, на который ссылается;
  3. ДРЕЙФ ПАМЯТИ    — источника вообще нет на диске, то есть число не подтверждается ничем.

Выход: 0 — горячий путь цел; 1 — что-то протухло, СПИСОК ЧТО ПЕРЕЧИТАТЬ печатается.
Запуск: python3 tools/hotpath_check.py [--quiet]
Карт не занимает, боевой не трогает.
"""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = "/mnt/d1/alex/reports"
RUN = "/mnt/d1/alex/VLLM_ch/run"
CUOBJ = "/home/alex/miniconda3/envs/cuda128/bin/cuobjdump"
BODY = "ILi8ELi87ELi6ELb0ELb1E"          # боевое тело: EPT=8, KVB=87(e4m3), GF=6

fails, warns = [], []

def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:
        return ""

def check(name, ok, detail, reload_hint):
    if ok:
        print(f"  ok    {name}")
    else:
        print(f"  ПЛОХО {name}: {detail}")
        fails.append((name, detail, reload_hint))

# --- 1. ДРЕЙФ ПАМЯТИ: источники на месте --------------------------------------------------
print("== источники ==")
SRC = {
    "спека декода":  f"{ROOT}/docs/SPEC_DECODE.md",
    "план ядер":     f"{ROOT}/docs/PLAN_FUSED_KERNELS.md",
    "карта горячего":f"{ROOT}/docs/HOTPATH.md",
    "примитивы":     f"{ROOT}/fa2_src/fmha_kernel/gemm/volta_prims.h",
    "отчёт: такт":   f"{REPORTS}/EV_decode_yarus0_clocks.md",
    "отчёт: конвейер":f"{REPORTS}/EV_decode_pipeline_pf_uu.md",
}
for n, p in SRC.items():
    check(n, os.path.exists(p), f"нет файла {p}",
          "число, на него опирающееся, НЕ ПОДТВЕРЖДЕНО — перемерить или снять заявку")

# --- 2. ДРЕЙФ ДОКУМЕНТА: числа карты против источников -------------------------------------
print("== числа против источников ==")
def num_in(path, pattern):
    if not os.path.exists(path):
        return None
    m = re.search(pattern, open(path, encoding="utf-8", errors="ignore").read())
    return m.group(1) if m else None

# упругость по такту: карта говорит 1.7 % и 8.4 %, источник обязан их содержать
for val in ("1.7", "8.4"):
    check(f"упругость такта {val} % есть в отчёте",
          val in open(SRC["отчёт: такт"], encoding="utf-8", errors="ignore").read()
          if os.path.exists(SRC["отчёт: такт"]) else False,
          "число из HOTPATH не найдено в его же источнике",
          "перечитать reports/EV_decode_yarus0_clocks.md — раскладка могла быть пересмотрена")

# доля весов 62.5 % — несущее число всей доктрины
check("доля весов 62.5 % есть в спеке",
      "62.5" in open(SRC["спека декода"], encoding="utf-8", errors="ignore").read()
      if os.path.exists(SRC["спека декода"]) else False,
      "в спеке нет 62.5 % — раскладка токена изменилась",
      "перечитать SPEC_DECODE.md §2.1 ЦЕЛИКОМ: на этой доле держится весь порядок ходов")

# ЛЕСТНИЦА ВЕЛИЧИН -- ГЛАВНОЕ, ЧТО ВЫВЕТРИВАЕТСЯ. Проверяется НЕ наличие файла, а наличие САМИХ
# ЧИСЕЛ: без них решение принимается на «ядро стало быстрее» вместо «где вообще лежит время».
# 07.08.2026 я трижды за день сползал к оптимизации тел, хотя пол запроса (1.6 с) в 94 раза больше
# всей работы шага. Эта проверка существует, чтобы такое сползание ловилось ДО решения.
_card = open(SRC["карта горячего"], encoding="utf-8", errors="ignore").read() \
    if os.path.exists(SRC["карта горячего"]) else ""
for _v, _why in (("2.375", "КОНТЕКСТНАЯ часть попадания = x1070 крыши -- ГЛАВНАЯ статья чата"),
                 ("42.7",  "шаг декода: внутри него живут ВСЕ ядерные победы"),
                 ("0.47",  "избыточных байт в декоде -- свёртка цепей НЕ байтовый рычаг"),
                 ("743",   "веса уже на 85 % канала -- иксов в умножениях нет")):
    check(f"лестница величин: {_v} ({_why})", _v in _card,
          f"числа {_v} нет в карте -- лестница выветрилась",
          "перечитать docs/HOTPATH.md РАЗДЕЛ 0 ЦЕЛИКОМ до любого решения, ЧТО оптимизировать")

# --- 3. ДРЕЙФ МАШИНЫ: тело, частоты, боевой ------------------------------------------------
print("== машина ==")
so = f"{RUN}/.fa2sm70_build/fa2sm70_decode/fa2sm70_decode.so"
if os.path.exists(so) and os.path.exists(CUOBJ):
    out = sh(f"{CUOBJ} -res-usage {so} 2>/dev/null | grep -A1 {BODY}")
    reg = re.search(r"REG:(\d+)", out)
    loc = re.search(r"LOCAL:(\d+)", out)
    reg = int(reg.group(1)) if reg else -1
    loc = int(loc.group(1)) if loc else -1
    blocks = 65536 // (reg * 128) if reg > 0 else 0
    check("боевое тело: регистры 168 и три блока на SM",
          reg == 168 and loc == 0 and blocks == 3,
          f"REG={reg} LOCAL={loc} блоков={blocks} (записано 168/0/3)",
          "тело пересобрано: ВСЕ выводы про занятость и обрыв 170.67 пересчитать заново")
else:
    warns.append(f"боевое тело не проверено (нет {so} или cuobjdump)")

clocks = sh("nvidia-smi -i 0,1 --query-gpu=clocks.sm --format=csv,noheader").split()
locked = [c for c in clocks if c.isdigit()]
check("частоты карт 0-1 зафиксированы (замер иначе недействителен)",
      len(locked) >= 2 and len(set(locked)) == 1 and int(locked[0]) >= 1200,
      f"частоты {clocks}",
      "ПРАВИЛО П2: без фиксации частот любой новый замер отвергается")

prod = ""
for pid in sh("pgrep -f vllm.entrypoints.openai.api_server").split():
    try:
        cl = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore").split("\0")
        if "--port" in cl and cl[cl.index("--port") + 1] == "8085":
            env = open(f"/proc/{pid}/environ", "rb").read().decode(errors="ignore")
            prod = env
    except Exception:
        pass
check("боевой :8085 жив и на картах 2,3",
      "CUDA_VISIBLE_DEVICES=2,3" in prod,
      "боевой не найден или не на 2,3",
      "прежде чем что-либо мерить — восстановить боевой; правило: карты 2-3 не трогать")

# --- итог ----------------------------------------------------------------------------------
print()
for w in warns:
    print(f"  (не проверено) {w}")
if not fails:
    print("ГОРЯЧИЙ ПУТЬ ЦЕЛ. Числа подтверждены источниками, машина в том же состоянии.")
    # ЗАГРУЗЧИК, А НЕ ТОЛЬКО СТОРОЖ. Проверка, которая печатает «ок», знание НЕ возвращает:
    # оно остаётся на диске, а решение принимается на выветрившемся. Поэтому при целом пути
    # карта ВЫКЛАДЫВАЕТСЯ ЦЕЛИКОМ — вызов инструмента и есть прогрев.
    if "--quiet" not in sys.argv:
        card = SRC["карта горячего"]
        if os.path.exists(card):
            print("\n" + "=" * 78)
            print("КАРТА ГОРЯЧЕГО ПУТИ (загружена в контекст; --quiet чтобы не печатать)")
            print("=" * 78)
            print(open(card, encoding="utf-8").read())
    sys.exit(0)
print("ГОРЯЧИЙ ПУТЬ ПРОТУХ. ПЕРЕЧИТАТЬ ДО ЛЮБОГО РЕШЕНИЯ:")
for n, d, hint in fails:
    print(f"  * {n}: {d}\n      -> {hint}")
print("\nПорядок загрузки: MEMORY.md -> заметки по теме -> docs/HOTPATH.md -> docs/SPEC_DECODE.md")
print("-> docs/PLAN_FUSED_KERNELS.md -> /mnt/d1/alex/reports/EV_*.md -> volta_prims.h")
sys.exit(1)
