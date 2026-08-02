# -*- coding: utf-8 -*-
"""СТАТИЧЕСКИЙ ЛИНТЕР РАСКЛАДОК РАЗДЕЛЯЕМОЙ ПАМЯТИ: ищет шаги, кратные 32 словам.

ЗАЧЕМ. Цена разделяемой памяти на Volta мерится ВАЙВФРОНТАМИ:

    вайвфронтов на команду = max( КОНФЛИКТНОСТЬ БАНКОВ , ширина_на_полосу / 8 Б , 1 )

и конвейер отдаёт ровно один вайвфронт за такт на SM. Замерено на стенде: при ЗАМОРОЖЕННЫХ 128
уникальных байтах шаг раскладки 4 -> 128 Б двигает цену 67.1 -> 2048 тактов, то есть **в 32 раза при
неизменном объёме данных**, а шаг 132 Б (дополнение на одно слово) стоит 67.7, как шаг 4. Механизм:
32 банка по слову; если шаг строки кратен 32 словам, полосы, идущие по СТОЛБЦУ, попадают в один банк.

Замер по отгруженным ядрам дал 20.8 % конфликтов в форварде и 34.6 % в бэкварде -- и всё это
находится БЕЗ ПРОФИЛИРОВЩИКА, простым счётом шага. Этот линтер существует именно поэтому: `ncu` на
Volta уже наполовину отказал (2025.x архитектуру не поддерживает, у 2024.1 CSV ломается о локаль),
а разбор исходника не зависит ни от карты, ни от sudo, ни от версии инструмента.

ГЛАВНОЕ ОГРАНИЧЕНИЕ, НАЙДЕННОЕ ЗАМЕРОМ 01.08.2026 (читать до того, как доверять выводу).

Правило "шаг строки кратен 32 словам" ловит ЛИШЬ ЧАСТЬ случаев, и на нашем главном ядре оно дало
ЛОЖНОЕ "чисто". Привязка конфликтов в volta_fwd_ws: плитка K имеет шаг LDK8 = 272 Б = 68 СЛОВ, что
на 32 НЕ делится, -- и при этом даёт 74.5 % ВСЕХ конфликтов ядра (14.5 % всего трафика разделяемой).
Зеркальный случай там же: sQ с шагом 130 слов по строкам ЧИСТ (строки через 8 дают 1040 mod 32 = 16),
чтения в мейнлупе конфликтов не дают -- а пролог перекладки двумя STS.64 даёт.

    Считать надо шаг не между СТРОКАМИ, а между СТАЛКИВАЮЩИМИСЯ ПОЛОСАМИ,
    а его задаёт КАРТА ПОЛОС доступа (например wmma_b_col(lane)), а не объявление массива.

То есть конфликтность -- свойство ПАРЫ (раскладка, карта полос), а не раскладки самой по себе. Этот
линтер видит только первую половину пары. Поэтому:

  * его "подозрение" -- повод посмотреть, а не приговор;
  * его МОЛЧАНИЕ ничего не доказывает: там, где карта полос собирает полосы в один банк при некратном
    шаге, линтер промолчит. Приговор выносит счётчик
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st} с посточечной привязкой.

Что нужно, чтобы правило стало полным: разбирать выражение индекса вместе с картой полос варпа и
считать распределение (адрес/4) mod 32 по 32 полосам одной команды. Это и есть следующая версия.

ЧЕСТНОСТЬ ИНСТРУМЕНТА. Неполный инструмент даёт не "меньше данных", а ДРУГОЙ ОТВЕТ, и звучит он так
же уверенно -- в этом проекте это уже стоило одного ложного вывода. Поэтому линтер ОБЯЗАН печатать
раздел "НЕ РАЗОБРАНО": объявления, чей шаг он не смог вычислить. Пустой список подозрений при
непустом списке неразобранного НЕ ОЗНАЧАЕТ "конфликтов нет".

ЗАПУСК:  python tools/smem_lint.py [корень]
"""
import os
import re
import sys

WORD = 4                      # банк = слово = 4 байта
BANKS = 32

SIZEOF = {
    "char": 1, "int8_t": 1, "uint8_t": 1, "unsigned char": 1,
    "half": 2, "__half": 2, "cutlass::half_t": 2, "half_t": 2, "short": 2, "uint16_t": 2,
    "float": 4, "int": 4, "unsigned": 4, "uint32_t": 4, "int32_t": 4, "unsigned int": 4,
    "double": 8, "int64_t": 8, "uint64_t": 8, "float2": 8, "half2": 4, "__half2": 4,
    "float4": 16, "int4": 16,
}

# __shared__ T name[A][B];  и  __shared__ T name[A];
DECL = re.compile(
    r"__shared__\s+(?:extern\s+)?([A-Za-z_][\w:<>,\s]*?)\s+(\w+)\s*((?:\[[^\]]+\])+)\s*;")
DIMS = re.compile(r"\[([^\]]+)\]")
# ИМЕНОВАННЫЕ ШАГИ ЛОВИМ НЕ ПО ИМЕНИ, А ПО УПОТРЕБЛЕНИЮ. Первая версия линтера искала константы с
# похожими именами (LD*, *PITCH, *STRIDE) -- и поймала NIT, ITERS, N_DIM, то есть счётчики итераций.
# Шумный инструмент это ВРУЩИЙ инструмент: он выдаёт другой ответ с той же уверенностью. Поэтому
# константа считается шагом ТОЛЬКО если она стоит множителем в индексе РАЗДЕЛЯЕМОГО массива.
SHNAME = re.compile(r"__shared__\s+(?:extern\s+)?[A-Za-z_][\w:<>,\s]*?\s+(\w+)\s*[\[;]")
ANYCONST = re.compile(r"(?:#define\s+(\w+)\s+(\d+)\b)"
                      r"|(?:(?:constexpr|const|static)\s+(?:int|unsigned|size_t)\s+(\w+)\s*=\s*(\d+)\s*;)")

SKIP_DIRS = {".git", "build", "_build", "__pycache__", "cutlass"}   # cutlass свизлит сам


def evaluate(expr, consts):
    """Осторожное вычисление константного выражения. Не смог -- вернуть None, а НЕ угадать."""
    e = expr.strip()
    for name, val in consts.items():
        e = re.sub(r"\b" + re.escape(name) + r"\b", str(val), e)
    if not re.fullmatch(r"[\d\s+\-*/()]+", e):
        return None
    try:
        v = eval(e, {"__builtins__": {}}, {})       # только цифры и арифметика, см. проверку выше
        return int(v) if isinstance(v, (int, float)) and v == int(v) else None
    except Exception:
        return None


def scan_file(path):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return [], []
    consts = {}
    for m in ANYCONST.finditer(src):
        name = m.group(1) or m.group(3)
        val = m.group(2) or m.group(4)
        if name:
            consts.setdefault(name, int(val))
    shared_names = set(SHNAME.findall(src))

    hits, unparsed = [], []
    for m in DECL.finditer(src):
        ty = m.group(1).strip().replace("volatile ", "").strip()
        name = m.group(2)
        dims = DIMS.findall(m.group(3))
        line = src[:m.start()].count("\n") + 1
        esz = SIZEOF.get(ty)
        if esz is None or len(dims) < 2:
            # одномерный массив шага не имеет -- это не "не разобрано", это неприменимо
            if esz is None:
                unparsed.append((path, line, f"{ty} {name}{''.join('['+d+']' for d in dims)}",
                                 "неизвестный тип элемента"))
            continue
        inner = evaluate(dims[-1], consts)
        if inner is None:
            unparsed.append((path, line, f"{ty} {name}[..][{dims[-1]}]",
                             "шаг -- невычислимое выражение"))
            continue
        stride_words = inner * esz / WORD
        if stride_words != int(stride_words):
            continue                                 # шаг не по границе слова -- отдельный разговор
        stride_words = int(stride_words)
        if stride_words and stride_words % BANKS == 0:
            hits.append((path, line, f"{ty} {name}[{']['.join(dims)}]", stride_words,
                         f"шаг {stride_words} слов = {stride_words // BANKS}x32 -> все полосы "
                         f"столбца в ОДИН банк; дополнить до {stride_words + 8}"))
    # ШАГИ ПО УПОТРЕБЛЕНИЮ: множитель в индексе разделяемого массива.  name[ ... * K ... ]
    for sname in shared_names:
        for m in re.finditer(re.escape(sname) + r"\s*\[([^\]\[]{1,120})\]", src):
            idx = m.group(1)
            for fm in re.finditer(r"(?:(\w+)\s*\*\s*(\w+))", idx):
                for cand in fm.groups():
                    val = consts.get(cand)
                    if val is None and cand.isdigit():
                        val = int(cand)
                    if val is None or val < 32:
                        continue
                    for esz in (2, 4):
                        w = val * esz // WORD
                        if w and w % BANKS == 0:
                            line = src[:m.start()].count("\n") + 1
                            hits.append((path, line, f"{sname}[{idx.strip()}]  ({cand} = {val})", w,
                                         f"множитель {val} при элементе {esz} Б даёт шаг {w} слов = "
                                         f"{w // BANKS}x32 -> столбец в один банк; "
                                         f"дополнить множитель до {val + (8 * WORD // esz)}"))
                            break
                    break
    return hits, unparsed


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    hits, unparsed, nfiles = [], [], 0
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in files:
            if f.endswith((".cu", ".cuh", ".h", ".hpp")):
                nfiles += 1
                h, u = scan_file(os.path.join(dirpath, f))
                hits += h
                unparsed += u

    print(f"# просмотрено файлов: {nfiles}\n")
    print("=" * 100)
    print("ПОДОЗРЕНИЯ: шаг строки кратен 32 словам (128 Б) -> полосы столбца в один банк")
    print("=" * 100)
    if not hits:
        print("(нет)")
    for path, line, what, w, why in sorted(hits, key=lambda x: -x[3]):
        print(f"{os.path.relpath(path, root)}:{line}\n    {what}\n    {why}")

    print("\n" + "=" * 100)
    print("НЕ РАЗОБРАНО -- линтер НЕ утверждает, что здесь чисто")
    print("=" * 100)
    if not unparsed:
        print("(всё разобрано)")
    for path, line, what, why in unparsed[:40]:
        print(f"{os.path.relpath(path, root)}:{line}  {what}  -- {why}")
    if len(unparsed) > 40:
        print(f"... и ещё {len(unparsed) - 40}")

    print(f"\nИТОГ: подозрений {len(hits)}, неразобранных объявлений {len(unparsed)}.")
    print("ПУСТОЙ СПИСОК ПОДОЗРЕНИЙ ПРИ НЕПУСТОМ СПИСКЕ НЕРАЗОБРАННОГО НЕ ОЗНАЧАЕТ 'КОНФЛИКТОВ НЕТ'.")
    print("Подозрение -- не приговор: если по столбцу этого массива никто не ходит, конфликта нет.")
    print("Приговор выносит счётчик l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
