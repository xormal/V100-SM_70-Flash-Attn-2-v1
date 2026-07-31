#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""УЧЁТ ПО СЛОТАМ ВЫДАЧИ для ядер sm_70 (V100).  Спецификация -- docs/VOLTA_SM70.md §3b.

ЧТО СЧИТАЕТ
  * выделяет МЕЙНЛУП по обратным дугам (доминаторы -> естественные циклы), а не на глаз;
  * nu = число выдач НЕТЕНЗОРНЫХ команд на одну тензорную в мейнлупе;
  * разбивку выдач по классам (тензорные / глобальные / разделяемые / адресные / целые /
    перестановки / барьеры / ветвления / плавающие / пересылки);
  * АТРИБУЦИЮ ПО РОЛЯМ: находит ветвление по threadIdx в начале ядра, разводит блоки по
    достижимости и относит каждую выдачу к везущим (producer) или считающим (consumer);
  * БЮДЖЕТ: tau-1 свободных слотов на тензорную и сколько из них занято; потолок доли
    тензорного пика, задаваемый ОДНИМИ ТОЛЬКО слотами выдачи (учитывая варпов на планировщик);
  * вердикт "под счётной стеной / у стены / за стеной" с числом;
  * режим --diff: два ядра рядом, разница по классам и по ролям, взвешенная множителем роли;
  * СОВЕТЧИК: кандидаты на сокращение выдач с оценкой выигрыша.

ЧЕГО НЕ СЧИТАЕТ -- и это надо знать, иначе выводы будут неверны
  * ДИНАМИКУ. Счёт СТАТИЧЕСКИЙ: предикатные команды считаются выданными (они и правда занимают
    слот), но раскатанный цикл с недоказуемым числом итераций даёт статически больше команд, чем
    исполняется. Такие места помечаются "[РАСКАТАН]" -- их надо читать глазами.
  * НЕОДНОРОДНОСТЬ ЦЕНЫ ВЫДАЧИ. Слот один, но загрузка занимает MIO дольше команды сложения.
    Инструмент считает СЛОТЫ, а не время в очереди -- поэтому доля отставания, посчитанная по
    слотам, может разойтись с замером (у нас разошлась: см. --diff и раздел "сверка" в отчёте).
  * ЗАДЕРЖКИ И ГЛУБИНУ ЦЕПОЧЕК. Сознательно: §3b доказывает, что ни то, ни другое не входит в
    границу. Расстояние LDG->потребитель печатается только как признак незапрятанного рандеву.
  * БАНКОВЫЕ КОНФЛИКТЫ LDS, занятость, состояние L2, расхождение полос внутри варпа.
  * ЧИСЛО ИТЕРАЦИЙ мейнлупа (от формы задачи) -- все величины даны НА ОДНУ ИТЕРАЦИЮ.

ПОЧЕМУ ПЕРЕСТАНОВКА КОМАНД НЕ ПОМОГАЕТ (чтобы следующий не начал с неверной посылки)
  Задача сводится к modulo scheduling: при свободном
  опережении граница периода ЦЕЛИКОМ СЧЁТНАЯ, T >= nu+1, а задержка команды и глубина цепочки
  зависимостей НЕ ВХОДЯТ в неё вообще. Переставив те же команды, получим то же nu и то же время.
  Замером это подтверждено дважды: компилятор сам выносит зависимую команду на ~16 позиций вперёд,
  а у int8 против e5m2 время растёт (+5.0%) МЕНЬШЕ и работы (+8.8%), и суммы стоек (+6.4%).
  Работают ровно два хода:
    (1) СОКРАТИТЬ nu -- убрать выдачи;
    (2) ПЕРЕНЕСТИ выдачу из непокрытой роли в покрытую -- у везущих 1 варп на планировщик и
        свободный слот занять некем, у считающих 2 и они кроют друг друга; одна и та же выдача у
        везущих стоит в 1.18...1.94 раза дороже. Перенос выгоден, даже если суммарное nu вырастет.

ЗАПУСК
  python3 tools/issue_slots.py <lib.so|cubin> --kernel <regex>
  python3 tools/issue_slots.py <lib.so|cubin> --diff <regex_A> <regex_B>
  python3 tools/issue_slots.py <lib.so|cubin> --selftest      # сверка с ручным счётом 2026-07-31
  python3 tools/issue_slots.py <lib.so|cubin> --list           # перечислить ядра
  ключи: --tau (по умолчанию 2.0), --threads, --roles prod=4,cons=8, --role-branch 0x1070,
         --json out.json, --no-advice

ОТКУДА tau = 2 (в docs/VOLTA_SM70.md числа нет, выведено арифметикой, не замером)
  V100: 125.3 TFLOP/s тензорных при 1530 МГц на 80 SM -> 511 МАС/такт на SM -> 128 МАС/такт на
  планировщик (их 4). Одна SASS-команда HMMA.884.F32.F32 несёт 256 МАС на варп (проверено на нашем
  ядре: 512 HMMA на итерацию при 131072 МАС расчётной работы -> ровно 256). 256/128 = 2 такта.
  Значит НА ПИКЕ тензорной трубы свободен ровно ОДИН слот на HMMA (tau-1 = 1). Если ядро идёт не на
  пике, бюджет шире: tau_eff = 2 / (доля тензорного пика). Ключ --tau позволяет это подставить.

  ПРОВЕРКА tau НЕЗАВИСИМЫМ ЯДРОМ (а не подгонкой): у volta_fwd_block -- рукописного мейнлупа,
  ЗАМЕРЕННОГО на 95.2% тензорного пика, -- инструмент даёт nu = 0.95 при tau-1 = 1.00. То есть
  ядро, которое почти достигает пика, стоит ровно НА счётной границе, а не под ней и не за ней.
  Совпадение независимой величины (доля пика по секундомеру) со счётной (nu по SASS) и есть
  подтверждение, что tau = 2 -- верное число, а правило §3b -- верное правило.
  Для сравнения на той же машине: mem_eff-ядро cutlass nu = 12.0 (потолок 15%), наш
  двухблочный volta_fwd_i8 nu = 2.60 (потолок 56%), ядро ролей nu = 1.70 (потолок 63% с учётом
  везущих), обратный проход volta_bwd_block nu = 1.28 (потолок 88%).
"""

import argparse
import collections
import json
import os
import re
import shutil
import subprocess

# --------------------------------------------------------------------------------------------
# 0. поиск cuobjdump (на этой машине он в conda-окружении cuda128, в PATH его нет)
# --------------------------------------------------------------------------------------------
_CUOBJ_CANDIDATES = [
    "cuobjdump",
    os.path.expanduser("~/miniconda3/envs/cuda128/bin/cuobjdump"),
    os.path.expanduser("~/anaconda3/envs/cuda128/bin/cuobjdump"),
    "/usr/local/cuda/bin/cuobjdump",
]


def find_cuobjdump():
    for c in _CUOBJ_CANDIDATES:
        p = shutil.which(c) if os.path.basename(c) == c else (c if os.path.exists(c) else None)
        if p:
            return p
    raise SystemExit("cuobjdump не найден: положите его в PATH или поправьте _CUOBJ_CANDIDATES")


# --------------------------------------------------------------------------------------------
# 1. разбор SASS
# --------------------------------------------------------------------------------------------
_PRED = re.compile(r"^@(!?)(P[T0-9]+)\s+")
_INS = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?);")
_HEX = re.compile(r"^\s*/\* (0x[0-9a-f]{16}) \*/\s*$")


class Ins(object):
    __slots__ = ("addr", "pred", "predneg", "op", "base", "text", "dst", "srcs", "addrregs",
                 "stall", "yld", "imm", "cls")

    def __repr__(self):
        return "%#06x %s" % (self.addr, self.text)


def _norm_op(op):
    """IMAD.MOV/CS2R -- это пересылка, IMAD.IADD -- сложение, IMAD.SHL -- сдвиг.
    Не привести их к смыслу = получить абсурдные 'адресной арифметики на загрузку'."""
    if op.startswith("IMAD.MOV"):
        return "MOV"
    if op.startswith("IMAD.IADD"):
        return "IADD3"
    if op.startswith("IMAD.SHL"):
        return "SHF"
    if op.startswith("IMAD.X"):
        return "IADD3"
    if op.startswith("CS2R"):
        return "MOV"
    return op


CLASS_RULES = [
    ("тензорные",     r"^(HMMA|IMMA|BMMA|DMMA)"),
    ("загр.глоб",     r"^(LDG|LD|LDGSTS|LDGDEPBAR)\b"),
    ("зап.глоб",      r"^(STG|ST|RED|ATOM|ATOMG|ATOMS)\b"),
    ("загр.разд",     r"^(LDS|LDSM)\b"),
    ("зап.разд",      r"^STS\b"),
    ("разлив",        r"^(LDL|STL)\b"),
    ("конст/спец",    r"^(LDC|S2R|CS2R|R2P|P2R|S2UR)\b"),
    ("цел.арифм",     r"^(IADD3|IMAD|LEA|SHF|ISCADD|IMNMX|IABS|SGXT)\b"),
    ("цел.лог",       r"^(LOP3|LOP|POPC|FLO|BFE|BFI|BREV)\b"),
    ("перестановки",  r"^(PRMT|SHFL|MOVM)\b"),
    ("плавающие",     r"^(FADD|FMUL|FFMA|FSETP|FSEL|FMNMX|MUFU|F2F|F2I|I2F|I2I|HADD2|HMUL2|HFMA2|HSETP2|DADD|DMUL|DFMA|FCHK|FSET)\b"),
    ("пересылки",     r"^(MOV|SEL|PRED|P2R)\b"),
    ("барьеры",       r"^(BAR|MEMBAR|DEPBAR|WARPSYNC|NANOSLEEP|YIELD|ERRBAR)\b"),
    ("ветвл./пред",   r"^(BRA|BRX|BSSY|BSYNC|EXIT|CALL|RET|JMP|JMX|PLOP3|ISETP|PSETP|PBK|BREAK|SSY|BMOV|VOTE|VOTEU|NOP|PMTRIG)\b"),
]

TENSOR = "тензорные"
MEM_OPS = ("загр.глоб", "зап.глоб", "загр.разд", "зап.разд", "разлив")


def classify(op):
    for name, pat in CLASS_RULES:
        if re.match(pat, op):
            return name
    return "прочее"


def parse_sass(text):
    """-> OrderedDict{имя ядра: [Ins]}"""
    kernels = collections.OrderedDict()
    cur, body = None, []
    pending = None
    for line in text.split("\n"):
        m = re.search(r"Function : (\S+)", line)
        if m:
            if cur and body:
                kernels[cur] = body
            cur, body, pending = m.group(1), [], None
            continue
        m = _INS.match(line)
        if m and cur is not None:
            it = Ins()
            it.addr = int(m.group(1), 16)
            raw = m.group(2).strip()
            pm = _PRED.match(raw)
            if pm:
                it.predneg = pm.group(1) == "!"
                it.pred = pm.group(2)
                raw = _PRED.sub("", raw)
            else:
                it.pred, it.predneg = None, False
            it.text = raw
            it.op = _norm_op(raw.split()[0])
            # БАЗОВЫЙ мнемоник: суффиксы (BRA.U, BRA.DIV, CALL.REL.NOINC, RET.REL.NODEC, BAR.SYNC)
            # иначе рвут CFG молча -- обратная дуга мейнлупа у нас именно BRA.U.
            it.base = it.op.split(".")[0]
            it.cls = classify(it.op)
            # ".reuse" -- подсказка кэша операндов, а не часть имени регистра. Не снять её =
            # регистр "R0.reuse" не совпадёт с "R0", и весь анализ по регистрам молча развалится.
            ops = (raw.split(" ", 1)[1] if " " in raw else "").replace(".reuse", "")
            # адресные регистры -- то, что стоит в квадратных скобках
            it.addrregs = set()
            for br in re.findall(r"\[([^\]]*)\]", ops):
                if "0x0]" in br or br.startswith("0x"):   # c[0x0][0x160] -- константный банк
                    continue
                it.addrregs |= set(re.findall(r"\bR\d+\b", br))
            parts = [p.strip() for p in ops.split(",")]
            # приёмник: первый регистровый операнд, если команда пишет регистр
            it.dst = None
            if parts and re.match(r"^R\d+$", parts[0]) and parts[0] != "RZ":
                if it.cls not in ("зап.глоб", "зап.разд") or it.op.startswith("ATOM"):
                    it.dst = parts[0]
            if it.base in ("ISETP", "PSETP", "PLOP3", "FSETP", "HSETP2", "VOTE", "BAR", "MEMBAR",
                           "DEPBAR", "BSSY", "BSYNC", "BRA", "EXIT", "CALL", "RET", "NOP",
                           "WARPSYNC", "PMTRIG", "STS", "STG", "STL"):
                it.dst = None
            srcs = set(re.findall(r"\bR\d+\b", ops))
            if it.dst:
                # первое вхождение приёмника не источник; но HMMA/аккумуляторы читают его тоже
                srcs = set(re.findall(r"\bR\d+\b", ",".join(parts[1:])))
                if it.cls == TENSOR:
                    srcs |= {it.dst}
            it.srcs = {r for r in srcs if r != "RZ"}
            im = re.findall(r"(?<![\w\[])(-?0x[0-9a-f]+)(?![\w\]])", ops)
            it.imm = [int(x, 16) for x in im]
            it.stall, it.yld = None, None
            body.append(it)
            pending = it
            continue
        m = _HEX.match(line)
        if m and pending is not None and pending.stall is None:
            hi = int(m.group(1), 16)
            # управляющее слово Volta: биты 105..125 команды = биты 41..61 старшего слова
            pending.stall = (hi >> 41) & 0xF
            pending.yld = (hi >> 45) & 0x1
            pending = None
    if cur and body:
        kernels[cur] = body
    return kernels


def parse_setp_pred(ins):
    """ISETP.GE.AND P0, PT, R0, 0x80, PT -> ('P0','GE','R0',0x80)"""
    if not ins.base == "ISETP":
        return None
    parts = [p.strip() for p in ins.text.replace(".reuse", "").split(" ", 1)[1].split(",")]
    if len(parts) < 4:
        return None
    dstp = parts[0]
    cmp_ = ins.text.split()[0].split(".")[1] if "." in ins.text.split()[0] else "?"
    a, b = parts[2], parts[3]
    if not re.match(r"^R\d+$", a):
        return None
    if not re.match(r"^-?0x[0-9a-f]+$", b):
        return None
    return (dstp, cmp_, a, int(b, 16))


# --------------------------------------------------------------------------------------------
# 2. CFG, доминаторы, естественные циклы
# --------------------------------------------------------------------------------------------
UNCOND_END = ("EXIT", "RET", "BRX", "JMX")


def _branch_target(ins):
    if ins.base in ("BRA", "BSSY", "CALL", "JMP"):
        m = re.search(r"(0x[0-9a-f]+)\s*$", ins.text)
        if m:
            return int(m.group(1), 16)
    return None


class CFG(object):
    def __init__(self, body):
        self.body = body
        self.idx = {it.addr: i for i, it in enumerate(body)}
        leaders = {body[0].addr}
        for i, it in enumerate(body):
            t = _branch_target(it)
            if t is not None and t in self.idx and it.base in ("BRA", "CALL", "JMP"):
                leaders.add(t)
            if it.base in ("BRA", "JMP", "CALL", "BSYNC") or it.base in UNCOND_END:
                if i + 1 < len(body):
                    leaders.add(body[i + 1].addr)
        self.leaders = sorted(leaders)
        self.blocks = {}          # начальный адрес -> [Ins]
        self.order = []
        for k, a in enumerate(self.leaders):
            b = self.leaders[k + 1] if k + 1 < len(self.leaders) else None
            lo = self.idx[a]
            hi = self.idx[b] if b is not None else len(body)
            self.blocks[a] = body[lo:hi]
            self.order.append(a)
        self.succ = {a: [] for a in self.order}
        self.pred = {a: [] for a in self.order}
        for k, a in enumerate(self.order):
            blk = self.blocks[a]
            last = blk[-1]
            nxt = self.order[k + 1] if k + 1 < len(self.order) else None
            tgt = _branch_target(last)
            outs = []
            if last.base == "BRA":
                if tgt is not None:
                    outs.append(tgt)
                if last.pred is not None and nxt is not None:
                    outs.append(nxt)
            elif last.base in ("CALL",):
                if tgt is not None:
                    outs.append(tgt)
                if nxt is not None:
                    outs.append(nxt)
            elif last.base in UNCOND_END:
                if last.pred is not None and nxt is not None:
                    outs.append(nxt)
            else:
                if nxt is not None:
                    outs.append(nxt)
            # предикатные EXIT/BRA внутри блока (не последняя команда) -- дополнительные выходы
            for it in blk[:-1]:
                if it.base == "BRA" and it.pred is not None:
                    t2 = _branch_target(it)
                    if t2 is not None:
                        outs.append(t2)
            for o in outs:
                if o in self.succ:
                    self.succ[a].append(o)
                    self.pred[o].append(a)
        self.entry = self.order[0]
        self.dom = self._dominators()

    def _dominators(self):
        order = self.order
        rpo = self._rpo()
        pos = {a: i for i, a in enumerate(rpo)}
        idom = {self.entry: self.entry}

        def inter(a, b):
            while a != b:
                while pos[a] > pos[b]:
                    a = idom[a]
                while pos[b] > pos[a]:
                    b = idom[b]
            return a

        changed = True
        while changed:
            changed = False
            for a in rpo[1:]:
                new = None
                for p in self.pred[a]:
                    if p in idom:
                        new = p if new is None else inter(new, p)
                if new is not None and idom.get(a) != new:
                    idom[a] = new
                    changed = True
        # полное множество доминаторов
        dom = {}
        for a in order:
            s, x = set(), a
            while x in idom:
                s.add(x)
                if idom[x] == x:
                    break
                x = idom[x]
            dom[a] = s
        return dom

    def _rpo(self):
        seen, out = set(), []

        def dfs(a):
            stack = [(a, iter(self.succ[a]))]
            seen.add(a)
            while stack:
                node, it = stack[-1]
                adv = False
                for s in it:
                    if s not in seen:
                        seen.add(s)
                        stack.append((s, iter(self.succ[s])))
                        adv = True
                        break
                if not adv:
                    out.append(stack.pop()[0])
        dfs(self.entry)
        for a in self.order:
            if a not in seen:
                dfs(a)
        return out[::-1]

    def natural_loops(self):
        """-> [(header, set(blocks), backedge_addr)] -- только НАСТОЯЩИЕ обратные дуги
        (цель доминирует источник). Это отсекает вынесенные компилятором блоки, которые
        возвращаются назад BRA и на глаз выглядят как цикл."""
        loops = []
        for a in self.order:
            for s in self.succ[a]:
                if s in self.dom.get(a, ()):          # s доминирует a -> обратная дуга
                    body = {s}
                    stack = [a]
                    while stack:
                        n = stack.pop()
                        if n in body:
                            continue
                        body.add(n)
                        for p in self.pred[n]:
                            if p not in body:
                                stack.append(p)
                    loops.append((s, body, a))
        return loops

    def reachable(self, start, blocked=()):
        seen, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n in seen or n in blocked:
                continue
            seen.add(n)
            for s in self.succ.get(n, ()):
                stack.append(s)
        return seen


# --------------------------------------------------------------------------------------------
# 3. роли
# --------------------------------------------------------------------------------------------
class RoleSplit(object):
    def __init__(self, kind, blocks_a, blocks_b, warps_a, warps_b, name_a, name_b, note):
        self.kind = kind
        self.blocks = {name_a: blocks_a, name_b: blocks_b}
        self.warps = {name_a: warps_a, name_b: warps_b}
        self.names = (name_a, name_b)
        self.note = note


def taint_tid(body):
    """регистры, производные от threadIdx.x"""
    tainted = set()
    for it in body:
        if it.op == "S2R" and "SR_TID.X" in it.text and it.dst:
            tainted.add(it.dst)
    for _ in range(6):
        grew = False
        for it in body:
            if it.dst and it.cls in ("цел.арифм", "цел.лог", "пересылки", "перестановки", "конст/спец"):
                if it.srcs & tainted and it.dst not in tainted:
                    tainted.add(it.dst)
                    grew = True
        if not grew:
            break
    return tainted


def total_threads(body, override=None):
    if override:
        return override
    best = 0
    for it in body:
        if it.base == "BAR" and it.imm:
            # BAR.SYNC 0x1, 0x180 -- второй непосредственный операнд = число нитей в защёлке.
            # Это единственный надёжный источник числа нитей ВНУТРИ cubin (EIATTR_MAX_THREADS
            # лежит в .nv.info и к дизассемблеру не привязан).
            if len(it.imm) >= 2:
                best = max(best, it.imm[1])
    return best or None


def detect_roles(cfg, threads=None, force_addr=None, warps_cfg=None):
    body = cfg.body
    tainted = taint_tid(body)
    thr = total_threads(body, threads)
    cands = []
    for i, it in enumerate(body):
        p = parse_setp_pred(it)
        if not p:
            continue
        dstp, cmp_, reg, imm = p
        if reg not in tainted:
            continue
        if imm <= 0 or imm % 32 != 0:
            continue
        if thr and imm >= thr:
            continue
        # ищем ветвление под этим предикатом
        for j in range(i + 1, len(body)):
            b = body[j]
            if b.base == "BRA" and b.pred == dstp:
                cands.append((it, b, cmp_, imm))
                break
            if b.dst == dstp:
                break
    if force_addr is not None:
        cands = [c for c in cands if c[1].addr == force_addr]
    if not cands:
        return None

    def nins(bl):
        return sum(len(cfg.blocks[a]) for a in bl)

    def hmma_of(bl):
        return sum(1 for a in bl for it in cfg.blocks[a] if it.cls == TENSOR)

    # НЕ "самое раннее" ветвление: по threadIdx сравниваются и границы перекладок в прологе.
    # Верный признак -- ветвление, которое РАЗВОДИТ ЯДРО НАДВОЕ так, что тензорные команды
    # достаются ровно одной стороне. Иначе ролью объявится сторожевое условие цикла.
    scored = []
    for setp, bra, cmp_, imm in cands:
        blk_of = None
        for a in cfg.order:
            if any(x.addr == bra.addr for x in cfg.blocks[a]):
                blk_of = a
                break
        succ = cfg.succ.get(blk_of, [])
        tgt = _branch_target(bra)
        if tgt is None or len(succ) < 2:
            continue
        fall = [s for s in succ if s != tgt]
        if not fall:
            continue
        fall = fall[0]
        ra, rb = cfg.reachable(tgt), cfg.reachable(fall)
        et, ef = ra - rb, rb - ra
        if not et or not ef:
            continue
        ht, hf = hmma_of(et), hmma_of(ef)
        clean = 1 if (ht == 0) != (hf == 0) else 0     # тензорные ровно у одной стороны
        scored.append((clean, min(nins(et), nins(ef)), setp, bra, cmp_, imm, et, ef))
    if not scored:
        return None
    scored.sort(key=lambda s: (s[0], s[1]))
    clean, _, setp, bra, cmp_, imm, excl_t, excl_f = scored[-1]
    # какая сторона -- "нити < imm"?
    taken_is_low = None
    if cmp_ in ("GE",):
        taken_is_low = bra.predneg          # @!P (не >=) -> tid < imm
    elif cmp_ in ("LT",):
        taken_is_low = not bra.predneg
    if taken_is_low is None:
        taken_is_low = bra.predneg
    warps_low = imm // 32
    warps_high = (thr // 32 - warps_low) if thr else None
    if taken_is_low:
        blocks_low, blocks_high = excl_t, excl_f
    else:
        blocks_low, blocks_high = excl_f, excl_t

    def hmma(bl):
        return sum(1 for a in bl for it in cfg.blocks[a] if it.cls == TENSOR)

    if hmma(blocks_low) > hmma(blocks_high):
        cons, prod = (blocks_low, warps_low), (blocks_high, warps_high)
    else:
        cons, prod = (blocks_high, warps_high), (blocks_low, warps_low)
    if warps_cfg:
        prod = (prod[0], warps_cfg.get("prod", prod[1]))
        cons = (cons[0], warps_cfg.get("cons", cons[1]))
    note = ("роль задаётся ветвлением %#06x по threadIdx (%s %s 0x%x); "
            "нитей в блоке %s" % (bra.addr, parse_setp_pred(setp)[2], cmp_, imm, thr))
    return RoleSplit("auto", prod[0], cons[0], prod[1], cons[1], "везущие", "считающие", note)


# --------------------------------------------------------------------------------------------
# 4. метрики области
# --------------------------------------------------------------------------------------------
def width_of(it):
    if ".128" in it.text:
        return 16
    if ".64" in it.text:
        return 8
    if ".U16" in it.text or ".S16" in it.text or ".U8" in it.text or ".S8" in it.text:
        return 2
    return 4


def region_stats(cfg, blocks, tau=2.0):
    ins = [it for a in sorted(blocks) for it in cfg.blocks[a]]
    mix = collections.Counter(it.cls for it in ins)
    n = len(ins)
    h = mix.get(TENSOR, 0)
    st = {
        "n": n,
        "hmma": h,
        "nu": (n - h) / float(h) if h else None,
        "mix": dict(mix),
        "byop": dict(collections.Counter(it.op.split(".")[0] for it in ins)),
        "stallsum": sum(it.stall or 0 for it in ins),
        "blocks": sorted(blocks),
    }
    st["addr_slice"] = address_slice(cfg, blocks)
    st["ldg_bytes"] = sum(width_of(it) for it in ins if it.cls == "загр.глоб")
    return st


SLICE_CLASSES = ("цел.арифм", "цел.лог", "пересылки", "конст/спец")


def address_slice(cfg, blocks, only_regs=None, exclude_regs=None):
    """АДРЕСНЫЙ СРЕЗ: команды, чей результат доходит до операнда [адреса] обращения к памяти.

    Обратный поток данных С УБИЙСТВОМ ПРИ ПЕРЕОПРЕДЕЛЕНИИ. Без убийства (наивное "регистр R17
    когда-нибудь был адресом") срез раздувается в разы: в теле на 1400 команд и 128 регистрах
    каждый регистр переиспользуется под данные, и наивный счёт дал 421 команду вместо 19.
    Предикатные записи регистр НЕ убивают (могло не выполниться) -- это осознанный перебор в
    безопасную сторону.

    only_regs   -- считать только срез, питающий ЭТИ адресные регистры (для разбора по потокам);
    exclude_regs-- вычесть срез, который питает и другие обращения (тогда получается ИСКЛЮЧИТЕЛЬНАЯ
                   адресная арифметика потока: то, что исчезнет вместе с ним).
    """
    ins = [it for a in sorted(blocks) for it in cfg.blocks[a]]

    def walk(seed_pred):
        live, marked = set(), set()
        for _ in range(3):           # 3 прохода -- ради переносимых через итерацию регистров
            for it in reversed(ins):
                if it.dst and it.dst in live:
                    if it.pred is None:
                        live.discard(it.dst)
                    if it.cls in SLICE_CLASSES:
                        marked.add(it.addr)
                        live |= it.srcs
                    # если адрес пришёл из загрузки/HMMA -- это уже не адресная арифметика, стоп
                if it.cls in MEM_OPS and seed_pred(it):
                    live |= it.addrregs
        return marked

    if only_regs is None:
        return sorted(walk(lambda it: True))
    inc = walk(lambda it: bool(it.addrregs & only_regs))
    if exclude_regs:
        exc = walk(lambda it: bool(it.addrregs & exclude_regs))
        inc = inc - exc
    return sorted(inc)


def pick_mainloop(cfg, blocks=None, prefer_tensor=True):
    """мейнлуп = естественный цикл с наибольшим числом тензорных команд;
    если тензорных нет вовсе -- самый крупный по числу команд."""
    loops = cfg.natural_loops()
    best, bestkey = None, None
    for hdr, body, be in loops:
        if blocks is not None:
            body = body & blocks
            if not body or hdr not in blocks:
                continue
        h = sum(1 for a in body for it in cfg.blocks[a] if it.cls == TENSOR)
        n = sum(len(cfg.blocks[a]) for a in body)
        key = (h, n) if prefer_tensor else (0, n)
        if bestkey is None or key > bestkey:
            best, bestkey = (hdr, body, be), key
    return best


def inner_loops(cfg, body_blocks, header):
    out = []
    for hdr, body, be in cfg.natural_loops():
        if hdr != header and hdr in body_blocks and body <= body_blocks:
            out.append((hdr, body, be))
    # только максимальные вложенные
    res = []
    for h, b, e in out:
        if not any(h != h2 and b < b2 for h2, b2, _ in out):
            res.append((h, b, e))
    return res


# --------------------------------------------------------------------------------------------
# 5. печать
# --------------------------------------------------------------------------------------------
SCHEDULERS = 4       # планировщиков варпов на SM у Volta
ROLE_MULT = (1.18, 1.94)   # замеренный множитель цены выдачи у НЕПОКРЫТОЙ роли (§3b)


def fmt_mix(mix, n):
    rows = sorted(mix.items(), key=lambda kv: -kv[1])
    return "\n".join("      %-16s %6d  %5.1f%%" % (k, v, 100.0 * v / n) for k, v in rows if v)


def verdict(nu, tau):
    free = tau - 1.0
    if nu is None:
        return "нет тензорных команд -- счёт по слотам к этой области не применим"
    if nu <= free * 0.95:
        return "ПОД счётной стеной (nu=%.2f <= tau-1=%.2f): свободные слоты есть, лишняя работа падает в них" % (nu, free)
    if nu <= free * 1.05:
        return "У СТЕНЫ (nu=%.2f ~ tau-1=%.2f): каждая следующая выдача уже платится тактом" % (nu, free)
    return "ЗА СТЕНОЙ (nu=%.2f > tau-1=%.2f): выдачи связывают, потолок доли тензорного пика = %.1f%%" % (
        nu, free, 100.0 * tau / (1.0 + nu))


def report_kernel(name, cfg, tau, roles, args):
    print("=" * 100)
    print("ЯДРО %s" % name)
    print("  команд всего: %d, базовых блоков: %d" % (len(cfg.body), len(cfg.order)))
    out = {"kernel": name, "tau": tau, "roles": {}}
    if roles is None:
        print("  роли: НЕ НАЙДЕНЫ (ветвления по threadIdx с порогом, кратным 32, нет) -- считаю ядро целиком")
        regions = [("всё ядро", set(cfg.order), None)]
    else:
        print("  роли: %s" % roles.note)
        regions = []
        for rn in roles.names:
            regions.append((rn, roles.blocks[rn], roles.warps[rn]))
    sched = {}
    for rn, blks, warps in regions:
        ml = pick_mainloop(cfg, blks if roles else None)
        if ml is None:
            print("\n  --- %s: мейнлуп не найден (обратных дуг нет)" % rn)
            continue
        hdr, lb, be = ml
        st = region_stats(cfg, lb, tau)
        inner = inner_loops(cfg, lb, hdr)
        print("\n  --- РОЛЬ %s%s" % (rn.upper(), ("  (варпов %s, на планировщик %.2f)" %
              (warps, warps / float(SCHEDULERS))) if warps else ""))
        print("      мейнлуп: шапка %#06x, обратная дуга %#06x, блоков %d, команд %d"
              % (hdr, be, len(lb), st["n"]))
        print("      выдач в мейнлупе: %d, из них тензорных: %d" % (st["n"], st["hmma"]))
        if st["hmma"]:
            print("      nu = %.3f  нетензорных выдач на тензорную" % st["nu"])
            print("      бюджет: tau-1 = %.2f свободных слотов на HMMA, занято %.2f (%.0f%%)"
                  % (tau - 1, st["nu"], 100.0 * st["nu"] / max(tau - 1, 1e-9)))
        print("      %s" % verdict(st["nu"], tau))
        print("      состав выдач:")
        print(fmt_mix(st["mix"], st["n"]))
        nmem = sum(st["mix"].get(k, 0) for k in MEM_OPS)
        print("      адресный срез (команды, чей результат доходит до [адреса]): %d"
              % len(st["addr_slice"]))
        if nmem:
            print("      обращений к памяти %d -> адресных команд на обращение: %.2f"
                  % (nmem, len(st["addr_slice"]) / float(nmem)))
        print("      сумма закодированных тактов стойла: %d  "
              "[СПРАВОЧНО: §3b -- суммировать их БЕСПОЛЕЗНО, это не период]" % st["stallsum"])
        for h2, b2, e2 in inner:
            n2 = sum(len(cfg.blocks[a]) for a in b2)
            ldg2 = sum(1 for a in b2 for it in cfg.blocks[a] if it.cls == "загр.глоб")
            if is_shfl_convergence(cfg, b2):
                print("      [петля сходимости SHFL] %#06x..%#06x: %d команд -- это обвязка "
                      "межполосного обмена, а не цикл алгоритма" % (h2, e2, n2))
            else:
                print("      [РАСКАТАН] вложенный цикл %#06x..%#06x: %d команд (%d глоб.загрузок) -- "
                      "число итераций статически неизвестно, в динамике может быть 1"
                      % (h2, e2, n2, ldg2))
        sched[rn] = (st, warps)
        out["roles"][rn] = {"warps": warps, "header": hdr, "n": st["n"], "hmma": st["hmma"],
                            "nu": st["nu"], "mix": st["mix"], "addr_slice": len(st["addr_slice"]),
                            "stallsum": st["stallsum"],
                            "inner_rolled": [[h2, e2] for h2, _, e2 in inner]}
    # сводка по планировщику
    if sched and all(w for _, w in sched.values()):
        slots = sum(st["n"] * w / float(SCHEDULERS) for st, w in sched.values())
        tcyc = sum(st["hmma"] * tau * w / float(SCHEDULERS) for st, w in sched.values())
        print("\n  === БЮДЖЕТ НА ПЛАНИРОВЩИК (на одну плитку мейнлупа) ===")
        for rn, (st, w) in sched.items():
            print("      %-12s %6.1f слотов  (%d команд x %.2f варпа на планировщик)"
                  % (rn, st["n"] * w / float(SCHEDULERS), st["n"], w / float(SCHEDULERS)))
        print("      ИТОГО слотов выдачи: %.0f" % slots)
        print("      тактов тензорной трубы при tau=%.1f: %.0f" % (tau, tcyc))
        if slots > 0:
            ceil = tcyc / slots
            print("      ПОТОЛОК доли тензорного пика ТОЛЬКО по слотам выдачи: %.1f%%" % (100 * ceil))
            out["ceiling_issue_bound"] = ceil
            out["slots_per_sched"] = slots
            out["tensor_cycles_per_sched"] = tcyc
    return out


# --------------------------------------------------------------------------------------------
# 6. советчик
# --------------------------------------------------------------------------------------------
def is_shfl_convergence(cfg, blocks):
    """Мелкая петля из VOTE/WARPSYNC/SHFL/YIELD -- это петля СХОДИМОСТИ варпа, которую ptxas
    ставит вокруг межполосного обмена, а не цикл алгоритма. Считать её раскатанной -- шум."""
    ins = [it for a in sorted(blocks) for it in cfg.blocks[a]]
    if len(ins) > 12:
        return False
    bases = {it.base for it in ins}
    return bool(bases & {"VOTE", "WARPSYNC", "SHFL", "YIELD"}) and not (bases & {"LDG", "STG", "HMMA"})


def mem_streams(cfg, blocks):
    """Число РАЗЛИЧНЫХ потоков обращений: обращения, делящие базовый регистр адреса,
    двигаются ОДНИМ инкрементом на итерацию. Столько адресных команд и достаточно."""
    ins = [it for a in sorted(blocks) for it in cfg.blocks[a]]
    keys = set()
    for it in ins:
        if it.cls in MEM_OPS and it.addrregs:
            keys.add(",".join(sorted(it.addrregs)))
    return max(len(keys), 1)


def advise(name, cfg, roles, tau, out):
    print("\n  === СОВЕТЧИК: кандидаты на сокращение выдач ===")
    if roles is None:
        print("      роли не разведены -- советы по ролям недоступны")
    cands = []
    regions = ([(rn, roles.blocks[rn], roles.warps[rn]) for rn in roles.names]
               if roles else [("всё ядро", set(cfg.order), None)])
    per_role = {}
    for rn, blks, warps in regions:
        ml = pick_mainloop(cfg, blks if roles else None)
        if ml is None:
            continue
        hdr, lb, be = ml
        ins = [it for a in sorted(lb) for it in cfg.blocks[a]]
        st = region_stats(cfg, lb, tau)
        per_role[rn] = (st, warps, lb, ins)
        wps = (warps or SCHEDULERS) / float(SCHEDULERS)
        # цена одной выдачи роли в слотах на планировщик (у везущих слот прикрыть некем)
        price = wps * (ROLE_MULT[1] if wps < 1.5 else 1.0)

        # --- A1. адресная арифметика, сворачиваемая в шаг по базе
        nmem = sum(st["mix"].get(k, 0) for k in MEM_OPS)
        na = len(st["addr_slice"])
        nstream = mem_streams(cfg, lb)
        if nmem and na > nstream + 4:
            save = na - nstream
            cands.append((save * price, rn,
                          "A1 адресная арифметика -> шаг по базе",
                          "в мейнлупе %d команд адресного среза на %d обращений, а РАЗЛИЧНЫХ потоков "
                          "обращений всего %d. Если каждый поток двигать одним инкрементом на "
                          "итерацию, адресных команд хватит %d -> минус до %d выдач на итерацию "
                          "(%.1f%% выдач роли). Смотреть на прямую адресацию с константным шагом."
                          % (na, nmem, nstream, nstream, save, 100.0 * save / st["n"])))

        # --- A2/A3. второй поток подачи: мелкая группа LDG на отдельной базе
        clusters = ldg_clusters(cfg, lb)
        tot_b = sum(c["bytes"] for c in clusters)
        small = [c for c in sorted(clusters, key=lambda c: c["bytes"])
                 if tot_b and c["bytes"] <= 0.10 * tot_b]
        if len(clusters) > 1 and small:
            n_i = sum(c["n"] for c in small)
            n_a = sum(c["addr"] for c in small)
            n_b = sum(c["bytes"] for c in small)
            issues = n_i + n_a
            cands.append((issues * price, rn,
                          "A2 второй поток подачи -> совместить с идущей загрузкой ИЛИ свернуть в ранг-1",
                          "ЭВРИСТИКА (группировка по базовому регистру адреса, полная раскрутка её "
                          "дробит -- читать глазами): %d глобальных загрузок на %d отдельных базах "
                          "везут %d Б = %.1f%% трафика роли, "
                          "но стоят %d выдач (%d загрузок + %d ИСКЛЮЧИТЕЛЬНО их адресной арифметики). "
                          "Два лечения, оба убивают поток ЦЕЛИКОМ: (а) положить эти данные РЯДОМ с "
                          "основными, чтобы они приезжали той же выдачей; (б) сделать величину на БЛОК, "
                          "а не на позицию -- тогда она сворачивается в ранг-1 вне ядра. Выигрыш до "
                          "%d выдач на плитку у роли '%s'."
                          % (n_i, len(small), n_b, 100.0 * n_b / tot_b, issues, n_i, n_a, issues, rn)))

        # --- A8. УЗКИЕ обращения: одна выдача везёт 4 Б там, где могла бы 16
        wid = collections.Counter()
        for it in ins:
            if it.cls in ("загр.глоб", "загр.разд", "зап.разд", "зап.глоб"):
                wid[(it.cls, width_of(it))] += 1
        for cls in ("загр.глоб", "загр.разд", "зап.разд"):
            narrow = wid.get((cls, 4), 0) + wid.get((cls, 2), 0)
            wide = wid.get((cls, 16), 0)
            tot = sum(v for (c, _), v in wid.items() if c == cls)
            if narrow >= 8 and narrow >= 0.20 * tot:
                bytes_n = wid.get((cls, 4), 0) * 4 + wid.get((cls, 2), 0) * 2
                ideal = (bytes_n + 15) // 16
                save = narrow - ideal
                cands.append((save * price, rn,
                              "A8 узкие обращения (%s) -> одна широкая выдача вместо четырёх" % cls,
                              "%d из %d обращений класса '%s' узкие (4 Б и меньше) и везут всего %d Б. "
                              "Одна выдача .128 берёт 16 Б, значит тех же данных хватило бы %d выдач "
                              "-> минус до %d выдач на итерацию (%.1f%% выдач роли). Слот считается "
                              "ОДИН и за 4 Б, и за 16 -- поэтому ширина здесь и есть сокращение nu. "
                              "Проверять ВЫРАВНИВАНИЕ шага строки: не кратен 16 -- отказ, а не "
                              "замедление. (Широких уже %d.)%s"
                              % (narrow, tot, cls, bytes_n, ideal, save,
                                 100.0 * save / st["n"], wide,
                                 (" ОГОВОРКА: у n-мажорного операнда широкая LDS ЗАПРЕЩЕНА "
                                  "раскладкой mma.884 -- сверить с §4v, прежде чем считать выигрышем.")
                                 if cls == "загр.разд" else "")))

        # --- A7. крупнейшая нетензорная статья у роли с тензорными командами
        if st["hmma"]:
            rest = [(k, v) for k, v in st["mix"].items() if k != TENSOR]
            rest.sort(key=lambda kv: -kv[1])
            if rest:
                k0, v0 = rest[0]
                dnu = v0 / float(st["hmma"])
                cands.append((0, rn, "A7 крупнейшая нетензорная статья: %s" % k0,
                              "%d выдач = %.2f из nu=%.2f (%.0f%% всей вспомогательной работы роли). "
                              "Каждая убранная выдача на одну HMMA поднимает потолок доли тензорного "
                              "пика на %.2f процентного пункта. Убрать статью целиком -> потолок "
                              "%.1f%% вместо %.1f%%."
                              % (v0, dnu, st["nu"], 100.0 * v0 / (st["n"] - st["hmma"]),
                                 100.0 * (tau / (1 + st["nu"] - 1.0 / st["hmma"]) - tau / (1 + st["nu"])),
                                 100.0 * tau / (1 + st["nu"] - dnu), 100.0 * tau / (1 + st["nu"]))))

        # --- A5. незапрятанная задержка на рандеву
        risky = ldg_before_barrier(ins)
        if risky:
            cands.append((0, rn, "A5 подача упирается в рандеву (это НЕ сокращение выдач)",
                          "%d глобальных загрузок стоят ближе 24 команд до BAR.SYNC или до конца тела: "
                          "их задержка ложится ПРЯМО на защёлку, и цена тут не в слоте, а в ожидании. "
                          "Лечится ранним запуском (пролог/предвыборка на плитку вперёд), а НЕ "
                          "перестановкой внутри тела -- перестановка периода не меняет (§3b)."
                          % len(risky)))

        # --- A6. раскатанный цикл с неизвестным числом итераций
        for h2, b2, e2 in inner_loops(cfg, lb, hdr):
            if is_shfl_convergence(cfg, b2):
                continue
            n2 = sum(len(cfg.blocks[a]) for a in b2)
            ldg2 = sum(1 for a in b2 for it in cfg.blocks[a] if it.cls == "загр.глоб")
            cands.append((0, rn, "A6 раскатанный цикл с недоказуемым числом итераций",
                          "цикл %#06x..%#06x на %d команд (%d глоб.загрузок): компилятор не доказал, "
                          "что итерация одна, и раскатал его со сторожевой копией. Статический счёт "
                          "здесь ЗАВЫШЕН, но в динамике остаётся хвост адресной арифметики и лишняя "
                          "проверка. Лечится тем, что индекс делают заведомо однопроходным."
                          % (h2, e2, n2, ldg2)))

    # --- A4. перенос из везущих в считающие
    if roles and all(rn in per_role for rn in roles.names):
        (stp, wp, _, _) = per_role["везущие"]
        (stc, wc, _, _) = per_role["считающие"]
        if wp and wc:
            pp = stp["n"] * wp / float(SCHEDULERS)
            cc = stc["n"] * wc / float(SCHEDULERS)
            cands.append((0, "везущие",
                          "A4 перенос выдач из везущих в считающие",
                          "у везущих %.0f слотов на планировщик против %.0f у считающих, но каждый их "
                          "слот стоит в %.2f...%.2f раза дороже (варпов на планировщик %.2f против "
                          "%.2f -- прикрывать некем): взвешенная цена везущих %.0f...%.0f слотов. "
                          "Перенос выгоден, ДАЖЕ ЕСЛИ суммарное число выдач вырастет. Первые "
                          "кандидаты на перенос -- адресная арифметика подачи и укладка мелких таблиц."
                          % (pp, cc, ROLE_MULT[0], ROLE_MULT[1], wp / float(SCHEDULERS),
                             wc / float(SCHEDULERS), pp * ROLE_MULT[0], pp * ROLE_MULT[1])))
    cands.sort(key=lambda c: -c[0])
    if not cands:
        print("      кандидатов не найдено")
    for k, (score, rn, title, why) in enumerate(cands, 1):
        tag = ("~%.0f слотов/пл." % score) if score else "оценка не в слотах"
        print("      %d) [%s] %s   [%s]" % (k, rn, title, tag))
        for ln in wrap(why, 92):
            print("         %s" % ln)
    out["advice"] = [{"score": c[0], "role": c[1], "title": c[2], "why": c[3]} for c in cands]


def wrap(s, w):
    words, line, out = s.split(), "", []
    for x in words:
        if len(line) + len(x) + 1 > w:
            out.append(line)
            line = x
        else:
            line = (line + " " + x).strip()
    if line:
        out.append(line)
    return out


def base_pointers(cfg, blocks):
    """Различные базовые регистры, участвующие в адресах обращений."""
    ins = [it for a in sorted(blocks) for it in cfg.blocks[a]]
    bases = set()
    for it in ins:
        if it.cls in MEM_OPS:
            bases |= it.addrregs
    return bases


def ldg_clusters(cfg, blocks):
    """Группирует глобальные загрузки по базовому регистру адреса."""
    ins = [it for a in sorted(blocks) for it in cfg.blocks[a]]
    by = collections.defaultdict(lambda: {"n": 0, "bytes": 0, "regs": set()})
    for it in ins:
        if it.cls != "загр.глоб":
            continue
        key = ",".join(sorted(it.addrregs)) or "?"
        by[key]["n"] += 1
        by[key]["bytes"] += width_of(it)
        by[key]["regs"] |= it.addrregs
    out = []
    allregs = set()
    for v in by.values():
        allregs |= v["regs"]
    for k, v in by.items():
        others = allregs - v["regs"]
        sl = address_slice(cfg, blocks, only_regs=v["regs"], exclude_regs=others)
        out.append({"base": k, "n": v["n"], "bytes": v["bytes"], "addr": len(sl),
                    "regs": v["regs"]})
    return out


def ldg_before_barrier(ins, window=24):
    """Глобальные загрузки, до ближайшего BAR.SYNC от которых меньше `window` команд."""
    out = []
    bars = [i for i, it in enumerate(ins) if it.base == "BAR"]
    n = len(ins)
    for i, it in enumerate(ins):
        if it.cls != "загр.глоб":
            continue
        nxt = next((b for b in bars if b > i), None)
        d = (nxt - i) if nxt is not None else (n - i)
        if d < window:
            out.append((it.addr, d))
    return out


# --------------------------------------------------------------------------------------------
# 7. режим сравнения
# --------------------------------------------------------------------------------------------
def diff(nameA, cfgA, rolesA, nameB, cfgB, rolesB, tau):
    print("=" * 100)
    print("СРАВНЕНИЕ  A = %s" % nameA)
    print("           B = %s" % nameB)
    res = {"A": nameA, "B": nameB, "roles": {}}
    names = rolesA.names if rolesA else ("всё ядро",)
    tot = {}
    for rn in names:
        mlA = pick_mainloop(cfgA, rolesA.blocks[rn] if rolesA else None)
        mlB = pick_mainloop(cfgB, rolesB.blocks[rn] if rolesB else None)
        if not mlA or not mlB:
            continue
        stA = region_stats(cfgA, mlA[1], tau)
        stB = region_stats(cfgB, mlB[1], tau)
        wA = rolesA.warps[rn] if rolesA else None
        print("\n  --- РОЛЬ %s  (варпов %s)" % (rn.upper(), wA))
        print("      %-22s %10s %10s %10s" % ("", "A", "B", "A-B"))
        print("      %-22s %10d %10d %+10d" % ("выдач в мейнлупе", stA["n"], stB["n"], stA["n"] - stB["n"]))
        print("      %-22s %10d %10d %+10d" % ("тензорных", stA["hmma"], stB["hmma"], stA["hmma"] - stB["hmma"]))
        if stA["hmma"] and stB["hmma"]:
            print("      %-22s %10.3f %10.3f %+10.3f" % ("nu", stA["nu"], stB["nu"], stA["nu"] - stB["nu"]))
        keys = sorted(set(stA["mix"]) | set(stB["mix"]))
        for k in keys:
            a, b = stA["mix"].get(k, 0), stB["mix"].get(k, 0)
            if a != b:
                print("      %-22s %10d %10d %+10d" % ("  " + k, a, b, a - b))
        # разбивка по опкодам для дельты
        oa, ob = stA["byop"], stB["byop"]
        d = {k: oa.get(k, 0) - ob.get(k, 0) for k in set(oa) | set(ob)}
        d = {k: v for k, v in d.items() if v}
        if d:
            print("      по опкодам: " + ", ".join("%s %+d" % (k, v) for k, v in
                                                   sorted(d.items(), key=lambda kv: -abs(kv[1]))))
        print("      адресный срез: A=%d  B=%d  (A-B = %+d)"
              % (len(stA["addr_slice"]), len(stB["addr_slice"]),
                 len(stA["addr_slice"]) - len(stB["addr_slice"])))
        tot[rn] = (stA, stB, wA)
        res["roles"][rn] = {"A": {"n": stA["n"], "hmma": stA["hmma"], "nu": stA["nu"],
                                  "mix": stA["mix"], "addr": len(stA["addr_slice"])},
                            "B": {"n": stB["n"], "hmma": stB["hmma"], "nu": stB["nu"],
                                  "mix": stB["mix"], "addr": len(stB["addr_slice"])},
                            "warps": wA, "delta": stA["n"] - stB["n"]}
    if not tot:
        return res
    print("\n  === ЧЬИ ВЫДАЧИ И СКОЛЬКО ОНИ СТОЯТ ===")
    print("      %-12s %8s %8s %10s %10s %14s"
          % ("роль", "дельта", "варпов", "варп/пл.", "слоты/пл.", "взвеш. цена"))
    tot_slots_A = tot_slots_B = 0.0
    wsum = {}
    for rn, (stA, stB, w) in tot.items():
        wps = (w or SCHEDULERS) / float(SCHEDULERS)
        dslots = (stA["n"] - stB["n"]) * wps
        mult = ROLE_MULT if wps < 1.5 else (1.0, 1.0)
        wsum[rn] = (dslots * mult[0], dslots * mult[1])
        print("      %-12s %+8d %8d %10.2f %+10.1f %7.1f...%-7.1f"
              % (rn, stA["n"] - stB["n"], (stA["n"] - stB["n"]) * (w or SCHEDULERS), wps,
                 dslots, wsum[rn][0], wsum[rn][1]))
        tot_slots_A += stA["n"] * wps
        tot_slots_B += stB["n"] * wps
    print("      (столбец 'варпов' = дельта x число варпов роли: столько ЛИШНИХ варповых команд")
    print("       добавлено на SM. Именно по нему нормируют замеренный множитель роли 1.18...1.94.)")
    lo = sum(v[0] for v in wsum.values())
    hi = sum(v[1] for v in wsum.values())
    if abs(lo) > 1e-9:
        print("\n      доли отставания ПО СЛОТАМ (множитель роли %.2f...%.2f у непокрытых):"
              % ROLE_MULT)
        for rn in wsum:
            p_lo = 100.0 * wsum[rn][0] / lo if lo else 0
            p_hi = 100.0 * wsum[rn][1] / hi if hi else 0
            print("        %-12s %5.0f%% ... %.0f%%" % (rn, min(p_lo, p_hi), max(p_lo, p_hi)))
    tcA = sum(stA["hmma"] * tau * ((w or SCHEDULERS) / float(SCHEDULERS)) for stA, _, w in tot.values())
    print("\n      слотов выдачи на планировщик: A=%.0f  B=%.0f  -> A дороже на %.1f%%"
          % (tot_slots_A, tot_slots_B, 100.0 * (tot_slots_A / tot_slots_B - 1)))
    print("      тактов тензорной трубы (tau=%.1f): %.0f" % (tau, tcA))
    print("      потолок доли тензорного пика по слотам:  A=%.1f%%   B=%.1f%%"
          % (100 * tcA / tot_slots_A, 100 * tcA / tot_slots_B))
    print("      => ЕСЛИ БЫ ядро упиралось в слоты, A было бы медленнее B ровно на %.1f%%."
          % (100.0 * (tot_slots_A / tot_slots_B - 1)))
    print("         Меньшая фактическая разница = ядро ПОД счётной стеной; большая = что-то, кроме")
    print("         слотов (очередь MIO, задержка загрузки на рандеву), и это счётом не ловится.")
    res["slots_A"] = tot_slots_A
    res["slots_B"] = tot_slots_B
    res["issue_bound_penalty"] = tot_slots_A / tot_slots_B - 1
    return res


# --------------------------------------------------------------------------------------------
# 8. САМОПРОВЕРКА: инструмент обязан НЕЗАВИСИМО воспроизвести ручной счёт 2026-07-31
#    (docs/VOLTA_SM70.md §3b, пример "почему int8 отстаёт от e5m2 на 5-7%").
# --------------------------------------------------------------------------------------------
KNOWN = {                       # что было посчитано РУКАМИ
    "nu_e5m2": 1.41,
    "nu_i8": 1.67,
    "cons_extra_arith": 128,    # +128 арифметики распаковки у СЧИТАЮЩИХ
    "prod_extra_ldg": 14,       # +14 глобальных загрузок у ВЕЗУЩИХ
    "prod_extra_addr": 22,      # +22 адресных у ВЕЗУЩИХ
    "prod_extra_total": 36,
    "share_arith": 0.35,        # доли отставания по ЗАМЕРУ
    "share_scales": 0.65,
}
SELFTEST_A = r"fwd_wsILi32ELi64ELi256ELi2ELi4ELb0ELi64ELi1ELi1ELi256ELi0E"   # int8 + масштабы
SELFTEST_B = r"fwd_wsILi32ELi64ELi256ELi2ELi4ELb0ELi64ELi1ELi1ELi256ELi1E"   # e5m2


def selftest(path, tau):
    ka = load(path, SELFTEST_A)
    kb = load(path, SELFTEST_B)
    if not ka or not kb:
        raise SystemExit("самопроверка: не нашёл ядра ролей attn_fwd_volta_i8 / _e5m2 в %s" % path)
    ca, cb = CFG(list(ka.values())[0]), CFG(list(kb.values())[0])
    ra, rb = detect_roles(ca), detect_roles(cb)
    if ra is None or rb is None:
        raise SystemExit("самопроверка: роли не разведены -- разбор SASS сломан")
    stat = {}
    for tag, cfg, roles in (("A", ca, ra), ("B", cb, rb)):
        for rn in roles.names:
            ml = pick_mainloop(cfg, roles.blocks[rn])
            stat[(tag, rn)] = region_stats(cfg, ml[1], tau)
    nu_i8 = stat[("A", "считающие")]["nu"]
    nu_e5 = stat[("B", "считающие")]["nu"]
    d_arith = (stat[("A", "считающие")]["mix"].get("цел.лог", 0)
               - stat[("B", "считающие")]["mix"].get("цел.лог", 0))
    d_ldg = (stat[("A", "везущие")]["mix"].get("загр.глоб", 0)
             - stat[("B", "везущие")]["mix"].get("загр.глоб", 0))
    d_prod = stat[("A", "везущие")]["n"] - stat[("B", "везущие")]["n"]
    d_addr = len(stat[("A", "везущие")]["addr_slice"]) - len(stat[("B", "везущие")]["addr_slice"])
    d_cons = stat[("A", "считающие")]["n"] - stat[("B", "считающие")]["n"]
    wA = ra.warps["везущие"] / float(SCHEDULERS)
    wC = ra.warps["считающие"] / float(SCHEDULERS)
    sh_prod_lo = d_prod * wA * ROLE_MULT[0]
    sh_prod_hi = d_prod * wA * ROLE_MULT[1]
    sh_cons = d_cons * wC
    tot_lo, tot_hi = sh_prod_lo + sh_cons, sh_prod_hi + sh_cons
    rows = [
        ("nu у e5m2",                     "%.2f" % KNOWN["nu_e5m2"], "%.3f" % nu_e5,
         abs(nu_e5 - KNOWN["nu_e5m2"]) <= 0.05),
        ("nu у int8",                     "%.2f" % KNOWN["nu_i8"],   "%.3f" % nu_i8,
         abs(nu_i8 - KNOWN["nu_i8"]) <= 0.05),
        ("+арифметика у считающих",       "%d" % KNOWN["cons_extra_arith"], "%d" % d_arith,
         d_arith == KNOWN["cons_extra_arith"]),
        ("+загрузки у везущих",           "%d" % KNOWN["prod_extra_ldg"],  "%d" % d_ldg,
         d_ldg == KNOWN["prod_extra_ldg"]),
        ("+адресные у везущих",           "%d" % KNOWN["prod_extra_addr"], "%d" % d_addr,
         abs(d_addr - KNOWN["prod_extra_addr"]) <= 10),
        ("+всего у везущих",              "%d" % KNOWN["prod_extra_total"], "%d" % d_prod,
         abs(d_prod - KNOWN["prod_extra_total"]) <= 10),
        ("доля отставания: масштабы",     "%.0f%%" % (100 * KNOWN["share_scales"]),
         "%.0f...%.0f%%" % (100 * sh_prod_lo / tot_lo, 100 * sh_prod_hi / tot_hi), False),
        ("доля отставания: арифметика",   "%.0f%%" % (100 * KNOWN["share_arith"]),
         "%.0f...%.0f%%" % (100 * sh_cons / tot_hi, 100 * sh_cons / tot_lo), False),
    ]
    print("=" * 100)
    print("САМОПРОВЕРКА: ручной счёт 2026-07-31 против инструмента")
    print("  A (int8+масштабы) = %s" % list(ka.keys())[0][:88])
    print("  B (e5m2)          = %s" % list(kb.keys())[0][:88])
    print("  %-32s %14s %18s   %s" % ("величина", "руками", "инструмент", "сходится"))
    ok = True
    for nm, man, got, good in rows:
        print("  %-32s %14s %18s   %s" % (nm, man, got, "да" if good else "НЕТ"))
        ok = ok and good
    print()
    print("  Первые ШЕСТЬ строк -- счёт по SASS, и они воспроизводятся независимо.")
    print("  Последние ДВЕ -- нет, и это не сбой инструмента, а ГРАНИЦА СЧЁТА ПО СЛОТАМ:")
    print("  по числу выдач масштабы дают меньшую долю, чем арифметика распаковки, ДАЖЕ с")
    print("  множителем роли 1.94. Замеренные 65% против 35% счётом не выводятся -- значит цена")
    print("  масштабов не в числе слотов, а в том, что это ЗАГРУЗКИ на критическом пути подачи")
    print("  (§3b сам называет это честной незакрытой частью).")
    return ok


# --------------------------------------------------------------------------------------------
def load(path, kregex=None):
    exe = find_cuobjdump()
    txt = subprocess.run([exe, "--dump-sass", path], capture_output=True, text=True).stdout
    ks = parse_sass(txt)
    if kregex:
        ks = collections.OrderedDict((k, v) for k, v in ks.items() if re.search(kregex, k))
    return ks


def main():
    ap = argparse.ArgumentParser(description="учёт по слотам выдачи, sm_70 (docs/VOLTA_SM70.md §3b)")
    ap.add_argument("binary", help=".so или .cubin")
    ap.add_argument("--kernel", help="regex по имени ядра")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="сравнить два ядра (regex)")
    ap.add_argument("--tau", type=float, default=2.0,
                    help="тактов тензорной трубы на одну SASS HMMA (по умолчанию 2.0 -- пик V100)")
    ap.add_argument("--threads", type=int, help="нитей в блоке, если не выводится из BAR.SYNC")
    ap.add_argument("--roles", help="ручная разметка: prod=4,cons=8")
    ap.add_argument("--role-branch", help="адрес ветвления ролей, шестнадцатеричный")
    ap.add_argument("--no-advice", action="store_true")
    ap.add_argument("--json", help="сохранить результат")
    ap.add_argument("--list", action="store_true", help="только перечислить ядра")
    ap.add_argument("--selftest", action="store_true",
                    help="сверить инструмент с ручным счётом 2026-07-31 (ядра ролей i8/e5m2)")
    args = ap.parse_args()

    warps_cfg = None
    if args.roles:
        warps_cfg = {}
        for part in args.roles.split(","):
            k, v = part.split("=")
            warps_cfg[k.strip()] = int(v)
    force = int(args.role_branch, 16) if args.role_branch else None

    if args.selftest:
        selftest(args.binary, args.tau)
        return
    if args.list:
        for k, v in load(args.binary).items():
            print("%6d  %s" % (len(v), k))
        return

    result = {}
    if args.diff:
        ka = load(args.binary, args.diff[0])
        kb = load(args.binary, args.diff[1])
        if not ka or not kb:
            raise SystemExit("не нашёл одно из ядер: A=%d B=%d" % (len(ka), len(kb)))
        na, ba = list(ka.items())[0]
        nb, bb = list(kb.items())[0]
        ca, cb = CFG(ba), CFG(bb)
        ra = detect_roles(ca, args.threads, force, warps_cfg)
        rb = detect_roles(cb, args.threads, force, warps_cfg)
        result["A_report"] = report_kernel(na, ca, args.tau, ra, args)
        if not args.no_advice:
            advise(na, ca, ra, args.tau, result["A_report"])
        result["B_report"] = report_kernel(nb, cb, args.tau, rb, args)
        result["diff"] = diff(na, ca, ra, nb, cb, rb, args.tau)
    else:
        ks = load(args.binary, args.kernel)
        if not ks:
            raise SystemExit("ядра не найдены")
        result["kernels"] = []
        for name, body in list(ks.items())[:6]:
            cfg = CFG(body)
            roles = detect_roles(cfg, args.threads, force, warps_cfg)
            r = report_kernel(name, cfg, args.tau, roles, args)
            if not args.no_advice:
                advise(name, cfg, roles, args.tau, r)
            result["kernels"].append(r)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print("\nсохранено: %s" % args.json)


if __name__ == "__main__":
    main()
