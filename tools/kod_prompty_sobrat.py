#!/usr/bin/env python3
# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ПУЛ КОДИНГОВЫХ ЗАДАЧ ДЛЯ КОРПУСА ЧЕРНОВИКА (записка 25 §144).
#
# ЗАЧЕМ. Приёмка черновика на прозе 1.7-2.0 при нужных 3.19 (§134), и это НЕСООТВЕТСТВИЕ
# модели: черновик учили под БАЗОВУЮ Qwen3.8-27B, боевая -- ABLITERATED. Дообучение идёт
# парами «признаки цели -> её же токены», которые собирает сам сервер
# (`FA2SM70_DFLASH2_DUMP`). Значит нужен ПОТОК ЗАПРОСОВ той природы, на которой черновик
# обязан работать. Здесь -- кодинговые задачи с HF.
#
# ИСТОЧНИКИ (скачаны 06.09 в /mnt/d1/alex/datasets):
#   ise-uiuc/Magicoder-OSS-Instruct-75K      75 197 задач, поля problem/solution, мультиязык
#   m-a-p/CodeFeedback-Filtered-Instruction 156 526 задач, поля query/answer
#
# ОТБОР: длина задачи в пределах [ЛМИН, ЛМАКС] знаков (слишком короткие не дают шагов
# декода, слишком длинные тратят время на префилл), дедупликация по первым 200 знакам,
# равные доли источников, порядок перемешан устойчивым зерном.
#
# ЗАПУСК: python3 tools/kod_prompty_sobrat.py [сколько] [выход.jsonl]
import hashlib, json, random, sys

ДАННЫЕ = [("/mnt/d1/alex/datasets/Magicoder-OSS-Instruct-75K/"
           "data-oss_instruct-decontaminated.jsonl", "problem", "magicoder"),
          ("/mnt/d1/alex/datasets/CodeFeedback-Filtered-Instruction/"
           "CodeFeedback-Filtered-Instruction.jsonl", "query", "codefeedback")]
СКОЛЬКО = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
ВЫХОД = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d1/alex/datasets/kod_prompty.jsonl"
ЛМИН, ЛМАКС = 200, 2000

взято, виден = [], set()
for путь, поле, метка in ДАННЫЕ:
    свои = []
    with open(путь, encoding="utf-8") as ф:
        for л in ф:
            т = (json.loads(л).get(поле) or "").strip()
            if not (ЛМИН <= len(т) <= ЛМАКС):
                continue
            к = hashlib.blake2b(т[:200].encode(), digest_size=8).hexdigest()
            if к in виден:
                continue
            виден.add(к)
            свои.append({"задача": т, "источник": метка})
    random.Random(20260906).shuffle(свои)
    взято.append(свои)
    print(f"  {метка}: годных {len(свои)}")

доля = СКОЛЬКО // len(взято)
итог = [з for свои in взято for з in свои[:доля]]
random.Random(4419).shuffle(итог)
with open(ВЫХОД, "w", encoding="utf-8") as ф:
    for з in итог:
        ф.write(json.dumps(з, ensure_ascii=False) + "\n")
дл = sorted(len(з["задача"]) for з in итог)
print(f"  записано {len(итог)} задач -> {ВЫХОД}")
print(f"  длина знаков: медиана {дл[len(дл)//2]}, 10 % {дл[len(дл)//10]}, 90 % {дл[9*len(дл)//10]}")
