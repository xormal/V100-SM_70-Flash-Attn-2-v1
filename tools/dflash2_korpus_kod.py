#!/usr/bin/env python3
# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ГОНИТ КОДИНГОВЫЕ ЗАДАЧИ В СЕРВЕР, ЧТОБЫ ОН СОБРАЛ КОРПУС ЧЕРНОВИКА (записка 25 §144).
#
# Корпус пишет САМ сервер при `FA2SM70_DFLASH2_DUMP=<каталог>`: пары «признаки цели ->
# её же логиты» берутся из штатного хода спекуляции, поэтому нужен просто ПОТОК
# запросов нужной природы. Здесь поток -- задачи с HF (см. kod_prompty_sobrat.py).
#
# ВАЖНО: гнать на СТЕНДЕ, не на боевом -- дамп пишет десятки килобайт на шаг и добавляет
# нагрузку. Спекуляция обязана быть включена, иначе пар не будет вовсе.
#
# ЗАПУСК: python3 tools/dflash2_korpus_kod.py [порт] [сколько] [потоков] [макс_токенов]
import json, os, queue, sys, threading, time, urllib.request

ПОРТ = sys.argv[1] if len(sys.argv) > 1 else "8095"
СКОЛЬКО = int(sys.argv[2]) if len(sys.argv) > 2 else 800
ПОТОКОВ = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ТОКЕНОВ = int(sys.argv[4]) if len(sys.argv) > 4 else 256
ПУЛ = os.environ.get("КОД_ПУЛ", "/mnt/d1/alex/datasets/kod_prompty.jsonl")
ДАМП = os.environ.get("ДАМП_КАТАЛОГ", "/mnt/d1/alex/dflash2_pary_kod")
МОДЕЛЬ = os.environ.get("МОДЕЛЬ", "Qwen3.6-27B-AWQ")
БАЗА = f"http://127.0.0.1:{ПОРТ}"

задачи = queue.Queue()
with open(ПУЛ, encoding="utf-8") as ф:
    for i, л in enumerate(ф):
        if i >= СКОЛЬКО:
            break
        задачи.put(json.loads(л)["задача"])
всего = задачи.qsize()
готово = [0]
токенов = [0]
замок = threading.Lock()
начало = time.time()

def работник():
    while True:
        try:
            з = задачи.get_nowait()
        except queue.Empty:
            return
        try:
            d = json.dumps({"model": МОДЕЛЬ, "max_tokens": ТОКЕНОВ, "temperature": 0.0,
                            "messages": [{"role": "user", "content": з}]}).encode()
            r = urllib.request.Request(БАЗА + "/v1/chat/completions", d,
                                       {"Content-Type": "application/json"})
            o = json.load(urllib.request.urlopen(r, timeout=600))
            n = o["usage"]["completion_tokens"]
        except Exception:
            n = 0
        with замок:
            готово[0] += 1
            токенов[0] += n
            if готово[0] % 25 == 0 or готово[0] == всего:
                мб = sum(os.path.getsize(os.path.join(ДАМП, f))
                         for f in os.listdir(ДАМП)) / 1e6 if os.path.isdir(ДАМП) else 0
                дт = time.time() - начало
                print(f"  {готово[0]:>4}/{всего} задач, {токенов[0]:>6} токенов, "
                      f"корпус {мб:7.1f} МБ, {токенов[0]/дт:5.1f} ток/с, {дт/60:.1f} мин",
                      flush=True)

потоки = [threading.Thread(target=работник, daemon=True) for _ in range(ПОТОКОВ)]
for т in потоки: т.start()
for т in потоки: т.join()
print(f"  ГОТОВО: {готово[0]} задач, {токенов[0]} токенов за {(time.time()-начало)/60:.1f} мин")
