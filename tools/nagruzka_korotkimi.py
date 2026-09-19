"""СУЖЕНИЕ НАГРУЗКИ: только короткие ИЛИ только длинные, чтобы понять, что роняет."""
import json, random, sys, threading, time, urllib.request
ПОРТ, РЕЖИМ = sys.argv[1], sys.argv[2]          # режим: korotkie | dlinnye
СЕК = int(sys.argv[3]) if len(sys.argv) > 3 else 300
ПОТОКОВ = int(sys.argv[4]) if len(sys.argv) > 4 else 10
БАЗА = f"http://127.0.0.1:{ПОРТ}"
НАП = ("Настоящий раздел содержит служебные сведения о порядке ведения складского учёта. "
       "Записи вносятся в журнал по мере поступления. ")
стоп = time.time() + СЕК; сделано = [0]; ошибок = [0]; замок = threading.Lock()
def работник(i):
    rnd = random.Random(500 + i)
    while time.time() < стоп:
        если_длин = (РЕЖИМ == "dlinnye")
        т = (НАП * rnd.choice([20, 40, 60]) if если_длин else "") + rnd.choice([
            "Ответь одним словом: столица Франции.", "Сколько будет 17*23?",
            "Перечисли планеты от Солнца до Марса.", "Кратко: что такое хеш-таблица?"])
        d = json.dumps({"model": "Qwen3.6-27B-AWQ", "max_tokens": rnd.choice([16, 40, 120]),
                        "temperature": 0.0, "messages": [{"role": "user", "content": т}]}).encode()
        r = urllib.request.Request(БАЗА + "/v1/chat/completions", d, {"Content-Type": "application/json"})
        try:
            json.load(urllib.request.urlopen(r, timeout=300))
        except Exception:
            with замок: ошибок[0] += 1
            time.sleep(1); continue
        with замок:
            сделано[0] += 1
            if сделано[0] % 50 == 0: print(f"  {РЕЖИМ}: {сделано[0]} запросов, ошибок {ошибок[0]}", flush=True)
пот = [threading.Thread(target=работник, args=(i,), daemon=True) for i in range(ПОТОКОВ)]
for т in пот: т.start()
for т in пот: т.join()
print(f"  ИТОГ {РЕЖИМ}: {сделано[0]} запросов, ошибок {ошибок[0]}")
