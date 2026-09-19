"""МУЛЬТИ-ИГЛА на настоящем тексте: 12 РАЗНЫХ кодов на 12 глубинах, 12 вопросов -- по одному на
код. Это ПРИЁМКА разреженности по правилу пускача: сравнивать с ПЛОТНЫМ эталоном на том же
тексте; текст годен только если плотный даёт 12/12. Одиночная игла без эталона не доказывает
ничего (записка 25 §303).
Запуск: multi_igla_tekst.py <порт> <токенов>
"""
import json, urllib.request, time, uuid, io, sys, random
PORT = sys.argv[1] if len(sys.argv) > 1 else "8085"
B = f"http://127.0.0.1:{PORT}"
цель = int(sys.argv[2]) if len(sys.argv) > 2 else 250000
M = json.load(urllib.request.urlopen(B + "/v1/models", timeout=30))["data"][0]["id"]
текст = io.open("/mnt/d1/alex/VLLM_fa2/rabota_70/korpus_en.txt", encoding="utf-8").read()

def токенов(s):
    r = urllib.request.Request(B + "/tokenize", data=json.dumps({"model": M, "prompt": s}).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.load(f)["count"]

зн = цель * 4
for _ in range(3):
    n = токенов(текст[:зн])
    if abs(n - цель) < цель * 0.01: break
    зн = int(зн * цель / max(n, 1))
куск = текст[:зн]
rng = random.Random(20260911)
цвета = ["VIOLET", "AMBER", "CRIMSON", "INDIGO", "OLIVE", "SCARLET", "TEAL", "MAGENTA", "COBALT", "IVORY", "SIENNA", "MAROON"]
коды = [f"{c}-{rng.randint(100000, 999999)}" for c in цвета]
имена = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"]
# 12 глубин от 4 % до 96 %
части = []; поз = 0
for i, (имя, код) in enumerate(zip(имена, коды)):
    гл = int(len(куск) * (0.04 + 0.92 * i / 11))
    части.append(куск[поз:гл]); части.append(f"\n\nIMPORTANT: the access code for warehouse {имя} is {код}.\n\n"); поз = гл
части.append(куск[поз:])
p = f"Document {uuid.uuid4()}.\n" + "".join(части)
итог_ток = токенов(p)
print(f"  промпт {итог_ток} токенов, 12 кодов на глубинах 4..96 %")
найдено = 0; t_всего = 0.0; ответы = []
for i, (имя, код) in enumerate(zip(имена, коды)):
    q = p + f"\n\nQuestion. What is the access code for warehouse {имя}? Answer with the code only."
    b = {"model": M, "messages": [{"role": "user", "content": q}], "max_tokens": 48, "temperature": 0.0}
    r = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(r, timeout=3600) as f:
        d = json.load(f)
    дт = time.time() - t; t_всего += дт
    a = (d["choices"][0]["message"]["content"] or "").strip()
    ок = код in a; найдено += int(ок); ответы.append((имя, ок, a[:28], дт))
    print(f"    {i+1:2d}. {имя:8s} глубина {4+92*i//11:2d}%  {'НАЙДЕН ' if ок else 'ПОТЕРЯН'}  {дт:6.1f} с  {a[:28]!r}")
print(f"  ИТОГ: {найдено}/12 кодов, первый запрос (префилл) {ответы[0][3]:.1f} с, остальные через кэш")
