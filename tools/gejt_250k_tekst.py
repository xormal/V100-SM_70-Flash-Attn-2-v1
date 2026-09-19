"""250K префилла на НАСТОЯЩЕМ тексте (wikitext) + игла. Заполнитель по времени льстит (§293),
поэтому время мерим только так. Длину подбираем по токенизатору сервера."""
import json, urllib.request, time, uuid, io, sys
B = "http://127.0.0.1:8095"
M = json.load(urllib.request.urlopen(B + "/v1/models", timeout=30))["data"][0]["id"]
текст = io.open("/mnt/d1/alex/VLLM_fa2/rabota_70/korpus_en.txt", encoding="utf-8").read()
цель = int(sys.argv[1]) if len(sys.argv) > 1 else 250000
код = "VIOLET-778291"


def токенов(s):
    r = urllib.request.Request(B + "/tokenize", data=json.dumps({"model": M, "prompt": s}).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.load(f)["count"]


# подбор длины куска текста под нужное число токенов (2 итерации достаточно)
зн = цель * 4
for _ in range(3):
    n = токенов(текст[:зн])
    print(f"  проба: {зн} знаков -> {n} токенов")
    if abs(n - цель) < цель * 0.01:
        break
    зн = int(зн * цель / max(n, 1))
куск = текст[:зн]
сер = len(куск) // 2
p = (f"Document {uuid.uuid4()}.\n" + куск[:сер]
     + f"\n\nIMPORTANT: the secret warehouse code is {код}.\n\n" + куск[сер:]
     + "\n\nQuestion. What is the secret warehouse code? Answer with the code only.")
итог_ток = токенов(p)
print(f"  ИТОГО промпт: {итог_ток} токенов")
b = {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": 48, "temperature": 0.0}
r = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(b).encode(),
                           headers={"Content-Type": "application/json"})
t = time.time()
with urllib.request.urlopen(r, timeout=3600) as f:
    d = json.load(f)
дт = time.time() - t
a = (d["choices"][0]["message"]["content"] or "").strip()
print(f"  РЕЗУЛЬТАТ: {итог_ток} токенов за {дт:.1f} с ({итог_ток/дт:.0f} ток/с), "
      f"игла {'НАЙДЕНА' if код in a else 'ПОТЕРЯНА'}, ответ {a[:40]!r}")
