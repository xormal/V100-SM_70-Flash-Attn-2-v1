"""Гейт тождества префикс-кэша ПО ДЛИНЕ.

Один и тот же запрос посылается дважды: первый идёт мимо кэша, второй попадает в него.
При temperature=0 выходы обязаны совпасть ПОБАЙТОВО -- состояние, взятое из кэша, обязано
равняться пересчитанному. Длины подобраны вокруг размера блока (4096): пока блоков в кэше
нет, тождество тривиально, и точка, где оно ломается, называет виновный механизм.
"""
import json, sys, time, urllib.request, glob

ПОРТ = sys.argv[1] if len(sys.argv) > 1 else "8095"
ДЛИНЫ = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                          else "3000,5000,9000,20000,60000".split(","))]
М = "huihui-qwen3.5-35b-a3b-abliterated-q8_0"
_д = "/mnt/d1/alex/VLLM_ch/run/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays/"
СЫРОЙ = ""
for f in sorted(glob.glob(_д + "*.txt")):
    СЫРОЙ += open(f, encoding="utf-8", errors="ignore").read() + "\n\n"

def зов(промпт, макс=64, таймаут=900):
    з = urllib.request.Request(f"http://127.0.0.1:{ПОРТ}/v1/chat/completions",
        data=json.dumps({"model": М, "messages": [{"role": "user", "content": промпт}],
                         "max_tokens": макс, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    о = json.load(urllib.request.urlopen(з, timeout=таймаут))
    u = о["usage"]
    кэш = (u.get("prompt_tokens_details") or {}).get("cached_tokens", -1)
    return о["choices"][0]["message"]["content"] or "", u["prompt_tokens"], кэш

плохих = 0
for L in ДЛИНЫ:
    хвост = f"\n\nВопрос: назови три темы этого текста, по одной на строку."
    мат = (СЫРОЙ * ((int(L / 0.242) // len(СЫРОЙ)) + 1))[: int(L / 0.242)] + хвост
    a, n, к1 = зов(мат)
    b, _, к2 = зов(мат)
    ок = (a == b)
    плохих += (not ок)
    print(f"  длина={n:>6} кэш1={к1:>6} кэш2={к2:>6} тождество={'ДА' if ок else 'НЕТ'}",
          flush=True)
    if not ок:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"    расходятся с символа {i}: ...{a[max(0,i-30):i+20]!r}"
                      f" против ...{b[max(0,i-30):i+20]!r}")
                break
        else:
            print(f"    один -- префикс другого: {len(a)} против {len(b)} символов")
print(f"  ИТОГ: нарушений тождества {плохих}/{len(ДЛИНЫ)}")
sys.exit(1 if плохих else 0)
