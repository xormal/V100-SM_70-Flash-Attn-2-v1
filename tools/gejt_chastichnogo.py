"""Гейт тождества при ЧАСТИЧНОМ попадании в префикс-кэш.

Полное попадание уже проверено побайтово. Здесь проверяется путь «часть блоков из кэша,
остаток досчитывается»: два запроса делят длинный префикс X, но имеют разные хвосты.
Порядок: холодный P2 -> P1 (кладёт блоки X в кэш) -> горячий P2. Ответы на P2 обязаны
совпасть: состояние, собранное из кэша плюс досчёт, обязано равняться пересчитанному.
"""
import json, sys, time, urllib.request, glob

ПОРТ = sys.argv[1] if len(sys.argv) > 1 else "8095"
М = "huihui-qwen3.5-35b-a3b-abliterated-q8_0"
_д = "/mnt/d1/alex/VLLM_ch/run/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays/"
СЫРОЙ = ""
for f in sorted(glob.glob(_д + "*.txt")):
    СЫРОЙ += open(f, encoding="utf-8", errors="ignore").read() + "\n\n"

def зов(промпт, макс=64):
    з = urllib.request.Request(f"http://127.0.0.1:{ПОРТ}/v1/chat/completions",
        data=json.dumps({"model": М, "messages": [{"role": "user", "content": промпт}],
                         "max_tokens": макс, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    о = json.load(urllib.request.urlopen(з, timeout=900))
    return о["choices"][0]["message"]["content"] or "", о["usage"]["prompt_tokens"]

плохих = 0
for L in (6000, 9000, 20000):
    X = (СЫРОЙ * 4)[: int(L / 0.242)]
    метка = f"{time.time():.6f}"
    P1 = X + f"\n\nПометка {метка}-один.\nВопрос: назови три темы текста."
    P2 = X + f"\n\nПометка {метка}-два.\nВопрос: назови три темы текста."
    хол, n = зов(P2)          # холодный: блоков X в кэше ещё нет
    зов(P1)                    # кладёт блоки X в кэш
    гор, _ = зов(P2)           # частичное попадание: X из кэша, хвост досчитан
    ок = (хол == гор)
    плохих += (not ок)
    print(f"  длина={n:>6} частичное попадание: тождество={'ДА' if ок else 'НЕТ'}", flush=True)
    if not ок:
        for i, (a, b) in enumerate(zip(хол, гор)):
            if a != b:
                print(f"    расходятся с символа {i}: {хол[max(0,i-25):i+15]!r}"
                      f" против {гор[max(0,i-25):i+15]!r}")
                break
        else:
            print(f"    один -- префикс другого: {len(хол)} против {len(гор)} символов")
print(f"  ИТОГ: нарушений {плохих}/3")
sys.exit(1 if плохих else 0)
