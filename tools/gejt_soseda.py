"""Гейт независимости от соседа по батчу.

При temperature=0 выход запроса не должен зависеть от того, кто ещё считается в этом шаге.
Прибор шлёт один и тот же запрос дважды: в одиночку и одновременно с длинным префиллом
соседа. Расхождение означает, что смешанный батч меняет численность -- это и есть остаток
по цели 8 (12/20 полных под нагрузкой против 20/20 без префикс-кэша).
"""
import json, sys, threading, time, urllib.request, glob

ПОРТ = sys.argv[1] if len(sys.argv) > 1 else "8095"
ПОВТОРОВ = int(sys.argv[2]) if len(sys.argv) > 2 else 6
М = "huihui-qwen3.5-35b-a3b-abliterated-q8_0"
_д = "/mnt/d1/alex/VLLM_ch/run/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays/"
СЫРОЙ = ""
for f in sorted(glob.glob(_д + "*.txt")):
    СЫРОЙ += open(f, encoding="utf-8", errors="ignore").read() + "\n\n"

def зов(промпт, макс=120, таймаут=900):
    з = urllib.request.Request(f"http://127.0.0.1:{ПОРТ}/v1/chat/completions",
        data=json.dumps({"model": М, "messages": [{"role": "user", "content": промпт}],
                         "max_tokens": макс, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    о = json.load(urllib.request.urlopen(з, timeout=таймаут))
    return о["choices"][0]["message"]["content"] or ""

ПРОБА = "Расскажи про историю мореплавания ровно в семи предложениях."
СОСЕД = (СЫРОЙ * 2)[: int(30000 / 0.242)] + "\n\nНазови три темы этого текста."

одиночка = зов(ПРОБА)
print(f"  эталон (в одиночку): {len(одиночка)} символов")
расх = 0
for i in range(ПОВТОРОВ):
    держи = threading.Event()
    def сосед():
        try: зов(СОСЕД, макс=40)
        except Exception: pass
        держи.set()
    т = threading.Thread(target=сосед, daemon=True); т.start()
    time.sleep(0.4)                      # чтобы префилл соседа шёл в том же шаге
    вместе = зов(ПРОБА)
    держи.wait(timeout=600)
    ок = (вместе == одиночка)
    расх += (not ок)
    print(f"  прогон {i+1}: с соседом {'СОВПАЛО' if ок else 'РАСХОЖДЕНИЕ'} "
          f"({len(вместе)} символов)", flush=True)
    if not ок and расх == 1:
        for j, (a, b) in enumerate(zip(одиночка, вместе)):
            if a != b:
                print(f"    с символа {j}: {одиночка[max(0,j-40):j+20]!r}")
                print(f"           против: {вместе[max(0,j-40):j+20]!r}")
                break
        else:
            print(f"    один -- префикс другого: {len(одиночка)} против {len(вместе)}")
print(f"  ИТОГ: расхождений {расх}/{ПОВТОРОВ}")
sys.exit(1 if расх else 0)
