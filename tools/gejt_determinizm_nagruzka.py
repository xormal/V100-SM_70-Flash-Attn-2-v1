"""Детерминизм при temperature=0 ПОД НАГРУЗКОЙ и БЕЗ неё.

Разводит две причины разброса ответов: порчу состояния (тогда разброс есть и в одиночку) и
батчинг (тогда он появляется только под нагрузкой, потому что с размером батча меняется
порядок редукций). Оба случая гоняются на одном подъёме.
"""
import json, sys, threading, time, urllib.request, hashlib, glob
ПОРТ = sys.argv[1] if len(sys.argv) > 1 else "8095"
М = "huihui-qwen3.5-35b-a3b-abliterated-q8_0"
_д = "/mnt/d1/alex/VLLM_ch/run/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays/"
С = ""
for f in sorted(glob.glob(_д + "*.txt")):
    С += open(f, encoding="utf-8", errors="ignore").read() + "\n\n"
def зов(промпт, макс=250, темп=0.0, таймаут=900):
    з = urllib.request.Request(f"http://127.0.0.1:{ПОРТ}/v1/chat/completions",
        data=json.dumps({"model": М, "messages": [{"role": "user", "content": промпт}],
                         "max_tokens": макс, "temperature": темп}).encode(),
        headers={"Content-Type": "application/json"})
    о = json.load(urllib.request.urlopen(з, timeout=таймаут))
    return (о["choices"][0]["message"]["content"] or "",
            о["usage"]["prompt_tokens"], о["usage"]["completion_tokens"])
стоп = threading.Event()
def фон():
    i = 0
    while not стоп.is_set():
        i += 1
        try: зов(f"Расскажи про число {i} в двух предложениях.", 80, 0.3, 120)
        except Exception: pass
        time.sleep(1)
вопрос = "\n\nПерескажи этот текст подробно, не менее 150 слов."
случаи = (("граница В ДЕКОДЕ", 8050), ("границ НЕТ", 5000))
итог = {}
for режим in ("в одиночку", "под нагрузкой"):
    if режим == "под нагрузкой":
        for _ in range(3): threading.Thread(target=фон, daemon=True).start()
        time.sleep(5)
    for имя, симв in случаи:
        промпт = (С * 2)[: int(симв / 0.242)] + вопрос
        виды, вх, вых = {}, 0, 0
        for _ in range(5):
            т, вх, вых = зов(промпт)
            виды[hashlib.sha1(т.encode()).hexdigest()[:10]] = 1
        итог[(режим, имя)] = len(виды)
        print(f"  {режим:<14} {имя:<17} вход={вх} сумма={вх+вых} -> РАЗНЫХ {len(виды)} из 5",
              flush=True)
стоп.set()
од = max(итог[("в одиночку", и)] for и, _ in случаи)
нг = max(итог[("под нагрузкой", и)] for и, _ in случаи)
print(f"  ИТОГ: в одиночку до {од} видов, под нагрузкой до {нг}")
print("  вывод: " + ("порча состояния" if од > 1 else
                     "батчинг (в одиночку детерминизм)" if нг > 1 else "детерминизм везде"))
