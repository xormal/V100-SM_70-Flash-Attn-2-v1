"""СКОРОСТЬ ДЕКОДА ПО ДЛИНЕ КОНТЕКСТА -- наклоном, а не полным временем.
t(200) - t(20) убирает префилл и TTFT; прогрев тем же промптом обязателен."""
import json, time, urllib.request
БАЗА="http://127.0.0.1:8095"; МОДЕЛЬ="Qwen3.6-27B-AWQ"
НАП=("Настоящий раздел содержит служебные сведения о порядке ведения складского "
     "учёта. Записи вносятся в журнал по мере поступления. Ответственный сотрудник "
     "заверяет каждую страницу. Копии хранятся в архиве согласно регламенту. ")
def зов(п,т,to=1800):
    з=urllib.request.Request(БАЗА+п,data=json.dumps(т).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(з,timeout=to) as ф: return json.load(ф)
нб=зов("/tokenize",{"model":МОДЕЛЬ,"prompt":НАП},300)["count"]
def промпт(целевых):
    n=max(1,целевых//max(1,нб))
    return НАП*n+"\n\nПерескажи своими словами, о чём этот текст, подробно и связно."
def проба(текст,ткн):
    t0=time.time()
    о=зов("/v1/chat/completions",{"model":МОДЕЛЬ,"max_tokens":ткн,"temperature":0.0,"ignore_eos":True,
          "messages":[{"role":"user","content":текст}]})
    return time.time()-t0, о["usage"]["prompt_tokens"], о["usage"]["completion_tokens"]
print("  длина    вход   ток/с декода   (t200-t20)")
for цель in (1000, 8000, 32000, 64000, 128000):
    try:
        т=промпт(цель)
        проба(т,16)                      # прогрев ТЕМ ЖЕ промптом
        t20,вх,n1=проба(т,20)
        t200,_,n2=проба(т,200)
        дт=t200-t20
        if n2-n1 < 100: print(f"  {цель:>6}  {вх:>6}   короткий ответ ({n1}/{n2} токенов) -- пропуск", flush=True); continue
        print(f"  {цель:>6}  {вх:>6}   {(n2-n1)/дт:6.1f}          ({дт:5.1f} с на {n2-n1} токенов)",
              flush=True)
    except Exception as e:
        print(f"  {цель:>6}  -- отказ: {type(e).__name__}: {str(e)[:70]}", flush=True)
