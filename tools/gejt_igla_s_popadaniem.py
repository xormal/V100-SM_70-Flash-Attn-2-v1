"""Пункт 3 своим прибором: НЕ побайтовое тождество, а КАЧЕСТВО при попадании в кэш.
Один и тот же длинный промпт дважды -- второй заведомо попадает. Игла обязана найтись ОБА раза."""
import json, urllib.request, time, uuid
B="http://127.0.0.1:8085"; M="Qwen3.6-27B-AWQ"
Ф=("Настоящий раздел содержит служебные сведения о порядке ведения складского учёта. "
   "Записи вносятся в журнал по мере поступления. Ответственный сотрудник заверяет каждую "
   "страницу. Копии хранятся в архиве организации согласно установленному регламенту. ")
код="ФИОЛЕТОВЫЙ-778291"
for цель in (128000, 229000):
    бл=[Ф]*(цель//60)
    бл.insert(len(бл)//2, f"\n\nВАЖНО: секретный код склада — {код}.\n\n")
    p=(f"Дело {uuid.uuid4()}.\n" + "".join(бл)
       + "\n\nВопрос. Назови секретный код склада. Ответь только кодом.")
    for заход in ("ХОЛОДНЫЙ", "ПОПАДАНИЕ"):
        b={"model":M,"messages":[{"role":"user","content":p}],"max_tokens":48,"temperature":0.0}
        r=urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(b).encode(),
                                 headers={"Content-Type":"application/json"})
        t=time.time()
        with urllib.request.urlopen(r,timeout=1800) as f: d=json.load(f)
        a=d["choices"][0]["message"]["content"]
        print(f"  {d['usage']['prompt_tokens']:>7} ткн {заход:10s} {time.time()-t:6.1f} с: "
              f"{'НАЙДЕНА' if код in a.upper() else 'ПОТЕРЯНА'}  {a.strip()[:40]!r}", flush=True)
