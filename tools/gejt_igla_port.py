"""Гейт качества на стенде: игла в длинном контексте + осмысленность короткого ответа.
Нужен, потому что новые пути (tm8 с подбором splits, наше ядро m4) меняют АРИФМЕТИКУ --
побайтовое тождество с прежним путём невозможно по построению, и проверять надо КАЧЕСТВО.
"""
import json, urllib.request, time, uuid, sys
B = "http://127.0.0.1:8095"
M = json.load(urllib.request.urlopen(B + "/v1/models", timeout=30))["data"][0]["id"]
Ф = ("Настоящий раздел содержит служебные сведения о порядке ведения складского учёта. "
     "Записи вносятся в журнал по мере поступления. Ответственный сотрудник заверяет каждую "
     "страницу. Копии хранятся в архиве организации согласно установленному регламенту. ")
код = "ФИОЛЕТОВЫЙ-778291"
цели = [int(x) for x in (sys.argv[1:] or ["16000", "64000"])]
всего = годных = 0
for цель in цели:
    бл = [Ф] * (цель // 60)
    бл.insert(len(бл) // 2, f"\n\nВАЖНО: секретный код склада — {код}.\n\n")
    p = (f"Дело {uuid.uuid4()}.\n" + "".join(бл)
         + "\n\nВопрос. Назови секретный код склада. Ответь только кодом.")
    b = {"model": M, "messages": [{"role": "user", "content": p}],
         "max_tokens": 48, "temperature": 0.0}
    r = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(b).encode(),
                              headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(r, timeout=1800) as f:
        d = json.load(f)
    a = (d["choices"][0]["message"]["content"] or "").strip()
    ок = код in a
    всего += 1; годных += int(ок)
    print(f"  игла {цель:>7d} ткн: {'НАЙДЕНА' if ок else 'ПОТЕРЯНА'}  {time.time()-t:6.1f} с  ответ: {a[:60]!r}")
# осмысленность: короткий вопрос со проверяемым ответом
for вопрос, должно in (("Сколько будет 17*23? Ответь только числом.", "391"),
                       ("Назови столицу Франции одним словом.", "ариж")):
    b = {"model": M, "messages": [{"role": "user", "content": вопрос}],
         "max_tokens": 4000, "temperature": 0.0}
    r = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(b).encode(),
                              headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        d = json.load(f)
    a = (d["choices"][0]["message"]["content"] or "")
    ок = должно in a
    всего += 1; годных += int(ок)
    print(f"  смысл {вопрос[:28]!r}: {'ОК' if ок else 'ПРОВАЛ'}  ответ: {a.strip()[-60:]!r}")
print(f"ГЕЙТ КАЧЕСТВА: {годных}/{всего}")
