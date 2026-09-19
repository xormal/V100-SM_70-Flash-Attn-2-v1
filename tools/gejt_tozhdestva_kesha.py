"""ГЕЙТ ТОЖДЕСТВА ПРИ ПОПАДАНИИ В ПРЕФИКС-КЭШ (пункт 3 цели).
Один и тот же длинный запрос дважды: второй обязан выдать ПОБАЙТОВО то же и быть быстрее."""
import json, urllib.request, time, uuid
B="http://127.0.0.1:8085"; M="Qwen3.6-27B-AWQ"
# УНИКАЛЬНАЯ МЕТКА: на живом сервере кэш уже полон, и «первый» вызов холодным НЕ
# является -- в первом заходе он дал 310, во втором 426 при неизменном втором (346).
p=(f"Регламент {uuid.uuid4()}. "+("Пункт о порядке хранения и выдачи материалов. "*4200)
   +"\n\nВопрос. Перескажи регламент своими словами в пяти предложениях.")
def зов():
    b={"model":M,"messages":[{"role":"user","content":p}],"max_tokens":160,"temperature":0.0}
    r=urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    t=time.time()
    with urllib.request.urlopen(r,timeout=900) as f: d=json.load(f)
    return d["choices"][0]["message"]["content"], time.time()-t, d["usage"]["prompt_tokens"]
a1,t1,n=зов(); a2,t2,_=зов()
print(f"  вход {n} токенов | первый {t1:.2f} с, второй {t2:.2f} с (x{t1/t2:.2f})")
print(f"  ТОЖДЕСТВО: {'ПОБАЙТОВО СОВПАЛО' if a1==a2 else 'РАЗОШЛОСЬ'} "
      f"(длина {len(a1)} / {len(a2)})")
