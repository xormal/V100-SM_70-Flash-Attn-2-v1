# Профиль ОДНОГО запроса на стенде: start_profile -> запрос -> stop_profile. Аргументы: порт токенов_промпта max_tokens [метка]
import sys, json, urllib.request, time
port, ntok, mt = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); metka = sys.argv[4] if len(sys.argv)>4 else ""
B=f"http://127.0.0.1:{port}"
def post(path, body=None, timeout=3600):
    r=urllib.request.Request(B+path, data=json.dumps(body or {}).encode(), headers={"Content-Type":"application/json"}, method="POST")
    return urllib.request.urlopen(r, timeout=timeout).read()
M=json.load(urllib.request.urlopen(B+"/v1/models", timeout=30))["data"][0]["id"]
txt=open("/mnt/d1/alex/VLLM_fa2/rabota_70/korpus_en.txt",encoding="utf-8",errors="ignore").read()
# подбор длины по токенизатору сервера
lo,hi=1000,len(txt); s=None
while lo<hi:
    mid=(lo+hi)//2; n=json.loads(post("/tokenize",{"model":M,"prompt":txt[:mid]},120))["count"]
    if n<ntok: lo=mid+1
    else: hi=mid
prompt=txt[:lo]+"\n\nSummarize the above in one sentence."
n=json.loads(post("/tokenize",{"model":M,"prompt":prompt},120))["count"]
if metka: prompt=metka+"\n"+prompt   # ломает префикс-кэш для холодного префилла
post("/start_profile",{},60); t=time.time()
d=json.loads(post("/v1/chat/completions",{"model":M,"messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0.0},3600))
dt=time.time()-t; post("/stop_profile",{},600)
u=d["usage"]; print(f"промпт {u['prompt_tokens']} ткн, выход {u['completion_tokens']} ткн, {dt:.1f} с, профиль остановлен")
