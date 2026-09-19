# Смешанный батч: длинный декод (спекулятивные строки) + параллельный ПРЕФИЛЛ нового промпта. Аргумент: порт.
import sys, json, urllib.request, threading, time
port=int(sys.argv[1]); B=f"http://127.0.0.1:{port}"
M=json.load(urllib.request.urlopen(B+"/v1/models",timeout=30))["data"][0]["id"]
def ask(content, mt, tag, out):
    r=urllib.request.Request(B+"/v1/chat/completions", data=json.dumps({"model":M,"messages":[{"role":"user","content":content}],"max_tokens":mt,"temperature":0.0}).encode(), headers={"Content-Type":"application/json"}, method="POST")
    t=time.time()
    try:
        d=json.loads(urllib.request.urlopen(r,timeout=900).read()); out[tag]=(d["usage"]["completion_tokens"], round(time.time()-t,1), "ok")
    except Exception as e: out[tag]=(0, round(time.time()-t,1), f"{type(e).__name__}: {str(e)[:60]}")
txt=open("/mnt/d1/alex/VLLM_fa2/rabota_70/korpus_en.txt",encoding="utf-8",errors="ignore").read()
out={}
t1=threading.Thread(target=ask, args=("Write a very long detailed story about a lighthouse keeper, at least 400 words.", 300, "декод", out)); t1.start()
time.sleep(3)   # декод уже идёт (спекулятивные строки) -> добавляем префилл
t2=threading.Thread(target=ask, args=(txt[:60000]+"\n\nSummarize in one sentence.", 8, "префилл", out)); t2.start()
t1.join(); t2.join()
for k,v in out.items(): print(f"  {k}: токенов {v[0]}, {v[1]} с, {v[2]}")
try: print("  health:", urllib.request.urlopen(B+"/health",timeout=5).status)
except Exception as e: print("  health: ОТКАЗ", type(e).__name__)
