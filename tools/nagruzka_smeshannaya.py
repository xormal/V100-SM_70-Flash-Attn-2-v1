"""СМЕШАННАЯ НАГРУЗКА: короткие и длинные запросы вперемешку, много потоков.
Цель -- гонять состав батча: строки батча переиспользуются РАЗНЫМИ запросами, а
именно на этом и ломается пара указателей свёртки (гипотеза падения 06.09)."""
import json, random, sys, threading, time, urllib.request
ПОРТ=sys.argv[1] if len(sys.argv)>1 else "8095"
СЕК=int(sys.argv[2]) if len(sys.argv)>2 else 240
ПОТОКОВ=int(sys.argv[3]) if len(sys.argv)>3 else 10
БАЗА=f"http://127.0.0.1:{ПОРТ}"
НАП=("Настоящий раздел содержит служебные сведения о порядке ведения складского учёта. "
     "Записи вносятся в журнал по мере поступления. ")
стоп=time.time()+СЕК; сделано=[0]; ошибок=[0]; замок=threading.Lock()
def один(rnd):
    длина=rnd.choice([0,0,0,1,2,3,8,20,60])     # от короткого до ~8K токенов
    т=НАП*(длина*6)+rnd.choice(["Ответь одним словом: столица Франции.",
        "Сколько будет 17*23?","Перечисли планеты от Солнца до Марса.",
        "Кратко объясни, что такое хеш-таблица.","Напиши функцию на Python: сумма чисел списка."])
    d=json.dumps({"model":"Qwen3.6-27B-AWQ","max_tokens":rnd.choice([16,40,120,200]),
                  "temperature":0.0,"messages":[{"role":"user","content":т}]}).encode()
    r=urllib.request.Request(БАЗА+"/v1/chat/completions",d,{"Content-Type":"application/json"})
    json.load(urllib.request.urlopen(r,timeout=300))
def работник(i):
    rnd=random.Random(1000+i)
    while time.time()<стоп:
        try: один(rnd); 
        except Exception: 
            with замок: ошибок[0]+=1
            time.sleep(1); continue
        with замок:
            сделано[0]+=1
            if сделано[0]%50==0: print(f"  запросов {сделано[0]}, ошибок {ошибок[0]}",flush=True)
пот=[threading.Thread(target=работник,args=(i,),daemon=True) for i in range(ПОТОКОВ)]
for т in пот: т.start()
for т in пот: т.join()
print(f"  ИТОГ: запросов {сделано[0]}, ошибок {ошибок[0]}")
