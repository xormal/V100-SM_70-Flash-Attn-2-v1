# СОГЛАСИЕ ЖАДНОГО ДЕКОДА С ПЛОТНЫМ -- непрерывный прибор качества разреженного префилла (записка 26 §4).
# Иглы ловят пики и слепы к порче рассеянных слоёв; здесь меряется, как далеко жадное продолжение длинного
# текста совпадает с плотным эталоном: первая точка расхождения, доля совпавших токенов, средний |Δ logprob|
# выбранного токена на общем префиксе.
# Запуск: soglasie_dekoda.py <порт> zapis <файл.json> <длины через запятую> [n_gen] [metka]   -- записать эталон
#         soglasie_dekoda.py <порт> sverka <файл.json> <длины через запятую> [n_gen] [metka]  -- сравнить с эталоном
import json, sys, time, urllib.request
port, rezhim, fjson = sys.argv[1], sys.argv[2], sys.argv[3]
dliny = [int(x) for x in sys.argv[4].split(",")]; n_gen = int(sys.argv[5]) if len(sys.argv) > 5 else 128
metka = sys.argv[6] if len(sys.argv) > 6 else ""
B = f"http://127.0.0.1:{port}"
def post(path, body, timeout=3600):
    r = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())
M = json.load(urllib.request.urlopen(B + "/v1/models", timeout=30))["data"][0]["id"]
txt = open("/mnt/d1/alex/VLLM_fa2/rabota_70/korpus_en.txt", encoding="utf-8", errors="ignore").read()
def tok(s): return post("/tokenize", {"model": M, "prompt": s}, 600)["count"]
def srez(n):   # текст длиной ровно ~n токенов (бинарный поиск по токенизатору сервера)
    lo, hi = 1000, len(txt)
    while lo < hi:
        mid = (lo + hi) // 2
        if tok(txt[:mid]) < n: lo = mid + 1
        else: hi = mid
    return txt[:lo]
rez = {}
for L in dliny:
    p = srez(L); p = (metka + "\n" + p) if metka else p
    t = time.time()
    d = post("/v1/completions", {"model": M, "prompt": p, "max_tokens": n_gen, "temperature": 0.0, "logprobs": 1})
    dt = time.time() - t
    ch = d["choices"][0]; lp = ch["logprobs"]
    toks = lp["tokens"]; lps = lp["token_logprobs"]
    rez[str(L)] = {"tokens": toks, "logprobs": lps, "prompt_tokens": d["usage"]["prompt_tokens"], "sec": dt}
    print(f"  L={L}: промпт {d['usage']['prompt_tokens']} ткн, {len(toks)} ткн выхода, {dt:.1f} с", flush=True)
if rezhim == "zapis":
    json.dump(rez, open(fjson, "w"), ensure_ascii=False); print(f"эталон записан: {fjson}")
else:
    et = json.load(open(fjson))
    print(f"{'L':>7} {'расх.@':>7} {'совп.%':>7} {'|Δlp| общ.преф.':>16} {'с':>7}")
    for L in dliny:
        a, b = et[str(L)], rez[str(L)]
        ta, tb = a["tokens"], b["tokens"]; n = min(len(ta), len(tb))
        div = next((i for i in range(n) if ta[i] != tb[i]), n)
        sovp = sum(1 for i in range(n) if ta[i] == tb[i]) / max(1, n)
        dlp = sum(abs(a["logprobs"][i] - b["logprobs"][i]) for i in range(div)) / max(1, div)
        print(f"{L:>7} {div:>7} {100*sovp:>6.1f}% {dlp:>16.4f} {b['sec']:>7.1f}")
