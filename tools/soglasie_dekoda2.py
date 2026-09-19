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
teg = sys.argv[7] if len(sys.argv) > 7 else ""
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
    d = post("/v1/completions", {"model": M, "prompt": p, "max_tokens": n_gen, "temperature": 0.0, "logprobs": 5})
    dt = time.time() - t
    ch = d["choices"][0]; lp = ch["logprobs"]
    toks = lp["tokens"]; lps = lp["token_logprobs"]
    rez[str(L)] = {"tokens": toks, "logprobs": lps, "top": lp.get("top_logprobs"), "prompt_tokens": d["usage"]["prompt_tokens"], "sec": dt}
    print(f"  L={L}: промпт {d['usage']['prompt_tokens']} ткн, {len(toks)} ткн выхода, {dt:.1f} с", flush=True)
if rezhim == "zapis":
    json.dump(rez, open(fjson, "w"), ensure_ascii=False); print(f"эталон записан: {fjson}")
else:
    et = json.load(open(fjson))
    tag = (teg or metka or "sverka").replace("/", "_")
    json.dump(rez, open(fjson.replace(".json", f".{tag}.json"), "w"), ensure_ascii=False)   # сырые данные сверки -- для разбора потом
    print(f"{'L':>7} {'расх.@':>7} {'|Δlp| общ.преф.':>16} {'зазор плотного':>15} {'ранг у плотного':>16} {'с':>7}")
    for L in dliny:
        a, b = et[str(L)], rez[str(L)]
        ta, tb = a["tokens"], b["tokens"]; n = min(len(ta), len(tb))
        div = next((i for i in range(n) if ta[i] != tb[i]), n)
        dlp = sum(abs(a["logprobs"][i] - b["logprobs"][i]) for i in range(div)) / max(1, div)
        zazor = rang = "-"
        if div < n and a.get("top") and a["top"][div]:
            top = a["top"][div]                       # {токен: logprob} у плотного в точке расхождения
            vals = sorted(top.values(), reverse=True)
            if len(vals) > 1: zazor = f"{vals[0] - vals[1]:.3f}"   # уверенность плотного: logprob топ-1 минус топ-2
            rang = str(sorted(top, key=top.get, reverse=True).index(tb[div]) + 1) if tb[div] in top else ">5"
        print(f"{L:>7} {div:>7} {dlp:>16.4f} {zazor:>15} {rang:>16} {b['sec']:>7.1f}")
    # Чтение: зазор < ~0.3 и ранг 2 -- расхождение на НИЧЬЕЙ (допустимо); зазор > 1 или ранг >5 -- порча.
