# ОРАКУЛ ИГЛЫ: почему отбор теряет иглу в поздних слоях. По дампу q/K последнего чанка (FA2SM70_DUMP_QK, один файл = один слой)
# для СТРОК ВОПРОСА (последние QR строк чанка) считает, попадает ли блок иглы в набор отбора при MASS по пяти правилам:
#   A  боевое: 16 зондов плитки, оценщик s=(q*kbar + gamma*sum|q_d|dev_d)*scale, масса блока E_J = sum_r exp(s_rJ - M) (общий M)
#   B  та же свёртка, но с ТОЧНЫМИ счётами по ключам (изолирует ошибку оценщика от свёртки)
#   C  зонды, но вклад строки НОРМИРОВАН (softmax по строке), E_J = sum_r P_r(J)
#   E  как C, но ВСЕ строки плитки (без прореживания зондами)
#   D  объединение по строкам вопроса: каждая держит MASS своей массы (оракул на иглу)
# Аргументы: igly.json каталог_дампов [MASS=0.98] [QR=40]. Карта: CUDA_VISIBLE_DEVICES.
import torch, sys, glob, json, re, math
ig = json.load(open(sys.argv[1])); D = sys.argv[2]; MASS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.98
QR = int(sys.argv[4]) if len(sys.argv) > 4 else 40
B = 128; TI = 128; PROBE = 16; GAMMA = 0.5
files = sorted(glob.glob(D + "/qk.*.r0.pt"), key=lambda f: int(re.search(r"qk\.(\d+)\.", f).group(1)))
print(f"иглы: {ig['needles'][0]} глубина {ig['depths'][0]}%; промпт {ig['n_in']} ткн; дампов {len(files)}; MASS={MASS} QR={QR}")
hdr = f"{'слой':>4} {'T':>7} {'Sq':>4} | {'блок иглы':>9} {'масса_вопр':>10} | {'A боевое':>14} {'An оц.+норм':>14} {'B точн.свёртка':>14} {'C норм.зонды':>14} {'E норм.все':>14} {'D по строкам':>14}"
print(hdr)
for li, f in enumerate(files):
    o = torch.load(f, map_location="cpu")
    q = o["q"].cuda().float(); k = o["k"].cuda().float(); scale = o["scale"]; T = o["T"]; Sq = o["Sq"]; Hkv = o["Hkv"]; d = o["d"]
    H = q.shape[1]; G = H // Hkv; NB = (T + B - 1) // B; q0 = T - Sq
    # сводки блоков
    kp = torch.zeros(Hkv, NB * B, d, device="cuda"); kp[:, :T] = k; kp = kp.view(Hkv, NB, B, d)
    cnt = torch.full((NB,), float(B), device="cuda"); cnt[-1] = T - (NB - 1) * B
    kbar = kp.sum(2) / cnt[None, :, None]                              # [Hkv, NB, d]
    dev = (kp - kbar[:, :, None, :]).abs()
    if T % B: dev[:, -1, T - (NB - 1) * B:, :] = 0
    dev = dev.amax(2)                                                   # [Hkv, NB, d] размах по измерению
    # последняя плитка чанка и её зонды (как в ядре)
    t0 = (Sq - 1) // TI * TI; TIe = Sq - t0
    probes = [t0 + (pr * (TIe - 1)) // (PROBE - 1) for pr in range(PROBE)] if PROBE > 1 else [t0]
    qrows = list(range(Sq - QR, Sq))                                    # строки вопроса
    tile_rows = list(range(t0, Sq))
    nvis = (q0 + Sq - 1) // B + 1
    jj = torch.arange(T, device="cuda")
    def true_mass(rows, hk):                                            # [len(rows)*G, NB] нормированная масса и ненормированные exp(s-M)
        Q = q[rows][:, hk * G:(hk + 1) * G].reshape(-1, d)              # [R*G, d]
        pos = (q0 + torch.tensor(rows, device="cuda")).repeat_interleave(G)
        S = (Q @ k[hk].t()) * scale; S.masked_fill_(jj[None, :] > pos[:, None], float("-inf"))
        M = S.max(); Ex = torch.exp(S - M)
        P = Ex / Ex.sum(1, keepdim=True)
        pad = lambda X: torch.cat([X, torch.zeros(X.shape[0], NB * B - T, device="cuda")], 1).view(X.shape[0], NB, B).sum(2)
        return pad(P), pad(Ex), Q
    def est_scores(Q, hk):                                              # оценщик ядра: [R, NB]
        return (Q @ kbar[hk].t() + GAMMA * (Q.abs() @ dev[hk].t())) * scale
    def select_by_mass(E):                                              # набор блоков по накопленной массе
        srt, idx = E.sort(descending=True); cum = srt.cumsum(0) / srt.sum(); n = int((cum < MASS).sum().item()) + 1
        sel = torch.zeros(NB, dtype=torch.bool, device="cuda"); sel[idx[:n]] = True; return sel
    res = {}; igla_blk = None; igla_mass = -1.0
    for hk in range(Hkv):
        Pq, Exq, Qq = true_mass(qrows, hk)                              # строки вопроса
        # блок иглы: наибольшая масса строк вопроса среди блоков вне стока (0) и вне локальных (последние 2)
        mq = Pq.sum(0).clone(); mq[0] = 0; mq[q0 // B:] = 0                 # вне стока и вне блоков последнего чанка
        dep = ig["depths"][0] / 100.0; lo = max(1, int((dep - 0.015) * T) // B); hi = min(NB, int((dep + 0.015) * T) // B + 1)
        win = mq[lo:hi]; jb = lo + int(win.argmax().item()); mval = float(Pq[:, jb].mean().item())   # блок иглы по глубине
        jc = int(mq.argmax().item()); res.setdefault("_ctx", []).append((jc, float(Pq[:, jc].mean().item())))
        if mval > igla_mass: igla_blk, igla_mass = jb, mval
        Pp, Exp_, Qp = true_mass(probes, hk)                            # зонды
        Pt, Ext, Qt = true_mass(tile_rows, hk)                          # вся плитка
        Sest = est_scores(Qp, hk)[:, :nvis]
        Sest_c = Sest.clone(); Sest_c[:, nvis:] = float("-inf")
        An = torch.softmax(Sest, dim=1).sum(0)                              # оценщик + НОРМИРОВКА ПО СТРОКЕ (кандидат в ядро)
        sets = {"A": select_by_mass(torch.exp(Sest - Sest.max()).sum(0)),
                "An": select_by_mass(An),
                "B": select_by_mass(Exp_.sum(0)[:nvis]),
                "C": select_by_mass(Pp.sum(0)[:nvis]),
                "E": select_by_mass(Pt.sum(0)[:nvis])}
        union = torch.zeros(NB, dtype=torch.bool, device="cuda")
        srt, idx = Pq.sort(1, descending=True); cum = srt.cumsum(1); need = (cum < MASS).sum(1) + 1
        for r in range(Pq.shape[0]): union[idx[r, :need[r]]] = True
        sets["D"] = union[:nvis]
        for nm, sel in sets.items():
            kept = float(Pq[:, :nvis][:, sel].sum() / Pq[:, :nvis].sum())    # удержано массы строк вопроса
            hit = bool(sel[jb].item())
            prev = res.get(nm, (0.0, 0.0, True)); res[nm] = (prev[0] + float(sel.float().mean()) / Hkv, prev[1] + kept / Hkv, prev[2] and hit)
    layer = re.search(r"qk\.(\d+)\.", f).group(1)
    cell = lambda nm: f"{100*res[nm][0]:5.1f}% {'+' if res[nm][2] else 'ПОТЕРЯН'} {100*res[nm][1]:4.0f}"
    ctx = max(res["_ctx"], key=lambda x: x[1])
    print(f"{layer:>4} {T:>7} {Sq:>4} | {igla_blk:>9} {100*igla_mass:9.2f}% | {cell('A'):>14} {cell('An'):>14} {cell('B'):>14} {cell('C'):>14} {cell('E'):>14} {cell('D'):>14} | лучший блок контекста {ctx[0]} ({100*ctx[1]:.2f}%)")
    del q, k, kp, dev, kbar; torch.cuda.empty_cache()
print("столбцы: доля блоков | блок иглы в наборе | удержано массы строк вопроса (%)")
