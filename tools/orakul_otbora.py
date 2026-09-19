# ОРАКУЛ ОТБОРА: по настоящим q/K (дамп FA2SM70_DUMP_QK) считает, сколько блоков ключей (B=64) нужно на плитку
# запросов (TI=128 строк x 6 q-голов одной kv-головы), чтобы удержать долю MASS ИСТИННОЙ массы софтмакса:
#   (а) по плитке (сумма масс всех строк плитки -- критерий нынешнего правила по массе);
#   (б) объединение по строкам (каждая строка держит MASS своей массы -- гарантия на иглу);
#   (в) оценщик «средний ключ»: E_J = sum_rows exp(q*kbar_J - M) по 16 зондам плитки (как в ядре, без поправок).
# Аргументы: файл дампа [MASS...]. Карта: CUDA_VISIBLE_DEVICES.
import torch, sys, math
f = sys.argv[1]; masses = [float(x) for x in sys.argv[2:]] or [0.95, 0.98, 0.99]
o = torch.load(f, map_location="cpu")
q = o["q"].cuda().half(); k = o["k"].cuda().half(); scale = o["scale"]; T = o["T"]; Sq = o["Sq"]; Hkv = o["Hkv"]; d = o["d"]
# q: [Sq, H, d] (запросы чанка, позиции T-Sq..T-1), k: [Hkv, T, d]
H = q.shape[1]; G = H // Hkv; B = 64; TI = 128
NB = (T + B - 1) // B
print(f"дамп {f}: Sq={Sq} T={T} H={H} Hkv={Hkv} d={d}; блоков ключей {NB}, плиток запросов {Sq // TI}")
kbar = torch.zeros(Hkv, NB, d, device="cuda", dtype=torch.float32)
for J in range(NB):
    kbar[:, J] = k[:, J * B:(J + 1) * B].float().mean(1)
res = {m: {"tile": [], "rows": [], "est": []} for m in masses}
q0_abs = T - Sq   # абсолютная позиция первой строки чанка (причинность: строка i видит ключи <= q0_abs+i)
for hk in range(Hkv):
    K = k[hk]                                      # [T, d]
    for t0 in range(0, Sq, TI):
        rows = q[t0:t0 + TI, hk * G:(hk + 1) * G].reshape(-1, d).float()     # [TI*G, d] (строка r = запрос r//G, голова r%G)
        pos = (q0_abs + t0 + torch.arange(TI, device="cuda").repeat_interleave(G))
        jj = torch.arange(T, device="cuda"); Tpad = NB * B
        Mb = torch.zeros(rows.shape[0], NB, device="cuda")
        for r0 in range(0, rows.shape[0], 128):                                # по 128 строк, чтобы не держать S целиком
            S = (rows[r0:r0 + 128] @ K.float().t()) * scale                    # [128, T]
            S.masked_fill_(jj[None, :] > pos[r0:r0 + 128, None], float("-inf"))   # причинность
            P = torch.softmax(S, dim=1)
            Pb = torch.zeros(P.shape[0], Tpad, device="cuda"); Pb[:, :T] = P
            Mb[r0:r0 + 128] = Pb.view(P.shape[0], NB, B).sum(2)
            del S, P, Pb
        nvis = int(pos.max().item()) // B + 1                                  # видимых блоков плитке
        # (а) по плитке
        tile_mass = Mb.sum(0)                                                   # [NB]
        srt, _ = tile_mass.sort(descending=True); cum = srt.cumsum(0) / srt.sum()
        # (б) объединение по строкам
        srt_r, idx_r = Mb.sort(dim=1, descending=True); cum_r = srt_r.cumsum(1)
        # (в) оценщик по 16 зондам: строки 0,8,16,... каждой из 6 голов -> берём каждую 8-ю строку плитки для всех голов
        probe = rows.view(TI, G, d)[::8].reshape(-1, d)                         # [16*G, d]
        E = (probe @ kbar[hk, :nvis].t()) * scale                              # [16G, nvis]
        E = torch.exp(E - E.max()).sum(0)                                      # [nvis]
        srt_e, idx_e = E.sort(descending=True); cum_e = srt_e.cumsum(0) / srt_e.sum()
        for m in masses:
            n_tile = int((cum < m).sum().item()) + 1
            need_r = (cum_r < m).sum(1) + 1                                     # блоков на строку
            union = torch.zeros(NB, dtype=torch.bool, device="cuda")
            for r in range(Mb.shape[0]):
                union[idx_r[r, :need_r[r]]] = True
            n_rows = int(union.sum().item())
            n_est = int((cum_e < m).sum().item()) + 1
            # сколько ИСТИННОЙ массы плитки удерживает набор оценщика
            keep = idx_e[:n_est]; kept = tile_mass[keep].sum() / tile_mass.sum()
            res[m]["tile"].append(n_tile / nvis); res[m]["rows"].append(n_rows / nvis); res[m]["est"].append((n_est / nvis, float(kept)))
def st(v):
    v = sorted(v); n = len(v); return f"медиана {100*v[n//2]:5.1f} %  среднее {100*sum(v)/n:5.1f} %  макс {100*v[-1]:5.1f} %"
for m in masses:
    print(f"MASS={m}:")
    print(f"  оракул по плитке      : {st(res[m]['tile'])}")
    print(f"  оракул по строкам (∪) : {st(res[m]['rows'])}")
    e = [x[0] for x in res[m]["est"]]; kk = [x[1] for x in res[m]["est"]]
    print(f"  оценщик средний ключ  : {st(e)}   удержано истинной массы: {st(kk)}")
