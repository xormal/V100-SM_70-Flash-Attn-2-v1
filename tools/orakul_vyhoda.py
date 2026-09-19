# ОРАКУЛ ПО ВЫХОДУ: ошибка выхода внимания (‖O'−O‖/‖O‖ по плитке и p95 по строкам) при бюджете блоков f
# для трёх правил ранжирования блоков: (1) истинная масса плитки; (2) оценка E средним ключом (16 зондов, как ядро);
# (3) ЗНАЧЕНИЕ-осведомлённое: истинная масса x ‖v̄_J − Ō‖ (блоки со «средними» значениями выкидывать дешевле);
# (4) точный вклад блока в выход плитки: ‖Σ_r Σ_{j∈J} p_rj (v_j − O_r)‖ (оракул ошибки). Дамп FA2SM70_DUMP_QK (q,k,v).
import torch, sys
f = sys.argv[1]; budgets = [float(x) for x in sys.argv[2:]] or [0.30, 0.45, 0.60]
o = torch.load(f, map_location="cpu")
q = o["q"].cuda().half(); k = o["k"].cuda().half(); v = o["v"].cuda().half(); scale = o["scale"]; T = o["T"]; Sq = o["Sq"]; Hkv = o["Hkv"]; d = o["d"]
H = q.shape[1]; G = H // Hkv; B = 64; TI = 128; NB = (T + B - 1) // B; Tpad = NB * B; q0_abs = T - Sq
print(f"дамп {f}: Sq={Sq} T={T} H={H} Hkv={Hkv} d={d}; блоков {NB}, плиток {Sq//TI}")
kbar = torch.stack([k[:, J*B:(J+1)*B].float().mean(1) for J in range(NB)], 1)     # [Hkv, NB, d]
vbar = torch.stack([v[:, J*B:(J+1)*B].float().mean(1) for J in range(NB)], 1)     # [Hkv, NB, d]
jj = torch.arange(T, device="cuda")
pravila = ["масса(ист)", "E(ср.ключ)", "масса·‖v̄−Ō‖", "вклад в выход(ист)"]
rez = {b: {p: [] for p in pravila} for b in budgets}
rez95 = {b: {p: [] for p in pravila} for b in budgets}
def vyhod(rows, K, V, pos, keep_blocks):   # O' при маске блоков (None = плотный); возвращает [rows, d]
    outs = []
    for r0 in range(0, rows.shape[0], 128):
        S = (rows[r0:r0+128] @ K.t()) * scale
        S.masked_fill_(jj[None, :] > pos[r0:r0+128, None], float("-inf"))
        if keep_blocks is not None:
            S.masked_fill_(~keep_blocks[None, :T], float("-inf"))
        P = torch.softmax(S, 1); outs.append(P @ V)
    return torch.cat(outs, 0)
for hk in range(Hkv):
    K = k[hk].float(); V = v[hk].float()
    for t0 in range(0, Sq, TI):
        rows = q[t0:t0+TI, hk*G:(hk+1)*G].reshape(-1, d).float()
        pos = (q0_abs + t0 + torch.arange(TI, device="cuda").repeat_interleave(G))
        nvis = int(pos.max().item()) // B + 1
        # истинные величины плитки
        Mb = torch.zeros(rows.shape[0], NB, device="cuda"); PVsum = torch.zeros(NB, d, device="cuda"); vklad = torch.zeros(NB, device="cuda")
        O = vyhod(rows, K, V, pos, None)                                            # плотный выход [rows, d]
        for r0 in range(0, rows.shape[0], 128):
            S = (rows[r0:r0+128] @ K.t()) * scale
            S.masked_fill_(jj[None, :] > pos[r0:r0+128, None], float("-inf"))
            P = torch.softmax(S, 1)                                                 # [128, T]
            Pb = torch.zeros(P.shape[0], Tpad, device="cuda"); Pb[:, :T] = P
            Mb[r0:r0+128] = Pb.view(P.shape[0], NB, B).sum(2)
            # вклад блока в ошибку: Σ_r ‖Σ_{j∈J} p_rj (v_j − O_r)‖ -- считаем через P·V по блокам
            PVb = torch.zeros(P.shape[0], Tpad, d, device="cuda") if False else None
            # дешевле: для каждого блока J вектор Σ_j p_rj v_j по строкам r -> [128, NB, d] слишком велик; берём
            # приближение по плитке: vklad_J = ‖Σ_r Σ_{j∈J} p_rj v_j − Σ_r m_rJ O_r‖
            colmass = P.sum(0)                                                      # [T]
            cv = (colmass[:, None] * V)                                             # [T, d]
            cvp = torch.zeros(Tpad, d, device="cuda"); cvp[:T] = cv
            PVsum += cvp.view(NB, B, d).sum(1)
            vklad += torch.zeros(NB, device="cuda")  # заполним ниже
            del S, P, Pb, cv, cvp
        mass = Mb.sum(0)                                                            # [NB]
        Obar = O.mean(0)                                                            # [d]
        # точный вклад в выход плитки: Σ_r m_rJ O_r -> [NB, d]
        MO = Mb.t() @ O                                                             # [NB, d]
        vklad = (PVsum - MO).norm(dim=1)                                            # [NB]
        # оценка E средним ключом по 16 зондам
        probe = rows.view(TI, G, d)[::8].reshape(-1, d)
        E = (probe @ kbar[hk, :nvis].t()) * scale; E = torch.exp(E - E.max()).sum(0)
        Efull = torch.zeros(NB, device="cuda"); Efull[:nvis] = E
        kriterii = {"масса(ист)": mass, "E(ср.ключ)": Efull,
                    "масса·‖v̄−Ō‖": mass * (vbar[hk] - Obar[None, :]).norm(dim=1),
                    "вклад в выход(ист)": vklad}
        On = O.norm()
        for b in budgets:
            nkeep = max(1, int(round(b * nvis)))
            for p, kr in kriterii.items():
                kr2 = kr.clone(); kr2[nvis:] = -1; kr2[0] = 1e30          # блок 0 (sink) всегда
                idx = kr2.topk(nkeep).indices
                keep = torch.zeros(NB, dtype=torch.bool, device="cuda"); keep[idx] = True
                keep_tok = keep.repeat_interleave(B)
                O2 = vyhod(rows, K, V, pos, keep_tok)
                err = (O2 - O).norm() / On
                rowerr = (O2 - O).norm(dim=1) / O.norm(dim=1).clamp_min(1e-6)
                rez[b][p].append(float(err)); rez95[b][p].append(float(rowerr.quantile(0.95)))
def st(vv): vv = sorted(vv); n = len(vv); return f"{100*vv[n//2]:5.2f} % (макс {100*vv[-1]:5.2f} %)"
for b in budgets:
    print(f"бюджет {100*b:.0f} % блоков: относительная ошибка выхода плитки, медиана (макс) | p95 по строкам, медиана")
    for p in pravila:
        print(f"  {p:20s}: {st(rez[b][p])} | {st(rez95[b][p])}")
