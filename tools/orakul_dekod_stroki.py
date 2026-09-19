# ОРАКУЛ ДЕКОДНОЙ СТРОКИ: может ли ОДНА строка (как в декоде) найти блок иглы по оценщику отбора?
# По дампам q/K последнего чанка: для строк вопроса (последние QR строк x H голов) ранг блока иглы в порядке
# оценщика s=(q*kbar + gamma*sum|q_d|dev_d)*scale (и в порядке ТОЧНЫХ масс) -> доля блоков, которую надо взять,
# чтобы игла попала (медиана / 90-й перцентиль по строкам). CPU.
import torch, sys, glob, json, re
ig = json.load(open(sys.argv[1])); D = sys.argv[2]; QR = int(sys.argv[3]) if len(sys.argv) > 3 else 40
B = 128; GAMMA = 0.5; torch.set_num_threads(16)
files = sorted(glob.glob(D + "/qk.*.r0.pt"), key=lambda f: int(re.search(r"qk\.(\d+)\.", f).group(1)))
dep = ig["depths"][0] / 100.0
print(f"игла {ig['needles'][0][0]} глубина {ig['depths'][0]}%; дампов {len(files)}; строк вопроса {QR} x голов")
print(f"{'слой':>4} | {'масса иглы у искателей':>18} | {'доля блоков до иглы, ОЦЕНЩИК: медиана / p90 / макс':>50} | {'ТОЧНЫЕ счёты: медиана / p90':>28} | {'искателей с иглой в top-10% / всего':>22}")
for f in files:
    o = torch.load(f, map_location="cpu"); q = o["q"].float(); k = o["k"].float(); scale = o["scale"]; T = o["T"]; Sq = o["Sq"]; Hkv = o["Hkv"]; d = o["d"]
    H = q.shape[1]; G = H // Hkv; NB = (T + B - 1) // B; q0 = T - Sq; nvis = (T - 1) // B + 1
    kp = torch.zeros(Hkv, NB * B, d); kp[:, :T] = k; kp = kp.view(Hkv, NB, B, d)
    cnt = torch.full((NB,), float(B)); cnt[-1] = T - (NB - 1) * B
    kbar = kp.sum(2) / cnt[None, :, None]; dev = (kp - kbar[:, :, None, :]).abs()
    if T % B: dev[:, -1, T - (NB - 1) * B:, :] = 0
    dev = dev.amax(2)
    lo = max(1, int((dep - 0.015) * T) // B); hi = min(NB, int((dep + 0.015) * T) // B + 1)
    rows = list(range(Sq - QR, Sq)); fr_est = []; fr_true = []; masses = []; top10 = 0; nrows = 0
    for hk in range(Hkv):
        Q = q[rows][:, hk * G:(hk + 1) * G].reshape(-1, d)                       # [QR*G, d]
        S = (Q @ k[hk].t()) * scale                                              # строки вопроса видят все ключи (они последние)
        P = torch.softmax(S, 1); Pb = torch.cat([P, torch.zeros(P.shape[0], NB * B - T)], 1).view(P.shape[0], NB, B).sum(2)
        mq = Pb.sum(0); mq[0] = 0; mq[q0 // B:] = 0
        jb = lo + int(mq[lo:hi].argmax().item())                                 # блок иглы по глубине
        est = (Q @ kbar[hk].t() + GAMMA * (Q.abs() @ dev[hk].t())) * scale       # [R, NB]
        est = est[:, :nvis]; Pb = Pb[:, :nvis]
        rk_e = (est > est[:, jb:jb + 1]).sum(1).float() / nvis                   # доля блоков строго выше иглы = доля, которую надо взять
        rk_t = (Pb > Pb[:, jb:jb + 1]).sum(1).float() / nvis
        sel = Pb[:, jb] > 0.05                                                   # ТОЛЬКО строки-искатели: масса иглы > 5 %
        fr_est += rk_e[sel].tolist(); fr_true += rk_t[sel].tolist(); masses += Pb[sel, jb].tolist()
        top10 += int((rk_e[sel] <= 0.10).sum().item()); nrows += int(sel.sum().item())
    layer = re.search(r"qk\.(\d+)\.", f).group(1)
    if not fr_est: print(f"{layer:>4} | строк-искателей нет"); continue
    fe = torch.tensor(fr_est); ft = torch.tensor(fr_true); m = torch.tensor(masses)
    print(f"{layer:>4} | {100*m.median():8.2f}% / {100*m.max():6.2f}% | {100*fe.median():7.1f}% / {100*fe.quantile(0.9):5.1f}% / {100*fe.max():5.1f}% | {100*ft.median():7.1f}% / {100*ft.quantile(0.9):5.1f}% | {top10:5d}/{nrows}")
