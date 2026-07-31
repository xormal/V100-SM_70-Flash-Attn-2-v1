"""ОТБОРЩИК ПРОТИВ ОРАКУЛА на НАСТОЯЩЕМ внимании модели.

Механизм без работающего отборщика ничего не стоит: ускорение от разреженности покупается только
той точностью, которую отборщик способен удержать. Оракул (tools/attn_oracle.py) отбирает по
ИСТИННОЙ массе -- это ПОТОЛОК. Здесь мерится, сколько от потолка добирает наш РЕАЛИЗУЕМЫЙ отбор:
по сводкам плиток (средний ключ + радиус), тот самый, что считает select_blocks на тензорных ядрах.

Метод и метрика ТЕ ЖЕ, что у оракула, буква в букву: relL2 ВЫХОДА, softmax ПЕРЕНОРМИРУЕТСЯ по
оставшимся, причинная маска, fp32. Отличается РОВНО ОДНА строка -- чем ранжируются плитки.
Иначе сравнение было бы не с оракулом, а с другим замером.
"""
import sys, os, json, math
import torch

D = os.environ.get("ATTN_DUMP_DIR")
if not D:
    raise SystemExit("set ATTN_DUMP_DIR to the external directory containing layerNN_rankN.pt")
BQ = BK = 64
DENS = [0.50, 0.65, 0.75, 0.85, 0.95]
dev = "cuda"


def rank_oracle(p, nb, end, vv, qb, kk, sc):
    """истинная масса плитки, усреднённая по запросам блока -- ПОТОЛОК"""
    pad = nb * BK - end
    pb = torch.nn.functional.pad(p, (0, pad)).view(BQ, nb, BK)
    return pb.sum(-1).mean(0)


def rank_mean(p, nb, end, vv, qb, kk, sc):
    """реализуемый отбор: max по запросам от q.kbar (только средний ключ)"""
    pad = nb * BK - end
    kp = torch.nn.functional.pad(kk[:end], (0, 0, 0, pad)).view(nb, BK, -1)
    kbar = kp.sum(1) / torch.clamp(
        torch.nn.functional.pad(torch.ones(end, device=dev), (0, pad)).view(nb, BK).sum(1, keepdim=True), min=1)
    return (qb @ kbar.T * sc).max(0).values


def rank_mean_rad(p, nb, end, vv, qb, kk, sc):
    """реализуемый отбор: max по запросам от (q.kbar + ||q||*rmax) -- со радиусом плитки"""
    pad = nb * BK - end
    ones = torch.nn.functional.pad(torch.ones(end, device=dev), (0, pad)).view(nb, BK)
    kp = torch.nn.functional.pad(kk[:end], (0, 0, 0, pad)).view(nb, BK, -1)
    cnt = torch.clamp(ones.sum(1, keepdim=True), min=1)
    kbar = kp.sum(1) / cnt
    r2 = ((kp - kbar.unsqueeze(1)) ** 2).sum(-1)
    r2 = r2 * ones                                     # дополненные строки не участвуют
    rmax = r2.max(1).values.sqrt()
    qn = qb.norm(dim=-1)
    s = qb @ kbar.T * sc                               # [BQ, nb]
    s = s + (qn.unsqueeze(1) * rmax.unsqueeze(0)) * sc
    return s.max(0).values


def _blocks(kk, end, nb):
    pad = nb * BK - end
    ones = torch.nn.functional.pad(torch.ones(end, device=dev), (0, pad)).view(nb, BK)
    kp = torch.nn.functional.pad(kk[:end], (0, 0, 0, pad)).view(nb, BK, -1)
    return kp, ones


def rank_bound(p, nb, end, vv, qb, kk, sc):
    """ТОЧНАЯ верхняя граница максимума в плитке, а не Коши-Буняковский:
        max_{i in j} q.k_i <= sum_d [ q_d^+ * kmax_jd + q_d^- * kmin_jd ]
    (покомпонентно, потому что каждое слагаемое максимизируется независимо). Сводка -- ДВЕ:
    покомпонентный максимум и минимум ключей плитки. Считается ДВУМЯ GEMM тем же оператором:
    q^+ x kmax и q^- x kmin, то есть ровно вдвое дороже среднего ключа."""
    kp, ones = _blocks(kk, end, nb)
    msk = (ones == 0).unsqueeze(-1)
    kmax = kp.masked_fill(msk, -1e30).max(1).values
    kmin = kp.masked_fill(msk, 1e30).min(1).values
    qp = qb.clamp(min=0); qm = qb.clamp(max=0)
    s = (qp @ kmax.T + qm @ kmin.T) * sc
    return s.max(0).values


def rank_lse(p, nb, end, vv, qb, kk, sc):
    """logsumexp по запросам вместо максимума: масса плитки складывается по ВСЕМ 64 запросам,
    а не определяется одним. Стоит столько же -- редукция другая, GEMM тот же."""
    kp, ones = _blocks(kk, end, nb)
    cnt = torch.clamp(ones.sum(1, keepdim=True), min=1)
    kbar = kp.sum(1) / cnt
    s = qb @ kbar.T * sc
    return torch.logsumexp(s, dim=0)


def rank_bound_lse(p, nb, end, vv, qb, kk, sc):
    kp, ones = _blocks(kk, end, nb)
    msk = (ones == 0).unsqueeze(-1)
    kmax = kp.masked_fill(msk, -1e30).max(1).values
    kmin = kp.masked_fill(msk, 1e30).min(1).values
    qp = qb.clamp(min=0); qm = qb.clamp(max=0)
    s = (qp @ kmax.T + qm @ kmin.T) * sc
    return torch.logsumexp(s, dim=0)


RANKERS = {"oracle": rank_oracle, "mean": rank_mean, "mean+rad": rank_mean_rad,
           "mean+lse": rank_lse, "minmax": rank_bound, "minmax+lse": rank_bound_lse}


def one(L, rank=0, nqb=12, heads=None):
    d = torch.load(f"{D}/layer{L:02d}_rank{rank}.pt", map_location="cpu", weights_only=False)
    q, k, v = d["q"], d["k"], d["v"]
    T, H, dk = q.shape
    Hkv = k.shape[1]; sc = dk ** -0.5
    k = k.to(dev).float(); v = v.to(dev).float()
    hs = range(H) if heads is None else heads
    acc = {r: {p: [0.0, 0.0] for p in DENS} for r in RANKERS}
    for h in hs:
        kk = k[:, h * Hkv // H] if H >= Hkv else k[:, 0]
        vv = v[:, h * Hkv // H] if H >= Hkv else v[:, 0]
        starts = torch.linspace(T // 8, T - BQ, nqb).long().tolist()
        for s0 in starts:
            qb = q[s0:s0 + BQ, h].to(dev).float()
            end = s0 + BQ
            sco = qb @ kk[:end].T * sc
            i = torch.arange(BQ, device=dev).view(-1, 1) + s0
            j = torch.arange(end, device=dev).view(1, -1)
            sco = sco.masked_fill(j > i, float("-inf"))
            p = torch.softmax(sco, -1)
            od = p @ vv[:end]
            nb = (end + BK - 1) // BK
            for rn, fn in RANKERS.items():
                mass = fn(p, nb, end, vv, qb, kk, sc).clone()
                mass[0] = float("inf"); mass[nb - 1] = float("inf")   # ЯКОРИ, как в ядре
                order = mass.argsort(descending=True)
                for frac in DENS:
                    nkp = max(2 if nb >= 2 else 1, int(round(frac * nb)))
                    keep = torch.zeros(nb, device=dev, dtype=torch.bool); keep[order[:nkp]] = True
                    m = keep.view(1, nb, 1).expand(BQ, nb, BK).reshape(BQ, -1)[:, :end]
                    ps = p * m
                    ps = ps / ps.sum(-1, keepdim=True).clamp_min(1e-30)
                    osp = ps @ vv[:end]
                    acc[rn][frac][0] += (osp - od).pow(2).sum().item()
                    acc[rn][frac][1] += od.pow(2).sum().item()
        del kk, vv
    del k, v; torch.cuda.empty_cache()
    return {rn: {f"{p:.2f}": (acc[rn][p][0] / max(acc[rn][p][1], 1e-30)) ** 0.5 for p in DENS}
            for rn in RANKERS}


if __name__ == "__main__":
    layers = [int(x) for x in sys.argv[1].split(",")]
    nqb = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    res = {}
    for L in layers:
        r = one(L, nqb=nqb)
        res[L] = r
        print(f"=== слой дампа {L:2d} (модели {3+4*L:2d}) ===")
        print("  отбор      " + " ".join(f"{x:>9}" for x in [f"{p:.0%}" for p in DENS]))
        for rn in RANKERS:
            print(f"  {rn:<10} " + " ".join(f"{r[rn][f'{p:.2f}']:9.2e}" for p in DENS))
        best = {}
        for p in DENS:
            kk_ = f"{p:.2f}"
            cand = min((rn for rn in RANKERS if rn != "oracle"), key=lambda rn: r[rn][kk_])
            best[kk_] = cand
        print("  ЛУЧШИЙ реализуемый:  " + " ".join(f"{best[f'{p:.2f}']:>9}" for p in DENS))
        print("  ЦЕНА РЕАЛИЗУЕМОСТИ (лучший / оракул): " +
              " ".join(f"{r[best[f'{p:.2f}']][f'{p:.2f}']/max(r['oracle'][f'{p:.2f}'],1e-30):6.2f}x" for p in DENS))
        sys.stdout.flush()
    print("\n" + json.dumps(res, indent=1))
