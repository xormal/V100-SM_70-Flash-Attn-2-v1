"""ОРАКУЛ РАЗРЕЖЕННОСТИ на настоящем внимании модели.
Считает ПОТОЛОК направления: при отборе блоков по ИСТИННОЙ массе (то есть оптимальном) какая
relL2 ВЫХОДА получается при заданной плотности. Ошибка мерится на выходе, а не на массе, и
софтмакс ПЕРЕНОРМИРУЕТСЯ по оставшимся -- ровно то, что делает ядро."""
import os, sys, json, torch
D = os.environ.get("ATTN_DUMP_DIR")
if not D:
    raise SystemExit("set ATTN_DUMP_DIR to the external directory containing layerNN_rankN.pt")
BQ = BK = 64
DENS = [0.50, 0.65, 0.75, 0.85, 0.95]
layers = [int(x) for x in sys.argv[1].split(",")]
nqb = int(sys.argv[2]) if len(sys.argv) > 2 else 24      # сколько блоков запросов сэмплировать
dev = "cuda"

def one(L, rank=0):
    d = torch.load(f"{D}/layer{L:02d}_rank{rank}.pt", map_location="cpu", weights_only=False)
    q, k, v = d["q"], d["k"], d["v"]
    T, H, dk = q.shape
    Hkv = k.shape[1]; sc = dk ** -0.5
    k = k.to(dev).float(); v = v.to(dev).float()
    out = {}
    for h in range(H):
        kk = k[:, h * Hkv // H] if H >= Hkv else k[:, 0]
        vv = v[:, h * Hkv // H] if H >= Hkv else v[:, 0]
        acc = {p: [0.0, 0.0] for p in DENS}          # [сумма кв.ошибки, сумма кв.нормы]
        # блоки запросов, равномерно по всей длине, но не слишком короткие контексты
        starts = torch.linspace(T // 8, T - BQ, nqb).long().tolist()
        for s0 in starts:
            qb = q[s0:s0 + BQ, h].to(dev).float()
            end = s0 + BQ
            sco = qb @ kk[:end].T * sc               # [BQ, end]
            i = torch.arange(BQ, device=dev).view(-1, 1) + s0
            j = torch.arange(end, device=dev).view(1, -1)
            sco = sco.masked_fill(j > i, float("-inf"))
            p = torch.softmax(sco, -1)               # [BQ, end]
            od = p @ vv[:end]                        # плотный эталон
            nb = (end + BK - 1) // BK
            pad = nb * BK - end
            pb = torch.nn.functional.pad(p, (0, pad)).view(BQ, nb, BK)
            mass = pb.sum(-1).mean(0)                # масса блока, усреднённая по запросам блока
            order = mass.argsort(descending=True)
            for frac in DENS:
                nk = max(1, int(round(frac * nb)))
                keep = torch.zeros(nb, device=dev, dtype=torch.bool); keep[order[:nk]] = True
                m = keep.view(1, nb, 1).expand(BQ, nb, BK).reshape(BQ, -1)[:, :end]
                ps = p * m
                ps = ps / ps.sum(-1, keepdim=True).clamp_min(1e-30)
                osp = ps @ vv[:end]
                acc[frac][0] += (osp - od).pow(2).sum().item()
                acc[frac][1] += od.pow(2).sum().item()
        out[h] = {f"{p:.2f}": (acc[p][0] / max(acc[p][1], 1e-30)) ** 0.5 for p in DENS}
        del kk, vv
    del k, v; torch.cuda.empty_cache()
    return out

res = {}
for L in layers:
    r = one(L)
    res[L] = r
    hs = sorted(r)
    print(f"=== слой дампа {L:2d} (модели {3+4*L:2d}) ===")
    print("  гол " + " ".join(f"{p:>8}" for p in [f"{x:.0%}" for x in DENS]))
    for h in hs:
        print(f"  h{h:<2d} " + " ".join(f"{r[h][f'{p:.2f}']:8.1e}" for p in DENS))
    best = {f"{p:.2f}": min(r[h][f"{p:.2f}"] for h in hs) for p in DENS}
    wors = {f"{p:.2f}": max(r[h][f"{p:.2f}"] for h in hs) for p in DENS}
    print("  ЛУЧШ " + " ".join(f"{best[f'{p:.2f}']:8.1e}" for p in DENS))
    print("  ХУДШ " + " ".join(f"{wors[f'{p:.2f}']:8.1e}" for p in DENS))
json.dump(res, open(f"{D}/../oracle_result.json","w"), indent=1)
print("\nсохранено в oracle_result.json")
