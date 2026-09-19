"""ПРИЁМКА W12A16. Ломает симметрию: упаковку пишет python, разбор -- CUDA.

Дисциплина (CLAUDE.md §4): две сверки, читающие один буфер, соглашаются, будучи обе неверны.
Поэтому эталон здесь НЕ переиспользует ни одной строки разбора из ядра -- он строится из ЦЕЛЫХ q и
масштабов, полученных ДО упаковки. Если раскладка в python и разбор в CUDA разойдутся хоть на бит,
разойдётся и ответ.

Проверяется ТРИ вещи, и порядок не случаен:
  1. распаковщик: CUDA-разбор упакованного == q*s (побитовая раскладка);
  2. GEMV на боевых формах и боевых M (декод) -- против плотного эталона;
  3. скорость против плотного fp16 -- ГДЕ перелом, а не «быстрее ли».
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from quantize_iN import квантовать, упаковать, разобрать   # noqa: E402
from fa2_sm70._ext import w12a16_ext                        # noqa: E402

ФОРМЫ = [("q_proj", 12288, 5120), ("k_proj", 1024, 5120), ("v_proj", 1024, 5120)]
G = 128


def подготовить(N, K, dev):
    w = torch.randn(N, K, dtype=torch.float32) * 0.02
    пак, масш, ош, qi = квантовать(w, 12, G)          # пак [K/8,3,N], масш [K/G,N], qi [N,K]
    # ЭТАЛОН ВЕСА -- из целых, а не из разбора: q[n,k] * s[k/G, n]
    s_nk = масш.float().t().repeat_interleave(G, dim=1)      # [N, K]
    wq = qi.float() * s_nk
    return пак.to(dev), масш.to(dev), wq.to(dev), ош


def main():
    dev = "cuda"
    e = w12a16_ext()
    torch.manual_seed(0)
    плохо = 0

    print("=== 1. РАСПАКОВЩИК: CUDA-разбор против целых, посчитанных ДО упаковки ===")
    for имя, N, K in ФОРМЫ:
        пак, масш, wq, ош = подготовить(N, K, dev)
        buf = torch.empty(K * N, dtype=torch.float16, device=dev)
        e.w12_dequant_into(пак, масш, buf, G)
        got = buf.view(K, N).t().float()                     # [N, K]
        d = (got - wq).abs().max().item()
        # порог -- ровно зернистость fp16 у самого крупного веса, не «на глаз»
        порог = wq.abs().max().item() * 2 ** -11
        ok = d <= порог
        плохо += not ok
        print(f"  {имя:8s} [{N},{K}] max|d| {d:.3e} порог {порог:.3e}  {'OK' if ok else 'РАСХОЖДЕНИЕ'}")
        # тождество упаковки на самом же буфере (python-разбор -- третий независимый путь)
        assert torch.equal(разобрать(пак.cpu(), 12), (wq.cpu() / масш.cpu().float().t()
                                                      .repeat_interleave(G, dim=1)).round().int()), \
            f"python-разбор разошёлся на {имя}"
        del пак, масш, wq, buf, got

    print("\n=== 2. GEMV против плотного эталона (M -- боевые для декода) ===")
    print(f"  {'форма':10s} {'M':>4s} {'relL2':>10s} {'вердикт'}")
    for имя, N, K in ФОРМЫ:
        пак, масш, wq, ош = подготовить(N, K, dev)
        for M in (1, 2, 4, 8, 16):
            x = (torch.randn(M, K, device=dev) * 0.5).half()
            y = e.w12a16_gemv(x, пак, масш, G)
            ref = x.float() @ wq.t()
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            # эталон сам в fp32, ядро копит в fp32 и пишет fp16 -> класс ошибки fp16, 1e-3
            ok = rel < 2e-3
            плохо += not ok
            print(f"  {имя:10s} {M:>4d} {rel:>10.2e}  {'OK' if ok else 'РАСХОЖДЕНИЕ'}")
        del пак, масш, wq

    print("\n=== 3. ГДЕ ПЕРЕЛОМ: упакованный путь против плотного fp16 ===")
    print(f"  {'форма':10s} {'M':>4s} {'упак,мкс':>9s} {'плотн,мкс':>10s} {'x':>6s}")
    for имя, N, K in ФОРМЫ:
        пак, масш, wq, ош = подготовить(N, K, dev)
        wd = wq.half().contiguous()
        for M in (1, 2, 4, 8, 16, 32):
            x = (torch.randn(M, K, device=dev) * 0.5).half()
            def т(f, n=30):
                for _ in range(5):
                    f()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(n):
                    f()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / n * 1e6
            tp = т(lambda: e.w12a16_gemv(x, пак, масш, G))
            td = т(lambda: torch.matmul(x, wd.t()))
            print(f"  {имя:10s} {M:>4d} {tp:>9.1f} {td:>10.1f} {td/tp:>6.2f}")
        del пак, масш, wq, wd

    print("\nИТОГ:", "ВСЁ СОШЛОСЬ" if плохо == 0 else f"РАСХОЖДЕНИЙ {плохо}")
    return 1 if плохо else 0


if __name__ == "__main__":
    sys.exit(main())
