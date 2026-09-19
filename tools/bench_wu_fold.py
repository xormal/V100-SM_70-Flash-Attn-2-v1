"""ПАРНЫЙ A/B ВТОРОГО СРЕЗА: пять пусков против трёх, на БОЕВОМ виде.

Меряется не «ядро против ядра», а ЦЕПОЧКА против ЦЕПОЧКИ -- потому что ход убирает ПУСК, а не
ускоряет тело. Обе стороны вызываются из ОДНОГО процесса с чередованием внутри раунда, медиана
берётся по ОТНОШЕНИЯМ (иначе дрейф частот попадает в результат как эффект правки).

    python tools/bench_wu_fold.py /tmp/claude-1000/gdn_boevoi.pt [--куски 4096 8192 16384]
"""
import argparse, os, sys, time, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from check_ho import подготовить  # noqa: E402  -- тот же вход, что у сверки


def мера(фн, n=20, раундов=5):
    for _ in range(5):
        фн()
    torch.cuda.synchronize()
    из = []
    for _ in range(раундов):
        t0 = time.perf_counter()
        for _ in range(n):
            фн()
        torch.cuda.synchronize()
        из.append((time.perf_counter() - t0) / n * 1e3)
    из.sort()
    return из[len(из) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("дамп")
    ap.add_argument("--куски", type=int, nargs="*", default=[3136])
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    d = torch.load(a.дамп, map_location="cpu", weights_only=False)
    q, k, v, g, beta, h0, cu, scale = подготовить(d, dev)
    from fa2_sm70 import gdn as G
    ext = G.gdn_ext()

    print(f"{'токенов':>8} {'пять пусков':>12} {'три пуска':>11} {'отношение':>10} "
          f"{'снято МБ/слой':>14}  чанков")
    for T in a.куски:
        T = min(T, q.shape[-3])
        NTc = (T + 63) // 64
        qc, kc, vc, gc, bc = (x[:, :T].contiguous() for x in (q, k, v, g, beta))
        cuc = torch.tensor([0, T], dtype=cu.dtype, device=cu.device)
        cu32, ci, co, NTk = G._chunk_meta(cuc)
        H = vc.shape[-2]
        v16, b16 = vc.to(torch.float16).contiguous(), bc.to(torch.float16).contiguous()

        def пять():
            A = ext.gdn_kkt(kc, b16, gc, cu32, ci, H)
            Ai = ext.gdn_solve_tril(A, cu32, ci)
            w, u = ext.gdn_wu(kc, v16, b16, gc, Ai, cu32, ci)
            return ext.gdn_ho(qc, kc, w, u, gc, h0, cu32, co, NTk, scale, True, False)

        def три():
            A = ext.gdn_kkt(kc, b16, gc, cu32, ci, H)
            Ai = ext.gdn_solve_tril(A, cu32, ci)
            return ext.gdn_ho(qc, kc, kc, v16, gc, h0, cu32, co, NTk, scale, True, False,
                              Ai=Ai, beta=b16)

        # ЧЕРЕДОВАНИЕ ВНУТРИ РАУНДА: иначе медленный дрейф карты уедет в разность путей
        от = []
        for _ in range(5):
            t5 = мера(пять, n=10, раундов=1)
            t3 = мера(три, n=10, раундов=1)
            от.append(t5 / t3)
        от.sort()
        м5, м3 = мера(пять, n=10), мера(три, n=10)
        снято = T * H * (128 + 128) * 2 / 1e6      # w [T,H,DK] + u [T,H,DV], fp16
        print(f"{T:>8} {м5:>10.3f} мс {м3:>9.3f} мс {от[len(от)//2]:>9.3f}x "
              f"{снято:>13.1f}  {NTc}")


if __name__ == "__main__":
    main()
