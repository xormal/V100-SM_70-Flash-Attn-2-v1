"""УСЛОВНОСТЬ НУЛЕВОЙ ТОЧКИ GPTQ -- РЕШАЕТСЯ ЗАМЕРОМ ПРОТИВ ЯДРА ДВИЖКА, А НЕ ЧТЕНИЕМ.

ЗАЧЕМ. Формат GPTQ существует в двух условностях: v1 хранит (истинный_ноль - 1), v2 -- истинный.
Промахнуться в ней значит сместить КАЖДЫЙ вес ровно на один квант, то есть на ~0.4 % размаха
группы -- вдвое грубее самой ошибки int8. Такой промах не падает и почти не виден на глаз, он
просто делает сеть хуже. Читать исходники ядра можно, но проверять всё равно придётся, поэтому
проверка идёт первой и она же служит доказательством.

КАК УСТРОЕНА ПРОВЕРКА. Берём ПЛОТНЫЙ вес с известными значениями, квантуем его своим упаковщиком,
скармливаем ЯДРУ ДВИЖКА (`ops.gptq_gemm`) и сравниваем с плотным умножением. Симметрия сломана:
эталон не проходит ни через одну строку упаковщика. Перебираются обе условности и оба пути ядра
(exllama и обычный) -- побеждает та, что даёт ошибку уровня самого квантования, а не вдвое больше.

    python tools/gptq_uslovnost.py
"""
import sys

import torch

БИТ = 8
G = 128


def упаковать_gptq(w, бит=БИТ, гр=G, минус1=False):
    """w: [K, N] fp32 (k-мажорно, как хранит GPTQ) -> (qweight, qzeros, scales).

    Раскладка (bits=8): qweight [K/4, N] int32 -- четыре подряд идущих k в одном слове;
    qzeros [K/G, N/4] int32 -- четыре подряд идущих n в слове; scales [K/G, N] fp16.
    Квантование МИНМАКСНОЕ по группе вдоль K: s = (max-min)/255, z = round(-min/s).
    `минус1` -- проверяемая условность: класть в файл (z-1) вместо z.
    """
    K, N = w.shape
    lvl = (1 << бит) - 1
    г = w.view(K // гр, гр, N)
    мин = г.amin(dim=1)
    макс = г.amax(dim=1)
    s = ((макс - мин) / lvl).clamp_min(1e-12)
    z = torch.round(-мин / s).clamp(0, lvl)
    q = torch.round(г / s.unsqueeze(1) + z.unsqueeze(1)).clamp(0, lvl).to(torch.int64)

    qw = torch.zeros(K // 4, N, dtype=torch.int64)
    qv = q.view(K, N)
    for i in range(4):
        qw |= (qv[i::4] & 0xFF) << (8 * i)
    zf = (z - 1).clamp(0, lvl) if минус1 else z
    zf = zf.to(torch.int64)
    qz = torch.zeros(K // гр, N // 4, dtype=torch.int64)
    for i in range(4):
        qz |= (zf[:, i::4] & 0xFF) << (8 * i)
    к32 = lambda t: torch.where(t >= (1 << 31), t - (1 << 32), t).to(torch.int32)   # noqa: E731
    return к32(qw), к32(qz), s.to(torch.float16), z, q


def main():
    from vllm import _custom_ops as ops
    dev = "cuda"
    torch.manual_seed(0)
    K, N, M = 1024, 512, 8
    w = (torch.randn(K, N) * 0.02).float()
    x = (torch.randn(M, K) * 0.5).half().to(dev)
    эт = x.float() @ w.to(dev)                     # ЭТАЛОН: плотное умножение, мимо упаковщика

    print(f"K={K} N={N} M={M}, группа {G}, разрядов {БИТ}")
    print(f"{'условность':>18s} {'путь':>10s} {'relL2':>10s}  вердикт")
    лучш = None
    for минус1, метка in ((False, "v2: z как есть"), (True, "v1: z-1")):
        qw, qz, s, z, q = упаковать_gptq(w, минус1=минус1)
        # ПОЛ ОШИБКИ: то, что даёт САМО квантование при верной условности. Ниже него не бывает.
        восст = (q.view(K, N).float() - z.repeat_interleave(G, dim=0).float()) * \
            s.float().repeat_interleave(G, dim=0)
        пол = ((x.float() @ восст.to(dev) - эт).norm() / эт.norm()).item()
        qzd, sd = qz.to(dev), s.to(dev)
        # ПОДГОТОВКА ПОВТОРЯЕТСЯ ДОСЛОВНО ЗА ДВИЖКОМ (gptq.py, process_weights_after_loading):
        # при desc_act=False g_idx кладётся ПУСТЫМ (не нулями!), и перед exllama обязателен
        # gptq_shuffle -- он переставляет вес НА МЕСТЕ. Первая редакция теста слала нули и не
        # звала перестановку: сошлось ноль комбинаций из восьми, и это был дефект теста, а не
        # раскладки. Отсюда правило: воспроизводить подготовку читателя целиком, а не по памяти.
        g_idx = torch.empty((0,), dtype=torch.int32, device=dev)
        # ТОЛЬКО exllama: при desc_act=False движок ходит ИМ ВСЕГДА, а обычный путь при пустом
        # g_idx делает АВАРИЙНОЕ ОБРАЩЕНИЕ к памяти и убивает контекст (проверено: первая редакция
        # теста падала на нём и уносила с собой все последующие комбинации).
        for v2 in (False, True):
            for exl in (True,):
                qwd = qw.to(dev).clone()          # перестановка портит вес -> копия на попытку
                if exl:
                    ops.gptq_shuffle(qwd, g_idx, БИТ)
                try:
                    y = ops.gptq_gemm(x, qwd, qzd, sd, g_idx, exl, v2, БИТ)
                except Exception as e:  # noqa: BLE001
                    print(f"{метка:>18s} {'exl' if exl else 'обыч':>4s}/v{2 if v2 else 1}  ОТКАЗ: {e}")
                    continue
                rel = ((y.float() - эт).norm() / эт.norm()).item()
                ок = rel < пол * 1.5
                print(f"{метка:>18s} {('exl' if exl else 'обыч') + '/v' + ('2' if v2 else '1'):>10s} "
                      f"{rel:>10.2e}  {'СОШЛОСЬ (пол %.2e)' % пол if ок else ''}")
                if ок and (лучш is None or rel < лучш[0]):
                    лучш = (rel, метка, exl, v2)
    print()
    if лучш is None:
        print("НИ ОДНА комбинация не сошлась -- писать слепок НЕЛЬЗЯ, раскладка не та.")
        return 2
    rel, метка, exl, v2 = лучш
    print(f"ВЕРДИКТ: класть нули как «{метка}», движку объявлять "
          f"checkpoint_format={'gptq_v2' if v2 else 'gptq'} (relL2 {rel:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
