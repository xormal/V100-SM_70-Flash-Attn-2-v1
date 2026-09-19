"""ОРАКУЛ ДЛЯ СЛИЯНИЯ h+o, НА БОЕВОМ ВИДЕ.

Отладка слитого тела идёт против ТОЙ ЖЕ пятёрки ядер, что работает сейчас, на ТЕХ ЖЕ тензорах,
которые сняты из живого сервера (`FA2SM70_GDN_ДАМП`). Сверка -- ПОСРЕЗНО: A, Ai, w, u, h, v_new,
ht, o. Косинус на выходе приёмкой не является: он усредняет и прячет ошибку в одном фрагменте.

    python tools/check_ho.py /tmp/claude-1000/gdn_boevoi.pt          # показать срезы образца
    python tools/check_ho.py <дамп> --срез o                          # сверить конкретный срез

Подготовка входа повторяет то, что делает подмена ПЕРЕД вызовом ядер (l2norm + накопленный гейт),
иначе оракул считал бы от других чисел, чем боевой путь.
"""
import argparse, os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def подготовить(d, dev):
    """Ровно те преобразования, что стоят в шиме до вызова: нормировка q,k и накопление гейта."""
    from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
    q, k = d["q"].to(dev), d["k"].to(dev)
    v, g, beta = d["v"].to(dev), d["g"].to(dev), d["beta"].to(dev)
    cu = None if d["cu_seqlens"] is None else d["cu_seqlens"].to(dev)
    h0 = None if d["initial_state"] is None else d["initial_state"].to(dev)
    if d["use_qk_l2norm_in_kernel"]:
        q, k = l2norm_fwd(q), l2norm_fwd(k)
    gg = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu)
    scale = d["scale"] if d["scale"] is not None else k.shape[-1] ** -0.5
    return (q.to(torch.float16).contiguous(), k.to(torch.float16).contiguous(),
            v, gg, beta, h0, cu, float(scale))


def эталон64(q, k, v, g, beta, h0, scale, NT, BT=64):
    """ВСЯ ЦЕПОЧКА В FP64, ПРЯМО ПО ОПРЕДЕЛЕНИЮ. Арбитр между путями, которые считают ОДНО И ТО ЖЕ
    разными последовательностями округлений: сравнение «наш против прежнего нашего» на такой вопрос
    не отвечает в принципе -- оба могут быть одинаково смещены. Здесь единственная сторона, которая
    ни одному из путей не родня: обращение (I+A) точное, накопление в fp64, порядок -- как в статье.
    """
    T, Hg, DK = q.shape[-3], q.shape[-2], q.shape[-1]
    H, DV = v.shape[-2], v.shape[-1]
    G = H // Hg
    qe = q.double()[0].repeat_interleave(G, dim=1).permute(1, 0, 2).contiguous()   # [H,T,DK]
    ke = k.double()[0].repeat_interleave(G, dim=1).permute(1, 0, 2).contiguous()
    ve = v.double()[0].permute(1, 0, 2).contiguous()                               # [H,T,DV]
    ge = g.double()[0].permute(1, 0).contiguous()                                  # [H,T]
    be = beta.double()[0].permute(1, 0).contiguous()
    h = (h0.double()[0].clone() if h0 is not None
         else torch.zeros(H, DV, DK, dtype=torch.float64, device=q.device))        # [H,DV,DK]
    o = torch.zeros(H, T, DV, dtype=torch.float64, device=q.device)
    I = torch.eye(BT, dtype=torch.float64, device=q.device)
    for it in range(NT):
        t0, t1 = it * BT, min((it + 1) * BT, T); L = t1 - t0
        qc, kc, vc = qe[:, t0:t1], ke[:, t0:t1], ve[:, t0:t1]
        gc, bc = ge[:, t0:t1], be[:, t0:t1]
        зат = torch.exp(gc.unsqueeze(2) - gc.unsqueeze(1))                         # [H,L,L]
        A = torch.tril(bc.unsqueeze(2) * зат * (kc @ kc.transpose(-1, -2)), -1)
        Ai = torch.linalg.solve_triangular(I[:L, :L] + A, I[:L, :L].expand(H, L, L),
                                           upper=False, unitriangular=True)
        w = Ai @ ((bc * torch.exp(gc)).unsqueeze(-1) * kc)
        u = Ai @ (bc.unsqueeze(-1) * vc)
        vn = u - w @ h.transpose(-1, -2)
        o[:, t0:t1] = scale * (torch.exp(gc).unsqueeze(-1) * (qc @ h.transpose(-1, -2))
                               + torch.tril(зат * (qc @ kc.transpose(-1, -2))) @ vn)
        gL = gc[:, L - 1:L]
        h = torch.exp(gL).unsqueeze(-1) * h + (vn * torch.exp(gL - gc).unsqueeze(-1)) \
            .transpose(-1, -2) @ kc
    return o.permute(1, 0, 2).unsqueeze(0).contiguous(), h.unsqueeze(0).contiguous()


def отн(a, b):
    a, b = a.float(), b.float()
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("дамп")
    ap.add_argument("--слить", action="store_true", help="прогнать слитое тело и сверить срезы")
    ap.add_argument("--арбитр", action="store_true",
                    help="сверить ВСЕ пути против fp64-эталона всей цепочки")
    ap.add_argument("--сшить", action="store_true",
                    help="ВТОРОЙ СРЕЗ: тело считает и w,u само (вход Ai+beta+сырое v)")
    ap.add_argument("--срез", default=None, help="какой срез печатать подробно")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    d = torch.load(a.дамп, map_location="cpu", weights_only=False)
    print(f"БОЕВОЙ ВИД: вызов {d['вызов']}, q={tuple(d['q'].shape)}/{d['q'].dtype}, "
          f"v={tuple(d['v'].shape)}, cu={None if d['cu_seqlens'] is None else d['cu_seqlens'].tolist()}")
    q, k, v, g, beta, h0, cu, scale = подготовить(d, dev)
    T, Hg, DK = q.shape[-3], q.shape[-2], q.shape[-1]
    H, DV = v.shape[-2], v.shape[-1]
    NT = (T + 63) // 64
    print(f"  T={T}  голов ключа {Hg}  голов значения {H}  DK={DK} DV={DV}  чанков {NT}  "
          f"хвост {T - (NT - 1) * 64} токенов  GQA={H // Hg}  scale={scale:.6f}")
    print(f"  начальное состояние: {'нет' if h0 is None else tuple(h0.shape)}")

    from fa2_sm70 import gdn as G
    ш = G.steps(q, k, v, g, beta, h0, cu, scale, out_bf16=(v.dtype == torch.bfloat16))
    print("\nСРЕЗЫ ОБРАЗЦА (то, с чем сверяется каждое вкладываемое умножение):")
    for имя in ("A", "Ai", "w", "u", "h", "v_new", "ht", "o"):
        t = ш.get(имя)
        if t is None:
            print(f"  {имя:<6} —"); continue
        f = t.float()
        print(f"  {имя:<6} {str(tuple(t.shape)):<26} {str(t.dtype):<16} "
              f"|.|={f.norm():.4e}  max|.|={f.abs().max():.4e}  "
              f"байт {t.numel()*t.element_size()/1e6:.1f} МБ")
    пром = sum(ш[i].numel() * ш[i].element_size() for i in ("A", "Ai", "w", "u", "h", "v_new"))
    print(f"\n  ПРОМЕЖУТОЧНЫЕ (исчезают при слиянии): {пром/1e6:.1f} МБ на слой при T={T}")
    print(f"  вход слоя: {sum(x.numel()*x.element_size() for x in (q,k,v,g,beta))/1e6:.1f} МБ, "
          f"выход o: {ш['o'].numel()*ш['o'].element_size()/1e6:.1f} МБ")
    # ---------------- СРЕЗ 1 СЛИЯНИЯ: первое слагаемое o = scale * e^{g_t} * (q h^T) ----------
    if a.слить:
        ext = G.gdn_ext()
        cu32, ci, co, NTk = G._chunk_meta(cu)
        w, u = ш["w"], ш["u"]
        # ВТОРОЙ СРЕЗ: вместо w,u подаются Ai, beta и СЫРОЕ v -- тело считает скобку само.
        доп = dict(Ai=ш["Ai"], beta=ш["b16"]) if a.сшить else {}
        л, п = (w, ш["v16"]) if a.сшить else (w, u)
        print(f"\nРЕЖИМ: {'СШИТОЕ тело (h+o+wu, вход Ai/beta/v)' if a.сшить else 'слитое тело (h+o, вход w/u)'}")
        o_сл, ht_сл = ext.gdn_ho(q, k, л, п, g, h0, cu32, co, NTk, scale, True, False, **доп)
        _, A_сл = ext.gdn_ho(q, k, л, п, g, h0, cu32, co, NTk, scale, True, True, **доп)
        # ЭТАЛОН A: e^{g_t-g_s} (q_t . k_s) при s<=t -- считается в торче, независимым путём
        T3, Hg3 = q.shape[-3], q.shape[-2]; H3 = v.shape[-2]
        Aэт = torch.zeros_like(A_сл, dtype=torch.float32)
        qf3, kf3, gf3 = q.float(), k.float(), g.float()
        for it in range(NT):
            t0, t1 = it * 64, min((it + 1) * 64, T3); L = t1 - t0
            for hv in range(H3):
                hk = hv // (H3 // Hg3)
                s = qf3[0, t0:t1, hk] @ kf3[0, t0:t1, hk].t()
                gt = gf3[0, t0:t1, hv]
                s = s * torch.exp(gt.unsqueeze(1) - gt.unsqueeze(0))
                s = torch.tril(s)
                Aэт[it, hv, :L, :L] = s
        rA = отн(A_сл.float(), Aэт)
        print(f"\nСРЕЗ 2 (A = e^{{g_t-g_s}} q k^T, маска): relL2 {rA:.3e}  "
              f"|наш|={A_сл.float().norm():.4e} |эталон|={Aэт.norm():.4e}  "
              f"-> {'СОВПАЛО' if rA < 3e-3 else 'РАСХОЖДЕНИЕ'}")
        # НЕЗАВИСИМЫЙ ЭТАЛОН ВЫХОДА, целиком в торче: первое слагаемое из среза h образца, второе --
        # из НАШЕЙ ЖЕ торчевой A и v_new образца. Прежде здесь стояло сравнение полного o с ОДНИМ
        # первым слагаемым, и строка вечно печатала "РАСХОЖДЕНИЕ" на верном ядре -- прибор врал в
        # сторону отказа. Слагаемые считаются по отдельности и печатаются, но вердикт -- по СУММЕ.
        hs = ш["h"].float()                      # [1, NT, H, DV, DK]
        vn = ш["v_new"].float()
        T2, Hg2 = q.shape[-3], q.shape[-2]
        H2, DV2 = v.shape[-2], v.shape[-1]
        эт1 = torch.zeros(1, T2, H2, DV2, device=q.device, dtype=torch.float32)
        эт2 = torch.zeros_like(эт1)
        qf, gf = q.float(), g.float()
        for it in range(NT):
            t0, t1 = it * 64, min((it + 1) * 64, T2); L = t1 - t0
            for hv in range(H2):
                hk = hv // (H2 // Hg2)
                # [len, DK] x [DV, DK]^T -> [len, DV]
                эт1[0, t0:t1, hv] = (qf[0, t0:t1, hk] @ hs[0, it, hv].t()) \
                                    * torch.exp(gf[0, t0:t1, hv]).unsqueeze(-1) * scale
                эт2[0, t0:t1, hv] = (Aэт[it, hv, :L, :L] @ vn[0, t0:t1, hv]) * scale
        эт = эт1 + эт2
        rп = отн(o_сл.float(), ш["o"].float())
        print(f"\nПОЛНЫЙ o против ОБРАЗЦА (пять ядер): relL2 {rп:.3e}  "
              f"|наш|={o_сл.float().norm():.4e} |образец|={ш['o'].float().norm():.4e}  "
              f"-> {'СОВПАЛО' if rп < 3e-3 else 'РАСХОЖДЕНИЕ'}")
        r = отн(o_сл.float(), эт)
        print(f"ПОЛНЫЙ o против НЕЗАВИСИМОГО эталона (торч): relL2 {r:.3e}   "
              f"|эталон|={эт.norm():.4e}  (слагаемые: |q h^T|={эт1.norm():.4e}, "
              f"|A v_new|={эт2.norm():.4e})  -> {'СОВПАЛО' if r < 3e-3 else 'РАСХОЖДЕНИЕ'}")
        if a.сшить:
            # СРЕЗ 4: v_new тело считает САМО (Ai x скобка). Проверяется косвенно -- через второе
            # слагаемое выхода, потому что наружу v_new больше не пишется: o - эт1 = A v_new_наш.
            наш2 = o_сл.float() - эт1
            r4 = отн(наш2, эт2)
            print(f"СРЕЗ 4 (v_new = Ai x скобка, виден через A v_new): relL2 {r4:.3e}  "
                  f"-> {'СОВПАЛО' if r4 < 5e-3 else 'РАСХОЖДЕНИЕ'}")
        if ht_сл is not None and ш.get("ht") is not None:
            rh = отн(ht_сл.float(), ш["ht"].float())
            # ПОРОГ ПО КЛАССУ ФОРМАТА, а не по «совпало с прежним». При входе w,u ht выходил БИТ
            # В БИТ -- но только потому, что вход был тождественным. Подстановка меняет порядок
            # округлений, и требовать нуля значит требовать, чтобы новый путь повторял ошибки
            # старого. Арбитр fp64 (--арбитр) показал: 1.267e-04 против 1.262e-04 у пятёрки ядер,
            # то есть разница между путями 0.4% от их общей ошибки к истине.
            print(f"СОСТОЯНИЕ ht (переносится в следующий кусок): relL2 {rh:.3e}  "
                  f"-> {'СОВПАЛО' if rh < 3e-4 else 'РАСХОЖДЕНИЕ'} (класс fp16 = 2.8e-04)")
    if a.арбитр:
        # ТРИ ПУТИ ПРОТИВ ОДНОГО FP64. Вопрос не «совпало ли с прежним», а «кто ближе к истине»:
        # подстановка v_new = Ai(V - diag(e^g)K h^T) меняет ПОРЯДОК округлений, и решить, стало ли
        # хуже, можно только стороной, которая ни одному из путей не родня.
        ext = G.gdn_ext()
        cu32, ci, co, NTk = G._chunk_meta(cu)
        o64, ht64 = эталон64(q, k, ш["v16"], g, ш["b16"], h0, scale, NT)
        o_сл, ht_сл = ext.gdn_ho(q, k, ш["w"], ш["u"], g, h0, cu32, co, NTk, scale, True, False)
        o_сш, ht_сш = ext.gdn_ho(q, k, ш["w"], ш["v16"], g, h0, cu32, co, NTk, scale, True, False,
                                 Ai=ш["Ai"], beta=ш["b16"])
        print("\n" + "=" * 78)
        print("АРБИТР FP64 (вся цепочка по определению, точное обращение (I+A))")
        print("=" * 78)
        print(f"{'путь':<34} {'o к fp64':>12} {'ht к fp64':>12}")
        for имя, oo, hh in (("пять ядер (образец)", ш["o"], ш["ht"]),
                            ("слитое h+o (вход w,u)", o_сл, ht_сл),
                            ("СШИТОЕ h+o+wu (вход Ai,beta,v)", o_сш, ht_сш)):
            print(f"{имя:<34} {отн(oo.float(), o64.float()):>12.3e} "
                  f"{отн(hh.float(), ht64.float()):>12.3e}")
        print("\nОдно округление fp16 = 2.8e-04 (верхняя оценка). Всё, что ниже, -- класс формата,")
        print("а не дефект; сравнивать пути между собой имеет смысл только по этим двум столбцам.")
    if a.срез:
        t = ш[a.срез]
        print(f"\nсрез {a.срез}: {tuple(t.shape)}\n{t.float().flatten()[:8]}")
    torch.save({k2: (None if val is None else val.cpu()) for k2, val in ш.items()},
               a.дамп.replace(".pt", "_срезы.pt"))
    print(f"\nсрезы образца сохранены -> {a.дамп.replace('.pt', '_срезы.pt')}")


if __name__ == "__main__":
    main()
