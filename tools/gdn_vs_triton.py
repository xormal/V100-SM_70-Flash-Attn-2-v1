# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# СВЕРКА НАШЕГО СКАНА С ОБРАЗЦОМ (fla/Triton) ПОШАГОВО И ЦЕЛИКОМ, плюс замер чередованием.
#
# Почему пошагово, а не только по выходу: цепочка из пяти ядер, и расхождение в конце не говорит,
# ГДЕ оно возникло. Каждый шаг сверяется со СВОИМ трионовским образцом на ОДНОМ И ТОМ ЖЕ входе,
# поэтому первый разошедшийся шаг виден сразу.
#
# Почему сверка с Triton, а не с fp64-эталоном: fp64-эталон уже есть в стендах на C++
# (tools/volta_gdn_*_test.cu) и отвечает на вопрос "верна ли формула". Здесь вопрос ДРУГОЙ --
# "совпадает ли с тем, что считает боевой путь сейчас", то есть годится ли подмена. Оба нужны.
#
# Запуск:
#   PYTHONPATH=<repo> python tools/gdn_vs_triton.py [T] [dtype]
import os
import sys
import time

import torch

vllm_site = os.environ.get("VLLM_SITE_PACKAGES")
if not vllm_site:
    raise SystemExit("set VLLM_SITE_PACKAGES to the vLLM site-packages directory")
sys.path.insert(0, vllm_site)

from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h,
)
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (  # noqa: E402
    chunk_scaled_dot_kkt_fwd,
)
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum  # noqa: E402
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd  # noqa: E402
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril  # noqa: E402
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd  # noqa: E402

from fa2_sm70 import gdn as G  # noqa: E402


def relerr(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    return (d.max() / (b.abs().max() + 1e-30)).item(), (
        d.pow(2).sum().sqrt() / (b.pow(2).sum().sqrt() + 1e-30)
    ).item()


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 15680
    dt = torch.float16 if (len(sys.argv) > 2 and sys.argv[2] == "fp16") else torch.bfloat16
    dev = "cuda"
    H, Hg, D = 24, 8, 128           # боевая форма Qwen3.5-27B при TP=2
    torch.manual_seed(7)

    q = torch.randn(1, T, Hg, D, device=dev, dtype=dt) * 0.5
    k = torch.randn(1, T, Hg, D, device=dev, dtype=dt) * 0.5
    v = torch.randn(1, T, H, D, device=dev, dtype=dt) * 0.5
    beta = torch.rand(1, T, H, device=dev, dtype=dt).sigmoid()
    graw = torch.nn.functional.logsigmoid(torch.rand(1, T, H, device=dev, dtype=torch.float32))
    h0 = torch.randn(1, H, D, D, device=dev, dtype=torch.float32) * 0.05
    cu = torch.tensor([0, T], device=dev, dtype=torch.long)
    scale = D ** -0.5

    print(f"=== T={T} H={H} Hg={Hg} D={D} dtype={dt} NT={(T + 63) // 64} ===")

    # ---- общая голова цепочки: l2norm + cumsum. ОСТАЁТСЯ на Triton у обеих сторон.
    qn, kn = l2norm_fwd(q), l2norm_fwd(k)
    g = chunk_local_cumsum(graw, chunk_size=64, cu_seqlens=cu)

    # ---- образец: шаги Triton по отдельности
    A_t = chunk_scaled_dot_kkt_fwd(k=kn, beta=beta, g=g, cu_seqlens=cu, output_dtype=torch.float32)
    Ai_t = solve_tril(A=A_t, cu_seqlens=cu, output_dtype=kn.dtype)
    w_t, u_t = recompute_w_u_fwd(k=kn, v=v, beta=beta, A=Ai_t, g_cumsum=g, cu_seqlens=cu)
    h_t, vn_t, ht_t = chunk_gated_delta_rule_fwd_h(
        k=kn, w=w_t, u=u_t, g=g, initial_state=h0, output_final_state=True, cu_seqlens=cu
    )
    o_t = chunk_fwd_o(q=qn, k=kn, v=vn_t, h=h_t, g=g, scale=scale, cu_seqlens=cu)

    # ---- наше: те же шаги. q,k подаются в fp16 (в бою их так пишет l2norm).
    q16, k16 = qn.to(torch.float16).contiguous(), kn.to(torch.float16).contiguous()
    st = G.steps(q16, k16, v, g, beta, h0, cu, scale, out_bf16=(dt == torch.bfloat16))

    print("шаг              max/max      relL2      (образец -- Triton на том же входе)")
    for name, ours, ref in (
        ("A (kkt)", st["A"], A_t),
        ("Ai (tril)", st["Ai"], Ai_t),
        ("w", st["w"], w_t),
        ("u", st["u"], u_t),
        ("h", st["h"], h_t),
        ("v_new", st["v_new"], vn_t),
        ("ht", st["ht"], ht_t),
        ("o", st["o"], o_t),
    ):
        m, l2 = relerr(ours, ref)
        print(f"  {name:<14} {m:9.3e}  {l2:9.3e}")

    # ---- цельный путь против цельного образца (то, что реально подменяется)
    o_ref, ht_ref = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=graw, beta=beta, scale=scale, initial_state=h0,
        output_final_state=True, cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True,
    )
    o_our, ht_our = G.chunk_gated_delta_rule_volta(
        q16, k16, v, g, beta, scale, h0, True, cu, dt
    )
    m, l2 = relerr(o_our, o_ref); print(f"ЦЕЛЬНЫЙ o   max/max {m:.3e}  relL2 {l2:.3e}")
    m2, l22 = relerr(ht_our, ht_ref); print(f"ЦЕЛЬНЫЙ ht  max/max {m2:.3e}  relL2 {l22:.3e}")
    print(f"NaN у нас: o {torch.isnan(o_our).sum().item()}  ht {torch.isnan(ht_our).sum().item()}")

    # ---- НЕЗАВИСИМЫЙ ОРАКУЛ. Сверка с чанковым Triton отвечает "совпадает ли", но НЕ отвечает
    # "чья ошибка больше": обе стороны считают одну и ту же чанковую схему, и общий сдвиг
    # относительно истины у них одинаков. Оракулом берётся fused_recurrent_gated_delta_rule --
    # ДРУГОЙ алгоритм (последовательная рекуррентность, без чанков и без WY-представления),
    # то есть независимая реализация той же математики. Если наша ошибка к нему НЕ ХУЖЕ
    # трионовской, замена точности не теряет -- а это и есть вопрос, ради которого сверка нужна.
    try:
        from vllm.model_executor.layers.fla.ops.fused_recurrent import (
            fused_recurrent_gated_delta_rule,
        )
        Tr = min(T, 2048)                       # рекуррентный оракул последователен и медленен
        sl = slice(0, Tr)
        cur = torch.tensor([0, Tr], device=dev, dtype=torch.long)
        idx = torch.zeros(1, dtype=torch.int32, device=dev)
        o_rec, _ = fused_recurrent_gated_delta_rule(
            q=q[:, sl].float(), k=k[:, sl].float(), v=v[:, sl].float(),
            g=graw[:, sl].float(), beta=beta[:, sl].float(),
            initial_state=h0.clone(), inplace_final_state=False, cu_seqlens=cur,
            ssm_state_indices=idx, use_qk_l2norm_in_kernel=True, scale=scale)
        o_ref_s = chunk_gated_delta_rule(
            q=q[:, sl], k=k[:, sl], v=v[:, sl], g=graw[:, sl], beta=beta[:, sl], scale=scale,
            initial_state=h0.clone(), output_final_state=True, cu_seqlens=cur,
            head_first=False, use_qk_l2norm_in_kernel=True)[0]
        qn2, kn2 = l2norm_fwd(q[:, sl]), l2norm_fwd(k[:, sl])
        g2 = chunk_local_cumsum(graw[:, sl], chunk_size=64, cu_seqlens=cur)
        o_our_s = G.chunk_gated_delta_rule_volta(
            qn2.to(torch.float16).contiguous(), kn2.to(torch.float16).contiguous(),
            v[:, sl].contiguous(), g2, beta[:, sl].contiguous(), scale, h0.clone(), True, cur, dt)[0]
        _, l2t = relerr(o_ref_s, o_rec)
        _, l2o = relerr(o_our_s, o_rec)
        verdict = "НЕ ХУЖЕ" if l2o <= l2t * 1.15 else "ХУЖЕ"
        print(f"ОРАКУЛ (fused_recurrent, fp32, T={Tr}): Triton relL2 {l2t:.3e}   "
              f"наше relL2 {l2o:.3e}   -> {verdict}")
    except Exception as e:  # noqa: BLE001
        print(f"ОРАКУЛ недоступен: {type(e).__name__}: {e}")

    # ---- ЗАМЕР ЧЕРЕДОВАНИЕМ (control/candidate в одном процессе, медиана).
    # Подряд мерить нельзя: V100 без persistence сбрасывает частоту, и вариант, замеренный вторым,
    # систематически проигрывает -- на этом же железе так родилась "двукратная" разница из ничего.
    def tri():
        chunk_gated_delta_rule(
            q=q, k=k, v=v, g=graw, beta=beta, scale=scale, initial_state=h0,
            output_final_state=True, cu_seqlens=cu, head_first=False,
            use_qk_l2norm_in_kernel=True)

    def our():
        qq, kk = l2norm_fwd(q), l2norm_fwd(k)
        gg = chunk_local_cumsum(graw, chunk_size=64, cu_seqlens=cu)
        G.chunk_gated_delta_rule_volta(
            qq.to(torch.float16), kk.to(torch.float16), v, gg, beta, scale, h0, True, cu, dt)

    for _ in range(5):
        tri(); our()
    torch.cuda.synchronize()
    ta, tb = [], []
    for _ in range(15):
        torch.cuda.synchronize(); t0 = time.perf_counter(); tri()
        torch.cuda.synchronize(); ta.append(time.perf_counter() - t0)
        torch.cuda.synchronize(); t0 = time.perf_counter(); our()
        torch.cuda.synchronize(); tb.append(time.perf_counter() - t0)
    ta.sort(); tb.sort()
    ma, mb = ta[len(ta) // 2] * 1e3, tb[len(tb) // 2] * 1e3
    print(f"ЗАМЕР (медиана из 15, чередование): Triton {ma:.3f} мс   наше {mb:.3f} мс   x{ma / mb:.2f}")
    print(f"  на 48 слоёв: Triton {ma * 48:.1f} мс   наше {mb * 48:.1f} мс   разница {(ma - mb) * 48:.1f} мс")

    # ---- ПОШАГОВЫЙ ЗАМЕР. Нужен, чтобы отвечать на вопрос "где именно выигрыш" и чтобы
    # СВЕРЯТЬ трионовскую сторону с ранее опубликованной раскладкой: если суммарный baseline
    # разошёлся, разойтись он мог только на конкретном шаге, и это надо видеть, а не спорить.
    import fa2_sm70
    ext = fa2_sm70._ext.gdn_ext()
    cu32, ci, co, NTv = G._chunk_meta(cu)
    Hv = v.shape[-2]
    v16 = v.to(torch.float16).contiguous()
    b16 = beta.to(torch.float16).contiguous()
    # [131] наш solve_tril читает Araw в fp16 -- тритоновский A_t (fp32) конвертируется на входе
    A16 = A_t.to(torch.float16).contiguous()
    Ai_o = ext.gdn_solve_tril(A16, cu32, ci)
    w_o, u_o = ext.gdn_wu(k16, v16, b16, g, Ai_o, cu32, ci)
    h_o, vn_o, _ = ext.gdn_h(k16, w_o, u_o, g, h0, cu32, co, NTv, True)
    steps_pairs = [
        ("l2norm+cumsum", lambda: (l2norm_fwd(q), l2norm_fwd(k),
                                   chunk_local_cumsum(graw, chunk_size=64, cu_seqlens=cu)),
         None),
        ("kkt", lambda: chunk_scaled_dot_kkt_fwd(k=kn, beta=beta, g=g, cu_seqlens=cu,
                                                 output_dtype=torch.float32),
         lambda: ext.gdn_kkt(k16, b16, g, cu32, ci, Hv)),
        ("solve_tril", lambda: solve_tril(A=A_t, cu_seqlens=cu, output_dtype=kn.dtype),
         lambda: ext.gdn_solve_tril(A16, cu32, ci)),
        ("w,u", lambda: recompute_w_u_fwd(k=kn, v=v, beta=beta, A=Ai_t, g_cumsum=g, cu_seqlens=cu),
         lambda: ext.gdn_wu(k16, v16, b16, g, Ai_o, cu32, ci)),
        ("h", lambda: chunk_gated_delta_rule_fwd_h(k=kn, w=w_t, u=u_t, g=g, initial_state=h0,
                                                   output_final_state=True, cu_seqlens=cu),
         lambda: ext.gdn_h(k16, w_o, u_o, g, h0, cu32, co, NTv, True)),
        ("o", lambda: chunk_fwd_o(q=qn, k=kn, v=vn_t, h=h_t, g=g, scale=scale, cu_seqlens=cu),
         lambda: ext.gdn_o(q16, k16, vn_o, h_o, g, cu32, ci, scale, dt == torch.bfloat16)),
    ]

    def med(fn, n=9):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    print("шаг            Triton мс   наше мс    x")
    sa = sb = 0.0
    for name, ftri, four in steps_pairs:
        t1 = med(ftri)
        t2 = med(four) if four else t1        # первый шаг общий: он НЕ заменяется
        sa += t1; sb += t2
        print(f"  {name:<14}{t1:8.3f}  {t2:8.3f}  {(t1 / t2 if t2 else 0):6.2f}")
    print(f"  {'СУММА':<14}{sa:8.3f}  {sb:8.3f}  {sa / sb:6.2f}"
          f"   (цельный путь выше: {ma:.3f} / {mb:.3f})")


if __name__ == "__main__":
    main()
