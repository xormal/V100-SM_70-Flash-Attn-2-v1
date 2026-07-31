# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Regenerate the README figures from MEASURED numbers. Every value here was produced by the script
# named beside it, on a V100-SXM2-32GB with clocks locked; nothing is modelled or interpolated.
# Re-run after a change that moves any of them, so the pictures never drift from the tables.
#   python tools/make_plots.py
#
# TWO LANGUAGES, ONE SET OF NUMBERS. README.md is English and README.ru.md is Russian, so every
# figure is emitted twice -- docs/img/en/ and docs/img/ru/ -- from the same measured arrays. The
# labels live in LANG below and nowhere else: a figure carrying Russian axis labels inside the
# English README is the same defect as Russian prose in it, and unlike prose it cannot be fixed by
# proofreading the text.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "docs", "img")
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

LANG = {
    "en": dict(
        ctx="context, tokens", us_step="us per decode step", rel_uu="time / time at UU=2",
        speed_sdpa="speedup over SDPA-eff", bh_causal="B,H,Sk (causal)", hkv_len="H/Hkv, KV length",
        v_before="V in a second wave (before)", v_after="V issued with K (now)",
        t_decode="Decode on int8 KV: hoisting the V load to K\nV100, B=1, H=32/Hkv=8, d=128",
        uu_default=" (default)",
        t_uu="FA2SM70_DEC_UU switch: unroll of the KV loop\nbelow 1.0 = faster; the optimum is a PEAK, not a slope",
        split_old="old rule (2048 CTAs)", split_new="new rule (512 CTAs)",
        t_split="Choosing the split count: the target is a CTA count, not a chunk size\nint8 KV, V100",
        nosplit="without split-K", withsplit="with split-K",
        t_sched="Prefill: scheduling under a causal mask\nleft - machine full, right - under-filled",
        bhs="B,H,S (causal, head_dim=256)", speedup_x="speedup",
        vs_cutlass="vs cutlass FMHA (this repo's previous path)", vs_sdpa="vs torch SDPA-eff",
        t_d256="Hand-written Volta mainloop at head_dim=256\nthe kernel the deployed 27B model runs on",
        ph_names=["1st GEMM  Q·Kᵀ", "2nd GEMM  P·V", "softmax", "hauling", "rendezvous", "the rest"],
        share="share of kernel time, %",
        t_phase="What the byte-KV kernel spends its time on\n(one phase removed per build; only the time is read)",
        ph_note="same tensor work,\nsame aux issues,\n1.78× the time",
        extra_t="extra time, %", slot_names=["count model\n(1 slot = 1 cycle)", "idle\ninstruction",
                                             "same count,\non the chain"],
        roles="+128 issues per computing warp",
        t_slots="What an extra issue actually costs\nthe number under each bar = cycles per slot",
    ),
    "ru": dict(
        ctx="контекст, токенов", us_step="мкс на шаг декода", rel_uu="время / время при UU=2",
        speed_sdpa="во сколько раз быстрее SDPA-eff", bh_causal="B,H,Sk (причинная)",
        hkv_len="H/Hkv, длина KV",
        v_before="V отдельной волной (было)", v_after="V вместе с K (сейчас)",
        t_decode="Декод на int8 KV: подъём загрузки V к K\nV100, B=1, H=32/Hkv=8, d=128",
        uu_default=" (умолчание)",
        t_uu="Переключатель FA2SM70_DEC_UU: разворот цикла по KV\nниже 1.0 = быстрее; оптимум — пик, а не склон",
        split_old="старое правило (2048 CTA)", split_new="новое правило (512 CTA)",
        t_split="Выбор числа срезов: цель — количество CTA, а не размер куска\nint8 KV, V100",
        nosplit="без split-K", withsplit="со split-K",
        t_sched="Префилл: планирование при причинной маске\nслева — машина полна, справа — недобрана",
        bhs="B,H,S (причинная, head_dim=256)", speedup_x="во сколько раз быстрее",
        vs_cutlass="против cutlass FMHA (прежний путь этого репозитория)", vs_sdpa="против torch SDPA-eff",
        t_d256="Рукописный мейнлуп Volta при head_dim=256\nто ядро, на котором работает развёрнутая 27B",
        ph_names=["1-е умножение  Q·Kᵀ", "2-е умножение  P·V", "софтмакс", "подача", "рандеву", "остальное"],
        share="доля времени ядра, %",
        t_phase="На что байтовое ядро тратит время\n(в каждой сборке снята одна фаза; читается только время)",
        ph_note="та же тензорная работа,\nто же число выдач,\nв 1.78 раза дольше",
        extra_t="прирост времени, %", slot_names=["модель счёта\n(слот = такт)", "холостая\nкоманда",
                                                  "тот же счёт,\nно в цепи"],
        roles="+128 выдач на считающий варп",
        t_slots="Сколько на самом деле стоит выдача\nчисло под столбиком = тактов на слот",
    ),
}

# ================================ MEASURED DATA (one copy, both languages) =======================
# 1. decode vs context, int8 paged KV -- the FA2SM70_DEC_* sweep; B=1, H=32, Hkv=8, d=128, e4m3 pool
CTX = [32768, 65536, 131072, 262144]
DEC_BEFORE = [296.7, 564.1, 1151.0, 2370.3]      # V issued in a second wave (the old shape)
DEC_AFTER = [256.8, 492.5, 954.7, 1848.3]        # V issued with K

# 2. the switchable unroll -- benchmarks/bench_decode_lengths.py over FA2SM70_DEC_UU
UU = {2: [22.9, 55.2, 91.9, 150.9, 257.6, 493.2, 947.3, 1845.7],
      3: [23.2, 57.2, 99.7, 146.6, 244.4, 451.2, 876.9, 1683.4],
      4: [22.4, 53.7, 92.8, 142.9, 239.8, 457.9, 887.3, 1732.8],
      6: [25.1, 68.1, 103.4, 167.2, 285.0, 531.3, 1036.6, 2049.2],
      8: [23.8, 70.3, 103.0, 163.7, 282.9, 517.6, 1023.7, 1994.1]}
UU_X = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

# 3. the split-count rule -- benchmarks/bench_decode.py, int8 KV
SPLIT_SHAPES = ["32/32\n2048", "32/32\n8192", "32/8\n8192", "32/32\n32768", "8/8\n16384", "32/32\n4096 d64"]
SPLIT_OLD = [65.2, 134.6, 103.7, 409.3, 71.5, 58.6]
SPLIT_NEW = [39.0, 117.7, 79.5, 398.8, 71.6, 47.5]

# 4. prefill causal scheduling -- benchmarks/bench_prefill.py, clocks locked 1530 MHz (2026-07-31)
#    (label, ours ms, no-split ms, SDPA-eff ms)
SCHED = [("1,32,2048", 0.947, 0.944, 1.031), ("1,32,8192", 13.697, 13.614, 15.071),
         ("1,1,2048", 0.061, 0.127, 0.121), ("1,1,4096", 0.140, 0.262, 0.269),
         ("1,2,2048", 0.092, 0.146, 0.151), ("1,1,2048 d64", 0.047, 0.156, 0.148)]

# 5. head_dim=256 forward -- benchmarks/bench_prefill_d256.py, clocks locked 1530 MHz (2026-07-31)
#    (label, ours ms, cutlass ms, SDPA-eff ms)
D256 = [("1,32,2048", 1.815, 2.798, 2.813), ("1,16,4096", 3.357, 6.194, 6.230),
        ("1,24,4096", 4.944, 9.156, 8.730), ("2,8,4096", 3.346, 6.384, 6.315),
        ("1,12,8192", 9.329, 14.024, 18.435), ("1,4,8192", 3.244, 4.830, 6.612)]

# 6. WHAT ACTUALLY BINDS the byte-KV role kernel -- benchmarks/bench_slots_ab.py + the phase
#    falsifiers (a build with one phase removed: the answer is wrong, only the time is read).
#    d=256, H=16/Hkv=2, Sq=512, causal, clocks locked 1530 MHz. See docs/VOLTA_SM70.md section 3b.
#    Left panel: share of time per phase. Right panel: what an extra issue actually costs.
PHASES = [("gemm1", 35.1), ("gemm2", 19.7), ("softmax", 15.2), ("feed", 5.7), ("rdv", 2.0),
          ("rest", 22.3)]
#    THE PRICE OF A SLOT IS NOT ONE NUMBER -- IT DEPENDS ON WHERE THE INSTRUCTION SITS.
#    Both rows below add the SAME +128 issues per computing warp and were reproduced independently:
#      * idle    -- a dead LOP3 pair (zero mask from constant memory), nothing waits on it: +1.93%
#                   (re-measured 2026-07-31 across Sk 8K/32K/128K: +2.10 / +1.92 / +1.89%)
#      * on-chain-- the real sign XOR of the byte expansion, on the dependency chain feeding the
#                   tensor op (formats 0 vs 6, differing by exactly that instruction): +4.24%
#                   (re-measured: +4.09 / +4.33 / +4.45%)
#    A count-based model ("1 slot = 1 cycle") predicts +8.7% for both. Ratio on-chain/idle = 2.2x.
#    (label, predicted extra time %, measured extra time %)
SLOT_PRICE = [("predicted", 8.7, 8.7), ("idle", 8.7, 1.93), ("on chain", 8.7, 4.24)]


def emit(lang):
    T = LANG[lang]
    out = os.path.join(IMG, lang)
    os.makedirs(out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(CTX, DEC_BEFORE, "o--", color="#999", label=T["v_before"])
    ax.plot(CTX, DEC_AFTER, "o-", color="#c0392b", label=T["v_after"])
    for x, a, b in zip(CTX, DEC_AFTER, DEC_BEFORE):
        ax.annotate(f"+{(b/a-1)*100:.0f}%", (x, a), textcoords="offset points", xytext=(0, -14),
                    ha="center", color="#c0392b", fontsize=8)
    ax.set_xscale("log", base=2); ax.set_xticks(CTX); ax.set_xticklabels(["32K", "64K", "128K", "256K"])
    ax.set_xlabel(T["ctx"]); ax.set_ylabel(T["us_step"]); ax.set_title(T["t_decode"])
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "decode_context_int8.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for k, v in UU.items():
        rel = [a / b for a, b in zip(v, UU[2])]
        ax.plot(UU_X, rel, "o-" if k == 4 else "o--", lw=2 if k == 4 else 1,
                label=f"UU={k}" + (T["uu_default"] if k == 4 else ""))
    ax.axhline(1.0, color="#999", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks(UU_X)
    ax.set_xticklabels(["1K", "4K", "8K", "16K", "32K", "64K", "128K", "256K"])
    ax.set_xlabel(T["ctx"]); ax.set_ylabel(T["rel_uu"]); ax.set_title(T["t_uu"])
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "switch_unroll.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    i = range(len(SPLIT_SHAPES))
    ax.bar([x - 0.2 for x in i], SPLIT_OLD, 0.4, label=T["split_old"], color="#999")
    ax.bar([x + 0.2 for x in i], SPLIT_NEW, 0.4, label=T["split_new"], color="#c0392b")
    for x, a, b in zip(i, SPLIT_OLD, SPLIT_NEW):
        if a > b * 1.05:
            ax.annotate(f"{a/b:.2f}x", (x + 0.2, b), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color="#c0392b")
    ax.set_xticks(list(i)); ax.set_xticklabels(SPLIT_SHAPES, fontsize=8)
    ax.set_ylabel(T["us_step"]); ax.set_xlabel(T["hkv_len"]); ax.set_title(T["t_split"])
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "split_rule.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    i = range(len(SCHED))
    ax.bar([x - 0.2 for x in i], [r[3] / r[2] for r in SCHED], 0.4, label=T["nosplit"], color="#999")
    ax.bar([x + 0.2 for x in i], [r[3] / r[1] for r in SCHED], 0.4, label=T["withsplit"], color="#c0392b")
    ax.axhline(1.0, color="#333", lw=0.8)
    ax.set_xticks(list(i)); ax.set_xticklabels([r[0] for r in SCHED], fontsize=7.5, rotation=12)
    ax.set_ylabel(T["speed_sdpa"]); ax.set_xlabel(T["bh_causal"]); ax.set_title(T["t_sched"])
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "prefill_scheduling.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    i = range(len(D256))
    ax.bar([x - 0.2 for x in i], [r[2] / r[1] for r in D256], 0.4, label=T["vs_cutlass"], color="#999")
    ax.bar([x + 0.2 for x in i], [r[3] / r[1] for r in D256], 0.4, label=T["vs_sdpa"], color="#c0392b")
    ax.axhline(1.0, color="#333", lw=0.8)
    for x, r in zip(i, D256):
        ax.annotate(f"{r[3]/r[1]:.2f}x", (x + 0.2, r[3] / r[1]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8, color="#c0392b")
    ax.set_xticks(list(i)); ax.set_xticklabels([r[0] for r in D256], fontsize=7.5, rotation=12)
    ax.set_ylabel(T["speedup_x"]); ax.set_xlabel(T["bhs"]); ax.set_title(T["t_d256"])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "d256_mainloop.png")); plt.close(fig)

    # ---- what actually binds: phase decomposition + the true price of an issue slot ----------
    # Two panels because the finding IS the pair: the left one names the resource, the right one
    # shows why counting instructions did not predict it. Either alone reads as a different claim.
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1.55, 1]})
    names, vals = T["ph_names"], [p[1] for p in PHASES]
    cols = ["#c0392b", "#e08a7d", "#7f8c8d", "#95a5a6", "#bdc3c7", "#dfe4e6"]
    y = list(range(len(names)))[::-1]
    axl.barh(y, vals, color=cols, height=0.62)
    for yy, v in zip(y, vals):
        axl.annotate(f"{v:.1f}", (v, yy), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8.5)
    axl.set_yticks(y); axl.set_yticklabels(names, fontsize=8.5)
    axl.set_xlabel(T["share"]); axl.set_title(T["t_phase"], fontsize=9.5)
    axl.set_xlim(0, max(vals) * 1.30)
    axl.annotate("", xy=(vals[0] * 1.06, y[0]), xytext=(vals[0] * 1.06, y[1]),
                 arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.1))
    axl.annotate(T["ph_note"], (vals[0] * 1.09, (y[0] + y[1]) / 2), va="center",
                 fontsize=7.5, color="#c0392b")

    i = range(len(SLOT_PRICE))
    cols2 = ["#bdc3c7", "#7f8c8d", "#c0392b"]
    axr.bar(list(i), [p[2] for p in SLOT_PRICE], 0.55, color=cols2)
    for x, p in zip(i, SLOT_PRICE):
        axr.annotate(f"{p[2]/p[1]:.2f}", (x, p[2]), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9, color="#333")
    axr.set_xticks(list(i)); axr.set_xticklabels(T["slot_names"], fontsize=8.5)
    axr.set_ylabel(T["extra_t"]); axr.set_xlabel(T["roles"], fontsize=8.5)
    axr.set_title(T["t_slots"], fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "binding_resource.png")); plt.close(fig)

    return sorted(os.listdir(out))


for lang in ("en", "ru"):
    print(f"docs/img/{lang}/:", ", ".join(emit(lang)))
