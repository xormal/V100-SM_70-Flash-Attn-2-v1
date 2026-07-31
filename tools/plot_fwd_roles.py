# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# График "ядро ролей против прочих путей" в двух языках. Подписи НЕ смешиваются: русские только в
# docs/img/ru/, английские только в docs/img/en/ -- иначе они пролезают в чужую версию README.
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NOTE_RU = (
    "ЦЕЛЬ: БИТ-ТОЧНЫЙ префилл на 250K контекста, и целевая конфигурация -- KV В INT8 (модель развёрнута\n"
    "на int8-весах и int8-KV, иначе 250K на 2xV100 не помещаются). Поэтому «точно» здесь означает точно\n"
    "ОТНОСИТЕЛЬНО ТЕХ ЗНАЧЕНИЙ, С КОТОРЫМИ МОДЕЛЬ РАБОТАЕТ, то есть относительно деквантованных KV:\n"
    "4.6e-4...5.4e-4 -- ТОТ ЖЕ ПОРЯДОК, что у fp16 против fp64 (2.7e-4...3.0e-4). Число 3.9e-3 -- это\n"
    "отличие от НЕквантованной модели, то есть цена решения квантовать KV ради 250K, а НЕ ошибка ядра.\n"
    "ПРИ ЭТОМ SM70 ПОД int8 НЕ ПРОЕКТИРОВАЛСЯ ВОВСЕ: тензорные ядра Volta умеют только fp16, инструкции\n"
    "IMMA у них НЕТ -- она появилась лишь в Turing. Байты попадают на тензорные ядра тем, что кодируются\n"
    "ТОЧНО в мантиссу fp16: целые до 2048 представимы в ней без потерь, накопление идёт в fp32."
)
NOTE_EN = (
    "GOAL: BIT-EXACT prefill at a 250K context, and the target configuration has KV IN INT8 (the model is\n"
    "deployed on int8 weights and int8 KV -- 250K does not fit on 2xV100 otherwise). So \"exact\" here means\n"
    "exact WITH RESPECT TO THE VALUES THE MODEL ACTUALLY USES, i.e. against dequantised KV:\n"
    "4.6e-4..5.4e-4 -- the SAME ORDER as fp16 against fp64 (2.7e-4..3.0e-4). The 3.9e-3 figure is the\n"
    "difference from an UNQUANTISED model: the price of quantising KV for 250K, NOT kernel error.\n"
    "AND SM70 WAS NEVER DESIGNED FOR int8 AT ALL: Volta's tensor cores do fp16 only and have NO IMMA\n"
    "instruction -- that arrived with Turing. Bytes reach the tensor cores by being encoded EXACTLY into\n"
    "an fp16 mantissa: integers up to 2048 are representable without loss, accumulating in fp32."
)

L = {
    "ru": dict(
        title="Прямой проход, head_dim=256, причинная маска (V100, частоты зафиксированы)",
        ylab="TFLOP/s (полезные)", xlab="форма (B, H, Sq)", note=NOTE_RU,
        keys=[("sdpa", "torch SDPA-eff"), ("cutlass", "cutlass (наш прежний путь)"),
              ("i8_old", "наш int8, 2 блока"), ("fp16", "наше fp16-ядро"),
              ("i8_roles", "наш int8 + роли варпов")]),
    "en": dict(
        title="Forward, head_dim=256, causal (V100, clocks locked)",
        ylab="useful TFLOP/s", xlab="shape (B, H, Sq)", note=NOTE_EN,
        keys=[("sdpa", "torch SDPA-eff"), ("cutlass", "cutlass (our previous path)"),
              ("i8_old", "our int8, 2 blocks"), ("fp16", "our fp16 kernel"),
              ("i8_roles", "our int8 + warp roles")]),
}
COL = ["#9aa0a6", "#7e57c2", "#ef6c00", "#1e88e5", "#2e7d32"]

if len(sys.argv) != 2:
    raise SystemExit("usage: python tools/plot_fwd_roles.py /external/path/bench_fwd_roles.json")
with open(sys.argv[1], encoding="utf-8") as f:
    D = json.load(f)
for lang, t in L.items():
    labels = [f"{r['shape'][0]},{r['shape'][1]},{r['shape'][2]}" for r in D]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    n = len(t["keys"])
    w = 0.8 / n
    for i, (k, nm) in enumerate(t["keys"]):
        vals = []
        for r in D:
            B, H, Sq, Sk = r["shape"]
            fl = 4.0 * B * H * Sq * Sk * 256 / 2.0      # причинная маска: половина; ДВА умножения
            vals.append(fl / (r[k] * 1e-3) / 1e12 if r.get(k) else np.nan)
        x = np.arange(len(D)) + (i - (n - 1) / 2) * w
        ax.bar(x, vals, w, label=nm, color=COL[i])
        for xx, vv in zip(x, vals):
            if vv == vv:
                ax.text(xx, vv, f"{vv:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(D)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(t["xlab"])
    ax.set_ylabel(t["ylab"])
    ax.set_title(t["title"], fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=.3)
    fig.text(0.01, -0.02, t["note"], fontsize=7.5, va="top")
    os.makedirs(f"docs/img/{lang}", exist_ok=True)
    fig.savefig(f"docs/img/{lang}/fwd_roles_d256.png", dpi=150, bbox_inches="tight")
    print(f"docs/img/{lang}/fwd_roles_d256.png")
