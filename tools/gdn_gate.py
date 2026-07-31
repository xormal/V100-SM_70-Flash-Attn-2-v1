# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ================================================================================================
# ПОВЕДЕНЧЕСКИЙ ГЕЙТ ВРЕЗКИ СКАНА GDN. Проверяет ДВА условия, и оба обязательны:
#   (1) при ВЫКЛЮЧЕННОЙ подмене результат СОВПАДАЕТ с образцом (побитово: путь тот же самый);
#   (2) при ВКЛЮЧЁННОЙ подмене результат ОТЛИЧАЕТСЯ от образца (но в пределах fp16) И
#       счётчик наших вызовов вырос.
#
# ПОЧЕМУ ИМЕННО ТАК. Правка, которая НЕ РАБОТАЕТ, проходит проверку "совпало при выключении"
# идеально: она ведь ничего не делает. Ровно этим в проекте уже обожглись -- пять ядер скана
# считались "сделанными, x6.5", а в боевом пути их не стояло ВООБЩЕ. Разница при включении --
# единственное, что отличает работающую подмену от неработающей.
#
# ПОЧЕМУ НЕ ЧЕРЕЗ /proc/<pid>/environ. У spawn-воркеров vLLM окружение в /proc не содержит даже
# CUDA_VISIBLE_DEVICES (7313 записей spawn-блоба). Проверять можно только поведением.
#
# Запуск:  PYTHONPATH=<repo>/integrations/vllm_sm70:<repo> python tools/gdn_gate.py
# ================================================================================================
import os
import subprocess
import sys

_CHILD = r"""
import os, sys, torch, json
site = os.environ["VLLM_SITE_PACKAGES"]
sys.path.insert(0, site)
import fa2_gdn_shim
fa2_gdn_shim.install()
# ИМПОРТ ПОТРЕБИТЕЛЯ ПОСЛЕ установки хука -- ровно как в бою.
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule

T, H, Hg, D = 2048, 24, 8, 128
torch.manual_seed(11)
dev = "cuda"
dt = torch.float16 if os.environ.get("GATE_DT", "bf16") == "fp16" else torch.bfloat16
q = (torch.randn(1, T, Hg, D, device=dev, dtype=dt) * 0.5)
k = (torch.randn(1, T, Hg, D, device=dev, dtype=dt) * 0.5)
v = (torch.randn(1, T, H, D, device=dev, dtype=dt) * 0.5)
beta = torch.rand(1, T, H, device=dev, dtype=dt).sigmoid()
g = torch.nn.functional.logsigmoid(torch.rand(1, T, H, device=dev, dtype=torch.float32))
h0 = torch.randn(1, H, D, D, device=dev, dtype=torch.float32) * 0.05
cu = torch.tensor([0, T], device=dev, dtype=torch.long)

o, ht = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, scale=D ** -0.5,
                               initial_state=h0, output_final_state=True, cu_seqlens=cu,
                               head_first=False, use_qk_l2norm_in_kernel=True)
st = fa2_gdn_shim.stats()
import hashlib
hb = hashlib.sha256(o.float().cpu().numpy().tobytes()).hexdigest()[:16]
print("RESULT " + json.dumps({
    "hash": hb,
    "mean": float(o.float().mean()), "absmax": float(o.float().abs().max()),
    "nan": int(torch.isnan(o.float()).sum()),
    "patched": bool(getattr(chunk_gated_delta_rule, "__fa2sm70__", False)),
    "calls": st["calls"], "ours": st["ours"], "fallback": st["fallback"],
    "reason": st["reason"],
}))
"""


def run(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    p = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            import json
            return json.loads(line[7:])
    print(p.stdout[-3000:])
    print(p.stderr[-4000:], file=sys.stderr)
    raise SystemExit("дочерний процесс не выдал RESULT")


def main():
    if not os.environ.get("VLLM_SITE_PACKAGES"):
        raise SystemExit("set VLLM_SITE_PACKAGES to the vLLM site-packages directory")
    dt = sys.argv[1] if len(sys.argv) > 1 else "bf16"
    print(f"=== ПОВЕДЕНЧЕСКИЙ ГЕЙТ ВРЕЗКИ СКАНА GDN (dtype={dt}) ===")
    off = run({"FA2SM70_GDN": "0", "GATE_DT": dt})
    on = run({"FA2SM70_GDN": "1", "GATE_DT": dt})
    print(f"  ВЫКЛ: hash={off['hash']} patched={off['patched']} "
          f"calls={off['calls']} ours={off['ours']} fallback={off['fallback']}")
    print(f"  ВКЛ : hash={on['hash']}  patched={on['patched']} "
          f"calls={on['calls']} ours={on['ours']} fallback={on['fallback']}"
          + (f" reason={on['reason']}" if on["reason"] else ""))

    # УСЛОВИЕ 1: выключено -> подмена не установлена И считает образец.
    c1 = (not off["patched"]) and off["ours"] == 0
    # УСЛОВИЕ 2: включено -> имя ПОДМЕНЕНО, наш путь ОТРАБОТАЛ, откатов НЕТ,
    #            и результат ОТЛИЧАЕТСЯ от образца (иначе подмена ничего не делает).
    c2 = on["patched"] and on["ours"] >= 1 and on["fallback"] == 0 and on["hash"] != off["hash"]
    # УСЛОВИЕ 3 (осмысленность): без NaN и того же масштаба, что образец. Расхождение обязано
    # быть уровня fp16, а не "другой ответ": ловит подмену, которая работает, но считает НЕ ТО.
    rel = abs(on["mean"] - off["mean"]) / (abs(off["mean"]) + 1e-9)
    amax = abs(on["absmax"] - off["absmax"]) / (off["absmax"] + 1e-9)
    c3 = on["nan"] == 0 and amax < 0.05

    print(f"  (1) ВЫКЛ идёт образцом ................ {'ДА' if c1 else 'НЕТ'}")
    print(f"  (2) ВКЛ идёт нашим И результат ИНОЙ ... {'ДА' if c2 else 'НЕТ'}")
    print(f"  (3) ответ осмыслен (NaN 0, |max| ~ ) .. {'ДА' if c3 else 'НЕТ'}"
          f"   d|max|={amax:.2e}  dmean_rel={rel:.2e}")
    ok = c1 and c2 and c3
    print("ГЕЙТ: " + ("ПРОЙДЕН" if ok else "ПРОВАЛЕН"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
