# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# GPU test matrix runner: executes every correctness suite as a subprocess and prints a pass/fail
# summary. Intended for the dedicated sm_70 CI runner. Run:  python tests/run_all.py
# Exit code is non-zero if any suite fails.
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = [
    "test_mxfp4.py",
    "test_attention.py",     # exact prefill vs fp32 + SDPA (causal/GQA/MQA/large-d)
    "test_decode.py",        # split-KV decode (fp16/int8/int4, deferred-U, MQA, VEQK)
    "test_backward.py",      # dQ/dK/dV vs fp32 (MHA/GQA/MQA, causal, Sq!=Sk)
    "test_varlen.py",        # cu_seqlens forward + backward (ragged)
    "test_dropout.py",       # reproducible Philox dropout (p=0 no-op, unbiased)
    "test_paged.py",         # paged block-table decode (uniform + ragged seqlens)
    "test_quant_cache.py",   # stateful int8/int4 KV cache (bit-exact vs one-shot quantize_kv)
    "test_decode_mod.py",    # decode-path softcap / ALiBi (online path)
    "test_alibi.py",         # per-head ALiBi position bias
    "test_window.py",        # local-attention window: causal past-window + two-sided (all head dims)
    "test_fa_interface.py",  # flash-attn-compatible [B,S,H,D] wrappers
    "test_volta_hmma.py",    # hand-written m8n8k4 operand map -- asserts COVERAGE, not just values
    "test_volta_fwd.py",     # hand-written block mainloop, head_dim=256 (GQA, batch, ragged, bottom-right causal)
    "test_gptq_dequant.py",  # GPTQ int8 weight unpack vs an explicitly-unpacked reference
    "test_w8a16.py",         # fused W8A16 GEMV (weights consumed packed)
    "test_vllm_i8_gather.py",# paged->int8 gather (bit-exact) + the byte prefill route into the role kernel
    "test_vllm_i8_pool.py",  # int8 KV POOL end to end: store -> gather -> prefill -> decode, ONE format
    "test_prefill_bytes.py", # public byte-KV prefill (format 8): BITWISE equal to the plain int8 path
    "test_fwd_e4m3.py",      # direct fp8_e4m3 path (KVFMT=11): exact expansion + bitwise vs the e5m2 reference
]


def main():
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    results = []
    for s in SUITES:
        path = os.path.join(HERE, s)
        if not os.path.exists(path):
            results.append((s, "MISSING", 0.0)); continue
        t0 = time.time()
        # ДВА СТИЛЯ НАБОРОВ, И ЗАПУСКАТЬ ИХ НАДО ПО-РАЗНОМУ.
        # Часть файлов -- скрипты: проверки идут на уровне модуля и выполняются простым импортом.
        # Часть -- в стиле pytest: `def test_*`, которые СОБИРАЕТ ТОЛЬКО PYTEST. Запуск такого файла
        # как скрипта импортирует модуль, не выполняет ни одной проверки и возвращает 0 -- то есть
        # даёт ПУСТОЙ "PASS". Так и было до 31.07.2026: четыре набора (test_volta_fwd,
        # test_vllm_i8_gather, test_vllm_i8_pool, test_prefill_bytes) -- 58 собираемых проверок -- числились
        # пройденными, ни разу не исполнившись. Ровно тот класс отказа, ради которого в этом проекте
        # гейт проверяется ПОВЕДЕНИЕМ: харнесс обязан доказать, что тесты шли, а не что файл открылся.
        src = open(path, encoding="utf-8").read()
        pytest_style = "\ndef test_" in src
        if pytest_style:
            cmd = [sys.executable, "-m", "pytest", "-q", path]
        else:
            cmd = [sys.executable, path]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.time() - t0
        # код 5 у pytest -- "ни одного теста не собрано"; для нас это ОТКАЗ, а не успех.
        ok = p.returncode == 0
        if pytest_style and p.returncode == 5:
            ok = False
            p = subprocess.CompletedProcess(cmd, 5, p.stdout + "\nНИ ОДНОГО ТЕСТА НЕ СОБРАНО", p.stderr)
        results.append((s, "PASS" if ok else "FAIL", dt))
        if not ok:
            print(f"\n----- {s} FAILED -----\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}\n")
    print("\n=== fa2_sm70 test matrix ===")
    for s, st, dt in results:
        print(f"  {st:5s}  {s:24s} {dt:6.1f}s")
    n_fail = sum(1 for _, st, _ in results if st != "PASS")
    print(f"=== {len(results) - n_fail}/{len(results)} suites passed ===")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
