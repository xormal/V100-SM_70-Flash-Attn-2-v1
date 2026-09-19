"""МИКРОБЕНЧ БОЕВОГО ПУТИ: int8-пул + шкалы, тот же вызов, что делает vLLM на каждый слой."""
import os, sys, time, torch
sys.path.insert(0, "/mnt/d1/alex/VLLM_fa2/solutions/fa2_sm70_cutlass_grade")
import fa2_sm70 as F
d = F._ext.decode_ext()
dev = "cuda:0"; torch.manual_seed(0)
# БОЕВЫЕ ФОРМЫ Qwen3.8-27B при TP=2: head_dim=256, 24 головы и 4 KV-головы делятся на две
# карты. Прежние 16/2/128 -- от ДРУГОЙ модели: на них меряется не наш путь, потому что
# EPT = D/32, а именно EPT задаёт и ширину загрузки, и выбор UU внутри ядра.
B, H, Hkv, D, BS = 1, 12, 2, 256, 256
qGrp = int(__import__("os").environ.get("QG","4"))                                    # строки спекуляции (виртуальный батч)
print(f"  B={B} H={H} Hkv={Hkv} D={D} block={BS} qGrp={qGrp}, KV=int8")
print("  kv_len     mks     GB/s")
for kv in (32768, 131072, 262144):
    nb = (kv + BS - 1) // BS
    K = torch.randint(-127, 127, (nb, BS, Hkv, D), device=dev, dtype=torch.int8)
    V = torch.randint(-127, 127, (nb, BS, Hkv, D), device=dev, dtype=torch.int8)
    ks = torch.rand(nb * BS, Hkv, device=dev, dtype=torch.float32) * 0.01 + 0.001
    vs = torch.rand(nb * BS, Hkv, device=dev, dtype=torch.float32) * 0.01 + 0.001
    bt = torch.arange(nb, device=dev, dtype=torch.int32).view(1, nb).contiguous()
    Q = torch.randn(B * qGrp, H, 1, D, device=dev, dtype=torch.float16)
    kmax = torch.full((B * qGrp,), 10.0, device=dev, dtype=torch.float32)
    sl = torch.full((B * qGrp,), kv, device=dev, dtype=torch.int32)
    import os as _o; ns = int(_o.environ.get("NS","32"))
    A = torch.zeros(B * qGrp * H * ns * D, device=dev, dtype=torch.float32)
    Bw = torch.zeros(2 * B * qGrp * H * ns, device=dev, dtype=torch.float32)
    out = torch.zeros(B * qGrp, H * D, device=dev, dtype=torch.float16)
    def zov():
        d.flash_decode_defer_mqa_paged_into(Q, K, V, kmax, bt, BS, 1.0/(D**0.5), ns, kv, sl,
                                            False, 1.0, 1.0, A, Bw, out, out.stride(0),
                                            ks, vs, None, None, None, None, None, 128, qGrp, 1)
    # [ЭТАЛОН] Тот же счёт на торче в float32: деквант int8 по шкалам, softmax, взвешенная V.
    # Без него микробенч меряет скорость КАКОГО-УГОДНО ответа -- а правки ядра портят тихо.
    def etalon():
        Kf = (K.view(-1, Hkv, D).float() * ks.view(-1, Hkv, 1))[:kv]       # [kv, Hkv, D]
        Vf = (V.view(-1, Hkv, D).float() * vs.view(-1, Hkv, 1))[:kv]
        r = H // Hkv
        o = torch.empty(B * qGrp, H, D, device=dev, dtype=torch.float32)
        for h in range(H):
            q = Q[:, h, 0, :].float() * (1.0 / (D ** 0.5))                 # [rows, D]
            s_ = q @ Kf[:, h // r, :].t()                                  # [rows, kv]
            p_ = torch.softmax(s_, dim=-1)
            o[:, h, :] = p_ @ Vf[:, h // r, :]
        return o
    try:
        for _ in range(3): zov()
        torch.cuda.synchronize()
        N = 20; t0 = time.time()
        for _ in range(N): zov()
        torch.cuda.synchronize()
        mks = (time.time()-t0)/N*1e6
        bajt = 2 * kv * Hkv * D * 1        # int8
        if kv <= 32768:                    # эталон на длинных не строим: память O(kv*D)
            e = etalon()
            g = out.view(B * qGrp, H, D).float()
            rel = (torch.linalg.vector_norm(g - e) / torch.linalg.vector_norm(e)).item()
            print(f"  {kv:>7}  {mks:>7.1f}  {bajt/(mks*1e-6)/1e9:>6.1f}   relL2={rel:.2e}")
        else:
            print(f"  {kv:>7}  {mks:>7.1f}  {bajt/(mks*1e-6)/1e9:>6.1f}")
    except Exception as e:
        print(f"  {kv:>7}  otkaz: {str(e)[:110]}")
