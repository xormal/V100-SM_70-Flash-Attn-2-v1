import sys, torch
sys.path.insert(0, "/mnt/d1/alex/VLLM_fa2/solutions/fa2_sm70_cutlass_grade")
from fa2_sm70 import _ext
e = _ext.gdn_ext()
torch.manual_seed(5)
dev = "cuda"
H, HV, K = 2, 4, 128
W = 2
def данные(T):
    q = torch.randn(1, T, H, K, dtype=torch.float16, device=dev) * 0.3
    k = torch.randn(1, T, H, K, dtype=torch.float16, device=dev) * 0.3
    v = torch.randn(1, T, HV, K, dtype=torch.float16, device=dev) * 0.3
    a = torch.randn(1, T, HV, dtype=torch.float16, device=dev) * 0.3
    b = torch.randn(1, T, HV, dtype=torch.float16, device=dev) * 0.3
    return q, k, v, a, b
A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
dt_bias = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
q5, k5, v5, a5, b5 = данные(2 * W + 1)
def срез(т, инд): return т[:, инд].contiguous()
якорь = [0]; ветвьA = [0, 1, 2]; ветвьB = [0, 3, 4]
def пуск(qq, kk, vv, aa, bb, слоты, branch_at, пул):
    T = qq.shape[1]
    si = torch.tensor([слоты], dtype=torch.int32, device=dev)
    na = torch.tensor([1], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    return e.gdn_rec_gating(A_log, aa, bb, dt_bias, qq, kk, vv, 1.0 / (K ** 0.5),
                            пул, si, na, cu, True, None, None, branch_at)
# эталон: цепь B отдельно (якорь, b0, b1) в слоты 10,13,14
пулЭ = torch.zeros(20, HV, K, K, dtype=torch.float32, device=dev)
oB = пуск(срез(q5, ветвьB), срез(k5, ветвьB), срез(v5, ветвьB),
          срез(a5, ветвьB), срез(b5, ветвьB), [10, 13, 14], 0, пулЭ)
# и цепь A отдельно (для контроля строк 0..2)
пулА = torch.zeros(20, HV, K, K, dtype=torch.float32, device=dev)
oA = пуск(срез(q5, ветвьA), срез(k5, ветвьA), срез(v5, ветвьA),
          срез(a5, ветвьA), срез(b5, ветвьA), [10, 11, 12], 0, пулА)
# проверяемое: один прогон T=5 с ветвлением на строке 3
пулД = torch.zeros(20, HV, K, K, dtype=torch.float32, device=dev)
oД = пуск(q5, k5, v5, a5, b5, [10, 11, 12, 13, 14], W + 1, пулД)
def rl2(x, y): return (x - y).float().norm().item() / max(y.float().norm().item(), 1e-9)
print(f"  строки A (0..2): relL2 = {rl2(oД[:, :3], oA):.2e}")
print(f"  строки B (3,4) против эталонной цепи B: relL2 = {rl2(oД[:, 3:], oB[:, 1:]):.2e}")
print(f"  состояние слота 13: relL2 = {rl2(пулД[13], пулЭ[13]):.2e}")
print(f"  состояние слота 14: relL2 = {rl2(пулД[14], пулЭ[14]):.2e}")
print(f"  контроль БЕЗ ветвления (branch_at=0) обязан РАСХОДИТЬСЯ:")
пулК = torch.zeros(20, HV, K, K, dtype=torch.float32, device=dev)
oК = пуск(q5, k5, v5, a5, b5, [10, 11, 12, 13, 14], 0, пулК)
print(f"  строки B цепью: relL2 = {rl2(oК[:, 3:], oB[:, 1:]):.2e} (должно быть >> 0)")
