"""Интеграционный офлайн-тест СВЁРТКИ при принятии ветви B (семантика откалибрована:
история в ГОЛОВЕ окна; после шага окно = [old[nacc], old[nacc+1], строки шага, нули]).

Схема: 2-й вызов ветви B в черновой слот -> при принятии B окно подменяется -> следующий
шаг идёт с ОБЫЧНЫМ nacc=m (сдвиг +W нужен только SSM-чтению, свёртке -- нет).
"""
import sys, torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
torch.manual_seed(7); dev = "cuda"
DIM, WIDTH, K_SPEC, W = 32, 4, 4, 2
SL = WIDTH - 1 + K_SPEC
T = 2 * W + 1
вес = torch.randn(DIM, WIDTH, device=dev) * 0.3
смещ = torch.randn(DIM, device=dev) * 0.1
def эт(токены):
    X = torch.stack(токены, 1); вых = []
    for t in range(X.shape[1]):
        о = torch.zeros(DIM, WIDTH, device=dev); ч = X[:, max(0, t-WIDTH+1):t+1]
        о[:, WIDTH-ч.shape[1]:] = ч; вых.append((о*вес).sum(1)+смещ)
    return torch.stack(вых, 1)
def ядро(группа, пул, слот, nacc, Tв):
    return causal_conv1d_update(torch.stack(группа).clone(), пул, вес, смещ, None,
        conv_state_indices=torch.tensor([слот], dtype=torch.int32, device=dev),
        num_accepted_tokens=torch.tensor([nacc], dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor([0, Tв], dtype=torch.int32, device=dev),
        max_query_len=Tв, validate_data=False)
h = [torch.randn(DIM, device=dev)*0.5 for _ in range(3)]
я1, a0, a1, b0, b1 = [torch.randn(DIM, device=dev)*0.5 for _ in range(5)]
я2, c0, c1, d0, d1 = [torch.randn(DIM, device=dev)*0.5 for _ in range(5)]
m = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ветвь = sys.argv[2] if len(sys.argv) > 2 else "B"
пул = torch.zeros(2, DIM, SL, device=dev); ГЛ, ЧЕРН = 0, 1
for j in range(3): пул[ГЛ, :, j] = h[j]        # история в ГОЛОВЕ, nacc_прош=1
пул[ЧЕРН].copy_(пул[ГЛ])
yN = ядро([я1, a0, a1, b0, b1], пул, ГЛ, 1, T)
yB = ядро([я1, b0, b1], пул, ЧЕРН, 1, W + 1)
этB = эт(h + [я1, b0, b1])[:, -3:]
print(f"  ветвь B (2-й вызов): relL2 = {(yB.t()-этB).norm()/этB.norm():.2e}")
этA = эт(h + [я1, a0, a1])[:, -3:]
print(f"  ветвь A (осн. вызов, стр. 0-2): relL2 = {(yN[:3].t()-этA).norm()/этA.norm():.2e}")
if ветвь == "B":
    пул[ГЛ].copy_(пул[ЧЕРН])                   # ПРИНЯТИЕ B: подмена окна
    приняты = [я1, b0, b1][:m]
else:
    приняты = [я1, a0, a1][:m]
yN1 = ядро([я2, c0, c1, d0, d1], пул, ГЛ, m, T)   # ОБЫЧНЫЙ nacc = m
эт1 = эт(h + приняты + [я2, c0, c1])[:, -3:]
print(f"  ветвь {ветвь}, m={m}: шаг N+1 [я2,c0,c1]: relL2 = "
      f"{(yN1[:3].t()-эт1).norm()/эт1.norm():.2e}")
