"""Полный боевой цикл свёртки: ДВА ДРЕВЕСНЫХ шага подряд, как в сервинге.

Каждый шаг: (1) черновой слот копирует окно основного; (2) второй вызов [я,b0,b1] в
черновой; (3) основной вызов T=5; (4) при принятой B -- окно основного подменяется
черновым (runner). Следующий шаг живёт nacc=m. Эталон -- питон-свёртка истинной цепи.
"""
import sys, torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
torch.manual_seed(9); dev = "cuda"
DIM, WIDTH, K_SPEC, W = 32, 4, 4, 2
SL, T = WIDTH - 1 + K_SPEC, 2 * W + 1
вес = torch.randn(DIM, WIDTH, device=dev) * 0.3
смещ = torch.randn(DIM, device=dev) * 0.1
def эт(ток):
    X = torch.stack(ток, 1); о = []
    for t in range(X.shape[1]):
        w = torch.zeros(DIM, WIDTH, device=dev); ч = X[:, max(0, t-WIDTH+1):t+1]
        w[:, WIDTH-ч.shape[1]:] = ч; о.append((w*вес).sum(1)+смещ)
    return torch.stack(о, 1)
def вызов(гр, пул, слот, nacc, Tв):
    return causal_conv1d_update(torch.stack(гр).clone(), пул, вес, смещ, None,
        conv_state_indices=torch.tensor([слот], dtype=torch.int32, device=dev),
        num_accepted_tokens=torch.tensor([nacc], dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor([0, Tв], dtype=torch.int32, device=dev),
        max_query_len=Tв, validate_data=False)
def шаг_дерева(пул, ГЛ, ЧЕРН, строки, наccв):
    """Боевой порядок: черн <- осн; 2-й вызов (я,b0,b1) в черн; основной T=5."""
    я, a0, a1, b0, b1 = строки
    пул[ЧЕРН].copy_(пул[ГЛ])
    yB = вызов([я, b0, b1], пул, ЧЕРН, наccв, W + 1)
    yосн = вызов(строки, пул, ГЛ, наccв, T)
    return yосн, yB
h = [torch.randn(DIM, device=dev)*0.5 for _ in range(3)]
ш1 = [torch.randn(DIM, device=dev)*0.5 for _ in range(5)]   # я1,a0,a1,b0,b1
ш2 = [torch.randn(DIM, device=dev)*0.5 for _ in range(5)]   # я2,c0,c1,d0,d1
ш3 = [torch.randn(DIM, device=dev)*0.5 for _ in range(3)]   # я3,e0,e1 (проверка 3-го шага)
m1 = int(sys.argv[1]); в1 = sys.argv[2]        # принято на шаге 1, ветвь
m2 = int(sys.argv[3]); в2 = sys.argv[4]        # принято на шаге 2, ветвь
пул = torch.zeros(2, DIM, SL, device=dev)
for j in range(3): пул[0, :, j] = h[j]
# ШАГ 1 (nacc=1 после «прошлого» одиночного)
y1, y1B = шаг_дерева(пул, 0, 1, ш1, 1)
if в1 == "B": пул[0].copy_(пул[1])
пр1 = ([ш1[0], ш1[3], ш1[4]] if в1 == "B" else ш1[:3])[:m1]
# ШАГ 2
y2, y2B = шаг_дерева(пул, 0, 1, ш2, m1)
if в2 == "B": пул[0].copy_(пул[1])
пр2 = ([ш2[0], ш2[3], ш2[4]] if в2 == "B" else ш2[:3])[:m2]
# сверка строк [я2, ветвь] шага 2 и якорной строки шага 3
ист2 = h + пр1
это2 = эт(ист2 + ([ш2[0], ш2[3], ш2[4]] if в2 == "B" else ш2[:3]))[:, -3:]
про2 = torch.stack([y2[0], *(y2B[1:] if в2 == "B" else y2[1:3])]).t()
print(f"  ш2 {в1}{m1}->{в2}{m2}: строки [я2,{'b' if в2=='B' else 'a'}]: "
      f"relL2={(про2-это2).norm()/это2.norm():.2e}", end="")
y3 = вызов(ш3, пул, 0, m2, W + 1)
эт3 = эт(ист2 + пр2 + ш3)[:, -3:]
print(f"   ш3 [я3,e0,e1]: relL2={(y3.t()-эт3).norm()/эт3.norm():.2e}")
