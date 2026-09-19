"""Калибровка семантики ядра: где в окне живёт история и что значит nacc."""
import sys, torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
torch.manual_seed(7); dev="cuda"
DIM, WIDTH, K = 8, 4, 4
SL = WIDTH - 1 + K
вес = torch.randn(DIM, WIDTH, device=dev)*0.3
смещ = torch.zeros(DIM, device=dev)
def эт(токены):
    X = torch.stack(токены, 1)
    вых=[]
    for t in range(X.shape[1]):
        о=torch.zeros(DIM,WIDTH,device=dev); ч=X[:,max(0,t-WIDTH+1):t+1]
        о[:,WIDTH-ч.shape[1]:]=ч; вых.append((о*вес).sum(1))
    return torch.stack(вых,1)
h=[torch.randn(DIM,device=dev)*0.5 for _ in range(SL)]
x=[torch.randn(DIM,device=dev)*0.5 for _ in range(3)]
этв = эт(h[-3:]+x)[:, -3:]     # эталон: окно истории = ПОСЛЕДНИЕ 3 из h
for поз in ("хвост","голова"):
    for на in (1,2,3,4):
        пул=torch.zeros(2,DIM,SL,device=dev)
        if поз=="хвост":
            for j in range(3): пул[0,:,SL-3+j]=h[-3+j]
        else:
            for j in range(3): пул[0,:,j]=h[-3+j]
        y=causal_conv1d_update(torch.stack(x).clone(), пул, вес, смещ, None,
            conv_state_indices=torch.tensor([0],dtype=torch.int32,device=dev),
            num_accepted_tokens=torch.tensor([на],dtype=torch.int32,device=dev),
            query_start_loc=torch.tensor([0,3],dtype=torch.int32,device=dev),
            max_query_len=3, validate_data=False)
        print(f"  история в {поз}е, nacc={на}: relL2={(y.t()-этв).norm()/этв.norm():.2e}"
              f"   окно после: {[round(float(пул[0,0,i]),3) for i in range(SL)]}")
print("  токены h[-3:], x:", [round(float(t[0]),3) for t in h[-3:]+x])
