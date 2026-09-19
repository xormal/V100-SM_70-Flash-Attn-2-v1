#!/usr/bin/env python3
"""Офлайн-приёмка черновика DFlash2: та же величина, что движок зовёт Mean acceptance length.

ЗАЧЕМ ОТДЕЛЬНЫЙ ПРИБОР. Обучение (§95, §96) шло на задаче «по 32 прошлым шагам предсказать
8 следующих», а в бою черновик решает другую: по состояниям ОДНОГО шага (T=k+1) предложить
k токенов, которые цель проверит на следующем шаге. На неверной задаче обучение и тела, и
селектора двигало свою метрику и роняло приёмку. Гейт годности прибора прямой: на исходном
черновике он обязан дать примерно то же, что бой (1.58-1.63), а не 1.0.

  CHERN=<каталог> PROB=400 python3 dflash2_priyomka.py <корпус>
"""
import os, sys, glob, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "integrations", "vllm_sm70"))
import dflash2_obuchenie as О

def main():
    # СЕМЯ ФИКСИРУЕТСЯ. Прибор берёт случайные позиции цепочки, и при 300 пробах разброс
    # доходит до 0.12 -- на нём один и тот же обученный черновик давал и +0.7 %, и +16 %.
    # Сравнивать версии можно только на ОДНИХ И ТЕХ ЖЕ позициях.
    torch.manual_seed(int(os.environ.get("SEED", "20260905")))
    корпус = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d1/alex/dflash2_distill"
    ЧЕРН = os.environ.get("CHERN", "/mnt/d1/alex/Qwen3.8-27B-DFlash2")
    проб = int(os.environ.get("PROB", "400"))
    k = int(os.environ.get("SPEC_K", "3"))
    dev = "cuda"
    # ЦЕПОЧКА. Пара -- (состояния[T,hidden], логиты цели[T,64], m). Истинны ровно первые m
    # позиций: остальные цель считала по ОТВЕРГНУТОМУ префиксу. Склеиваем (состояние, токен)
    # подряд -- получается связный текст и ровно те входы, что черновик видит в бою.
    цепь_с, цепь_т = [], []
    for f in sorted(glob.glob(os.path.join(корпус, "*.pt"))):
        try: пары = torch.load(f, map_location="cpu")
        except Exception: continue
        for п in пары:
            if len(п) < 3: continue
            m = max(1, min(int(п[2]), п[0].shape[0]))
            топ1 = (п[1][1][:, 0] if isinstance(п[1], tuple) else п[1].reshape(-1))
            for j in range(min(m, топ1.numel())):
                цепь_с.append(п[0][j]); цепь_т.append(int(топ1[j]))
    print(f"  цепочка: {len(цепь_т)} токенов")
    T = int(os.environ.get("KONTEKST", "8"))
    from safetensors.torch import load_file
    import json
    ч = О.ЧерновикDFlash2(load_file(os.path.join(ЧЕРН, "model.safetensors")),
                          json.load(open(os.path.join(ЧЕРН, "config.json"))),
                          устройство=dev, тип=torch.float32)
    W = О.tenzor_celi("lm_head.weight", dev)
    _idx = json.load(open(os.path.join(О.ЦЕЛЬ, "model.safetensors.index.json")))
    _кл = next(n for n in _idx["weight_map"] if n.endswith("embed_tokens.weight"))
    О.ВЛОЖЕНИЕ_МАСКИ = О.stroka_celi(_кл, ч.маска_id, dev).to(ч.тип)
    from dflash2.otbor import отобрать_по_префиксу
    гр = int(len(цепь_т) * 0.9)                            # отложенная часть
    сум, n = 0.0, 0
    with torch.no_grad():
        for _ in range(проб):
            i = torch.randint(гр + T, len(цепь_т) - k - 1, (1,)).item()
            сост = torch.stack(цепь_с[i - T:i]).to(dev, ч.тип)
            цель_ток = torch.tensor(цепь_т[i:i + k], device=dev, dtype=torch.long)
            поз_к = torch.arange(T, device=dev)
            K, V = ч.kv_контекста(сост, поз_к)
            kv = [(K[l], V[l], поз_к) for l in range(ч.слоёв)]
            вл = О.ВЛОЖЕНИЕ_МАСКИ[None].repeat(k, 1)
            скр = ч.проход_блока(вл, torch.arange(T, T + k, device=dev), kv_внеш=kv)
            логиты = (скр.to(torch.float16) @ W.T).float()
            унар, канд = логиты.topk(16, dim=-1)
            h_r = скр @ ч.в["candidate_selector.hidden_projection.weight"].T
            якорь = torch.tensor([цепь_т[i - 1]], device=dev, dtype=torch.long)
            ток = отобрать_по_префиксу(ч.в["candidate_selector.predecessor_codebook"],
                                       ч.в["candidate_selector.successor_codebook"],
                                       канд[None], унар[None], h_r[None], якорь)[0]
            длина = 1
            for б in (ток[:k] == цель_ток[:k]).tolist():
                if б: длина += 1
                else: break
            сум += длина; n += 1
    print(f"{ЧЕРН.split('/')[-1]}: приёмка {сум / max(n,1):.3f}  (проб {n}, k={k})")

if __name__ == "__main__":
    main()
