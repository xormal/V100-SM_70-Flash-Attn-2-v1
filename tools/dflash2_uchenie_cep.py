#!/usr/bin/env python3
"""Обучение черновика DFlash2 на ИСТИННЫХ цепочках (корпус с числом принятых).

Отличие от `dflash2_obuchenie.py`: там задача была «по 32 прошлым шагам предсказать 8
следующих», а метки брались со ВСЕХ позиций пары -- включая посчитанные по отвергнутому
префиксу. Три захода обучения (§95, §96) двигали свою метрику и роняли приёмку именно из-за
этого. Здесь вход и цель ровно те же, что в бою, а проверка идёт БОЕВОЙ мерой (приёмка),
согласованной с сервером по порядку версий.

  CHERN=<каталог> UCHIM=telo|selektor|obe LR=... SOHRANIT=<файл> python3 dflash2_uchenie_cep.py <корпус> <шагов>
"""
import os, sys, glob, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "integrations", "vllm_sm70"))
import dflash2_obuchenie as О
from safetensors.torch import load_file

def цепочка(корпус):
    """Пара -- (состояния[T,h], логиты[T,64], m). Истинны ровно первые m позиций."""
    с, т, зн_, ид_ = [], [], [], []
    for f in sorted(glob.glob(os.path.join(корпус, "*.pt"))):
        try: пары = torch.load(f, map_location="cpu")
        except Exception: continue
        for п in пары:
            if len(п) < 3: continue
            m = max(1, min(int(п[2]), п[0].shape[0]))
            зн, ид = п[1] if isinstance(п[1], tuple) else (None, п[1])
            for j in range(min(m, ид.shape[0])):
                с.append(п[0][j]); т.append(int(ид[j, 0] if ид.dim() > 1 else ид[j]))
                if зн is not None: зн_.append(зн[j]); ид_.append(ид[j])
    return с, т, зн_, ид_

def main():
    корпус = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d1/alex/dflash2_k7"
    шагов = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    ЧЕРН = os.environ.get("CHERN", "/mnt/d1/alex/Qwen3.8-27B-DFlash2")
    УЧИМ = os.environ.get("UCHIM", "telo")
    T, k = int(os.environ.get("KONTEKST", "8")), int(os.environ.get("SPEC_K", "3"))
    БАТЧ = int(os.environ.get("BATCH", "8"))
    dev = "cuda"
    с, т, зн_, ид_ = цепочка(корпус)
    гр = int(len(т) * 0.9)
    print(f"  цепочка: {len(т)} токенов (обучение {гр}, проверка {len(т)-гр})")
    ч = О.ЧерновикDFlash2(load_file(os.path.join(ЧЕРН, "model.safetensors")),
                          json.load(open(os.path.join(ЧЕРН, "config.json"))),
                          устройство=dev, тип=torch.float32)
    for и, t_ in ч.в.items():
        if hasattr(t_, "requires_grad_"):
            сел = и.startswith("candidate_selector.")
            t_.requires_grad_(сел if УЧИМ == "selektor" else (not сел) if УЧИМ == "telo" else True)
    учимые = [t_ for t_ in ч.в.values() if getattr(t_, "requires_grad", False)]
    print(f"  обучаемых тензоров: {len(учимые)} (UCHIM={УЧИМ})")
    W = О.tenzor_celi("lm_head.weight", dev)
    _idx = json.load(open(os.path.join(О.ЦЕЛЬ, "model.safetensors.index.json")))
    _кл = next(n for n in _idx["weight_map"] if n.endswith("embed_tokens.weight"))
    О.ВЛОЖЕНИЕ_МАСКИ = О.stroka_celi(_кл, ч.маска_id, dev).to(ч.тип)
    from dflash2.otbor import отобрать_по_префиксу
    opt = torch.optim.SGD(учимые, lr=float(os.environ.get("LR", "1e-4")), momentum=0.9)

    def проход(i):
        сост = torch.stack(с[i - T:i]).to(dev, ч.тип)
        поз_к = torch.arange(T, device=dev)
        K, V = ч.kv_контекста(сост, поз_к)
        kv = [(K[l], V[l], поз_к) for l in range(ч.слоёв)]
        вл = О.ВЛОЖЕНИЕ_МАСКИ[None].repeat(k, 1)
        скр = ч.проход_блока(вл, torch.arange(T, T + k, device=dev), kv_внеш=kv)
        return скр, (скр.to(torch.float16) @ W.T).float()

    def приёмка(проб=200):
        сум = 0.0
        with torch.no_grad():
            for _ in range(проб):
                i = torch.randint(гр + T, len(т) - k - 1, (1,)).item()
                скр, логиты = проход(i)
                унар, канд = логиты.topk(16, dim=-1)
                h_r = скр @ ч.в["candidate_selector.hidden_projection.weight"].T
                як = torch.tensor([т[i - 1]], device=dev, dtype=torch.long)
                ток = отобрать_по_префиксу(ч.в["candidate_selector.predecessor_codebook"],
                                           ч.в["candidate_selector.successor_codebook"],
                                           канд[None], унар[None], h_r[None], як)[0]
                цель = torch.tensor(т[i:i + k], device=dev, dtype=torch.long)
                д = 1
                for б in (ток[:k] == цель).tolist():
                    if б: д += 1
                    else: break
                сум += д
        return сум / проб

    print(f"  приёмка ДО обучения: {приёмка():.3f}")
    import time; т0 = time.time()
    for шаг in range(шагов):
        opt.zero_grad(set_to_none=True)
        for _ in range(БАТЧ):
            i = torch.randint(T, гр - k - 1, (1,)).item()
            скр, логиты = проход(i)
            цель = torch.tensor(т[i:i + k], device=dev, dtype=torch.long)
            if зн_ and os.environ.get("KL", "1") == "1":
                # Дистилляция по top-64 цели: распределение несёт больше, чем один токен.
                зн = torch.stack([зн_[j] for j in range(i, i + k)]).to(dev, torch.float32)
                ид = torch.stack([ид_[j] for j in range(i, i + k)]).to(dev, torch.long)
                loss = torch.nn.functional.kl_div(
                    torch.log_softmax(torch.gather(логиты, 1, ид), dim=-1),
                    torch.softmax(зн, dim=-1), reduction="batchmean") / БАТЧ
            else:
                loss = torch.nn.functional.cross_entropy(логиты, цель) / БАТЧ
            loss.backward()
        torch.nn.utils.clip_grad_norm_(учимые, 1.0)
        opt.step()
        if шаг % max(1, шагов // 10) == 0 or шаг == шагов - 1:
            print(f"  шаг {шаг:5d}: loss {loss.item()*БАТЧ:.4f}  приёмка {приёмка(120):.3f}", flush=True)
    print(f"  {шагов} шагов за {time.time()-т0:.1f} с")
    print(f"  приёмка ПОСЛЕ обучения: {приёмка(400):.3f}")
    п = os.environ.get("SOHRANIT", "")
    if п:
        torch.save({и: t_.detach().cpu() for и, t_ in ч.в.items()}, п); print(f"  веса сохранены: {п}")

if __name__ == "__main__":
    main()
