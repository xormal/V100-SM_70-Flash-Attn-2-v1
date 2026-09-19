#!/usr/bin/env python3
"""ОБУЧЕНИЕ ЧЕРНОВИКА DFlash2 В ХАРНЕССЕ, КОТОРЫЙ ВОСПРОИЗВОДИТ БОЙ.

Предыстория (записка 25 §202 и далее). Три прежних захода учили черновик в приборе, который
давал 1-4 % точности позиции 0 там, где бой даёт 54-77 %: он подавал ВЫХОД `fc` за ВХОД,
крутил RoPE чередованием вместо neox, ронял тело в NaN на fp16 и, главное, не подавал
БОНУСНЫЙ токен -- единственный токенный вход черновика. Прибор `dflash2_gejt.py` исправлен и
даёт tau 1.96-2.00 при боевых 1.79-1.88; учить можно только через него.

ЧТО УЧИМ. Задача черновика в бою: по впрыснутому контексту (признаки цели через `fc`) и
бонусному токену предсказать k следующих. Здесь ровно она: потеря -- взвешенная по позициям
перекрёстная энтропия против токенов, которые цель выдала на самом деле.

ЗАПУСК:
  SHAGOV=2000 B=8 KONTEKST=128 UCHIT=fc,norm,attn LR=1e-5 \
  python3 dflash2_uchit.py <корпус> --сохранить /mnt/d1/alex/DFlash2-nash
"""
import os, sys, json, time, math
import torch

_КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_КОРЕНЬ, "..", "integrations", "vllm_sm70"))
sys.path.insert(0, _КОРЕНЬ)
from dflash2.telo_polnoe import ЧерновикDFlash2
from dflash2_gejt import цепочка, тензор_цели, ЦЕЛЬ


def группы_весов(ч, что):
    """Какие тензоры отпускаем в обучение. Полный набор в fp32 с Adam не помещается в 32 ГиБ
    (1.86 млрд параметров -> 31 ГиБ только на веса и моменты), поэтому набор -- рычаг."""
    что = set(x.strip() for x in что.split(",") if x.strip())
    выбор = []
    for имя, т in ч.в.items():
        if not hasattr(т, "requires_grad"):
            continue
        взять = (
            ("fc" in что and имя == "fc.weight")
            or ("norm" in что and имя.endswith("norm.weight"))
            or ("attn" in что and ".self_attn." in имя)
            or ("mlp" in что and (".mlp." in имя or "_conv." in имя))
            or ("sel" in что and имя.startswith("candidate_selector."))
        )
        if взять:
            т.requires_grad_(True)
            выбор.append((имя, т))
    return выбор


def main():
    корпус = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d1/alex/dflash2_syryo"
    сохранить = None
    if "--сохранить" in sys.argv:
        сохранить = sys.argv[sys.argv.index("--сохранить") + 1]
    ЧЕРН = os.environ.get("CHERN", "/mnt/d1/alex/Qwen3.8-27B-DFlash2")
    шагов = int(os.environ.get("SHAGOV", "2000"))
    B = int(os.environ.get("B", "8"))
    L = int(os.environ.get("KONTEKST", "128"))
    k = int(os.environ.get("SPEC_K", "3"))
    lr = float(os.environ.get("LR", "1e-5"))
    что = os.environ.get("UCHIT", "fc,norm,attn")
    предел = int(os.environ.get("PREDEL", "200000"))
    вес_kl = float(os.environ.get("KL", "0"))
    # [РЕШЁТЧАТЫЙ ЧЕРНОВИК -- новая конструкция, 08.09]
    # Замер показал: блок читает ТОЛЬКО маски (у строки с настоящим токеном выход бесполезен --
    # 10 % против 57 %), поэтому «второй проход с подставленным токеном» на нынешних весах
    # рушит точность втрое. Значит частично раскрытый блок надо УЧИТЬ. Здесь с вероятностью
    # RASKR первые z (1..k-1) позиций блока раскрываются, а потеря считается ТОЛЬКО по ещё
    # закрытым: модель учится дописывать блок, зная его начало. На выводе это даёт второй
    # проход, который добавляет к позиции j знание позиции j-1 -- то, чего у параллельного
    # блока нет по построению (маргинальная точность падает 57 -> 31 -> 21 %).
    # Раскрываем НЕ ТОЛЬКО истиной: на выводе раскрытый токен -- ДОГАДКА (верна в 57 %), и
    # модель, учившаяся только на истине, встретит незнакомый вход. Доля SHUM задаёт, как
    # часто вместо истины подставляется чужой токен из того же ряда.
    раскр = float(os.environ.get("RASKR", "0"))
    шум = float(os.environ.get("SHUM", "0.35"))
    # Веса позиций: приёмка обрывается на первом расхождении, значит ранняя позиция дороже.
    ВЕСА = [float(x) for x in os.environ.get(
        "POS_W", ",".join(str(v) for v in (1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.007))
    ).split(",")][:k]
    torch.manual_seed(int(os.environ.get("SEED", "20260908")))
    dev = "cuda"

    Ф, Т, ЛОГ, отрезки = цепочка(корпус, предел, вернуть_логиты=True)
    годные = [(a, b) for a, b in отрезки if b - a >= L + k + 6]
    всего = sum(b - a for a, b in годные)
    print(f"  корпус: {Ф.shape[0]} токенов, годных отрезков {len(годные)} ({всего} токенов)",
          flush=True)
    гр = int(len(годные) * 0.9)
    учебные, отложенные = годные[:гр], годные[гр:]
    print(f"  учебных отрезков {len(учебные)}, отложенных {len(отложенные)}", flush=True)

    конф = json.load(open(os.path.join(ЧЕРН, "config.json")))
    from safetensors.torch import load_file
    ч = ЧерновикDFlash2(load_file(os.path.join(ЧЕРН, "model.safetensors")),
                        конф, устройство=dev, тип=torch.float32)
    lm = тензор_цели("lm_head.weight", dev, torch.float16)
    idx = json.load(open(os.path.join(ЦЕЛЬ, "model.safetensors.index.json")))
    имя_вл = next(n for n in idx["weight_map"] if n.endswith("embed_tokens.weight"))
    вложения = тензор_цели(имя_вл, dev, torch.float16)
    вл_маски = вложения[ч.маска_id].float()

    выбор = группы_весов(ч, что)
    п_всего = sum(т.numel() for _, т in выбор)
    print(f"  учим {len(выбор)} тензоров, {п_всего / 1e6:.1f} млн параметров ({что})", flush=True)
    опт = torch.optim.AdamW([т for _, т in выбор], lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    план = torch.optim.lr_scheduler.OneCycleLR(опт, max_lr=lr, total_steps=шагов,
                                               pct_start=0.05, final_div_factor=20)

    def проход(отр, обучение):
        """Один минибатч. Возвращает (потеря, [совпало по позициям], число проб)."""
        потеря = 0.0
        совп = [0] * k
        учтено = [0] * k
        for _ in range(B):
            a, b = отр[torch.randint(len(отр), (1,)).item()]
            i = torch.randint(a + L, b - k - 3, (1,)).item()
            сыр = Ф[i - L + 1:i + 1].to(dev, torch.float32)
            сост = ч.слить_признаки(сыр)
            поз_к = torch.arange(L, device=dev)
            K, V = ч.kv_контекста(сост, поз_к)
            kv = [(K[l], V[l], поз_к) for l in range(ч.слоёв)]
            истина = Т[i + 2:i + 2 + k].to(dev)
            # z -- сколько позиций блока раскрыто (0 = как в бою сегодня)
            z = 0
            if обучение and раскр > 0 and torch.rand(1).item() < раскр and k > 1:
                z = int(torch.randint(1, k, (1,)).item())
            строки = [вложения[Т[i + 1].item()][None].float()]
            for j in range(k):
                if j < z:
                    ток = истина[j].item()
                    if torch.rand(1).item() < шум:
                        # чужой токен из того же ряда -- дешёвая замена «догадке черновика»
                        ток = int(Т[torch.randint(a, b, (1,)).item()].item())
                    строки.append(вложения[ток][None].float())
                else:
                    строки.append(вл_маски[None])
            вх = torch.cat(строки, dim=0)
            скр = ч.проход_блока(вх, torch.arange(L, L + k + 1, device=dev), kv_внеш=kv)[1:]
            логиты = (скр.half() @ lm.T).float()
            for j in range(z, k):
                потеря = потеря + ВЕСА[j] * torch.nn.functional.cross_entropy(
                    логиты[j][None], истина[j][None])
                if вес_kl > 0 and ЛОГ is not None:
                    зн, ид = ЛОГ[0][i + 1 + j].to(dev).float(), ЛОГ[1][i + 1 + j].to(dev).long()  # логит позиции p+1+j предсказывает p+2+j
                    p = torch.softmax(зн, -1)
                    lq = torch.log_softmax(логиты[j], -1)[ид]
                    потеря = потеря + вес_kl * ВЕСА[j] * (p * (p.clamp_min(1e-9).log() - lq)).sum()
            совп = [совп[j] + int(логиты[j].argmax().item() == истина[j].item())
                    if j >= z else совп[j] for j in range(k)]
            учтено = [учтено[j] + (1 if j >= z else 0) for j in range(k)]
        return потеря / B, совп, учтено

    t0 = time.time()
    окно_потеря, окно_совп, окно_n = 0.0, [0] * k, [0] * k
    for шаг in range(1, шагов + 1):
        опт.zero_grad(set_to_none=True)
        потеря, совп, учт = проход(учебные, True)
        потеря.backward()
        torch.nn.utils.clip_grad_norm_([т for _, т in выбор], 1.0)
        опт.step(); план.step()
        окно_потеря += потеря.item(); окно_совп = [окно_совп[j] + совп[j] for j in range(k)]
        окно_n = [окно_n[j] + учт[j] for j in range(k)]
        if шаг % 50 == 0:
            стр = " ".join(f"поз{j}={окно_совп[j] / max(окно_n[j], 1) * 100:.1f}%"
                           for j in range(k))
            print(f"  шаг {шаг}/{шагов} потеря={окно_потеря / 50:.4f} {стр} "
                  f"lr={план.get_last_lr()[0]:.2e} ({time.time() - t0:.0f} с)", flush=True)
            окно_потеря, окно_совп, окно_n = 0.0, [0] * k, [0] * k
        if шаг % 500 == 0 or шаг == шагов:
            with torch.no_grad():
                с = [0] * k; н = [0] * k
                for _ in range(20):
                    _, c2, n2 = проход(отложенные, False)
                    с = [с[j] + c2[j] for j in range(k)]
                    н = [н[j] + n2[j] for j in range(k)]
            print(f"  [ОТЛОЖЕННАЯ] " + "  ".join(
                f"поз{j}={с[j] / max(н[j], 1) * 100:.1f}%" for j in range(k)), flush=True)

    if сохранить:
        os.makedirs(сохранить, exist_ok=True)
        from safetensors.torch import save_file
        веса = {и: т.detach().to(torch.float16).cpu().contiguous() for и, т in ч.в.items()}
        save_file(веса, os.path.join(сохранить, "model.safetensors"))
        import shutil
        for ф in ("config.json",):
            shutil.copy(os.path.join(ЧЕРН, ф), os.path.join(сохранить, ф))
        print(f"  сохранено в {сохранить}", flush=True)


if __name__ == "__main__":
    main()
