"""ДООБУЧЕНИЕ ЧЕРНОВИКА DFlash2 ПОД НАШУ МОДЕЛЬ (ABLITERATED).

ЗАЧЕМ. Черновик обучен для БАЗОВОЙ Qwen3.8-27B (его README и config), боевая сеть --
ABLITERATED. Отсюда tau 1.79-1.88 при заявленных авторами 3.0+: предел НЕСООТВЕТСТВИЯ, а не
конструкции (на базовой тот же черновик давал 1.08 на блок против 0.90 на нашей).

КАК УЧИМ РОВНО ТОМУ, ЧТО ДЕЛАЕТСЯ В БОЮ. В выводе черновик не прогоняет контекст: его K/V
ВПРЫСКИВАЮТСЯ из признаков цели. Поэтому обучение идёт через ту же сшивку: `kv_контекста`
даёт K/V из состояний корпуса, `проход_блока` принимает их доводом `kv_внеш` (добавлен 01.09,
при None -- прежнее поведение до бита, гейты эталона целы).

ДАННЫЕ. Пары собраны самим сервером (`FA2SM70_DFLASH2_DUMP`): состояния цели ПОСЛЕ `fc` и её
принятые токены, шаги одного запроса идут подряд. Контекст берётся из соседних шагов, метки --
из следующих.

ЗАПУСК:  python3 obuchenie.py <katalog_par> [shagov] [--sohranit put]
"""
import sys, glob, os, json, time
import torch

sys.path.insert(0, "/mnt/d1/alex/VLLM_fa2/solutions/fa2_sm70_cutlass_grade/integrations/vllm_sm70")
from dflash2.telo_polnoe import ЧерновикDFlash2

КАТ   = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d1/alex/dflash2_pary"
ШАГОВ = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ЧЕРН  = os.environ.get("CHERN", "/mnt/d1/alex/Qwen3.8-27B-DFlash2")
ЦЕЛЬ  = "/mnt/d1/alex/VLLM_ch/Qwen3.8-27B-ABL_int8_i12"
КОНТЕКСТ = 32
ВЛОЖЕНИЕ_МАСКИ = None          # сколько шагов корпуса берём как контекст впрыска

def tenzor_celi(имя, dev):
    """Один тензор из целевой модели: голова словаря и вложения у черновика ЧУЖИЕ -- он их
    не хранит (в его весах 81 тензор, ни одного embed/lm_head). Это и правильно: черновик
    работает В ПРОСТРАНСТВЕ цели, поэтому и учить его надо через её же голову и вложения."""
    from safetensors.torch import safe_open
    idx = json.load(open(os.path.join(ЦЕЛЬ, "model.safetensors.index.json")))
    файл = idx["weight_map"][имя]
    with safe_open(os.path.join(ЦЕЛЬ, файл), framework="pt", device="cpu") as f:
        # fp16: голова цели 248320x5120 в fp32 занимает 5 ГБ и не оставляет места
        # состояниям оптимизатора. Она ЗАМОРОЖЕНА, точность fp16 здесь достаточна.
        return f.get_tensor(имя).to(dev, torch.float16)

def stroka_celi(имя, idx_stroki, dev):
    """ОДНА строка большой матрицы: вложение маски -- это один вектор, а матрица вложений
    занимает 2.5 ГБ. Грузим на процессоре, берём строку, и только её кладём на карту."""
    from safetensors.torch import safe_open
    idx = json.load(open(os.path.join(ЦЕЛЬ, "model.safetensors.index.json")))
    with safe_open(os.path.join(ЦЕЛЬ, idx["weight_map"][имя]), framework="pt", device="cpu") as f:
        return f.get_tensor(имя)[idx_stroki].to(dev, torch.float32)

def _neispolzuemo(dev):
    from safetensors.torch import safe_open
    idx = json.load(open(os.path.join(ЦЕЛЬ, "model.safetensors.index.json")))
    файл = idx["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(ЦЕЛЬ, файл), framework="pt", device="cpu") as f:
        W = f.get_tensor("lm_head.weight")
    return W.to(dev, torch.float32)

def корпус():
    ряд = []
    for f in sorted(glob.glob(os.path.join(КАТ, "pary_*.pt"))):
        try: ряд += torch.load(f, map_location="cpu")
        except Exception as e: print(f"  срез пропущен: {e}")
    return ряд

def _проверка(ч, W, ряд_вал, блок, dev, проб=int(os.environ.get("PROB","16"))):
    """Loss на отложенной части корпуса: без градиентов, той же мерой, что и обучение."""
    import torch as _t
    if len(ряд_вал) <= КОНТЕКСТ + блок + 1:
        return float("nan")
    сум = 0.0
    with _t.no_grad():
        for _ in range(проб):
            i = _t.randint(КОНТЕКСТ, len(ряд_вал) - блок - 1, (1,)).item()
            сост = _t.stack([ряд_вал[j][0][-1] for j in range(i - КОНТЕКСТ, i)]).to(dev, ч.тип)
            дист = isinstance(ряд_вал[i + 1][1], tuple)
            поз_к = _t.arange(КОНТЕКСТ, device=dev)
            K, V = ч.kv_контекста(сост, поз_к)
            kv = [(K[l], V[l], поз_к) for l in range(ч.слоёв)]
            вл = ВЛОЖЕНИЕ_МАСКИ[None].repeat(блок, 1)
            поз_б = _t.arange(КОНТЕКСТ, КОНТЕКСТ + блок, device=dev)
            _скр = ч.проход_блока(вл, поз_б, kv_внеш=kv)
            логиты = (_скр.to(_t.float16) @ W.T).float()
            if СЕЛЕКТОР:
                # При замороженном теле loss тела постоянен и мерить его бессмысленно.
                # Настоящая мера здесь -- ДОЛЯ СОВПАВШИХ: попал ли токен, выбранный по рёбрам,
                # в токен цели. Это ровно та величина, которая решает приёмку.
                from dflash2.otbor import оценить_рёбра
                унар, канд = логиты.topk(16, dim=-1)
                h_r = _скр @ ч.в["candidate_selector.hidden_projection.weight"].T
                цель_ток = (_t.stack([ряд_вал[i + j + 1][1][1][-1][0] for j in range(блок)]).to(dev, _t.long)
                            if дист else
                            _t.tensor([ряд_вал[i + j + 1][1].reshape(-1)[0].item() for j in range(блок)],
                                      dtype=_t.long, device=dev))
                рёбра = оценить_рёбра(ч.в["candidate_selector.predecessor_codebook"],
                                      ч.в["candidate_selector.successor_codebook"],
                                      канд[None], унар[None], h_r[None], цель_ток[:1])
                # БОЕВАЯ МЕРА: путь строится САМ (Горнер, как в бою), и считается длина
                # совпавшего НАЧАЛА -- ровно то, что движок зовёт Mean acceptance length.
                # Прежняя мера (совпадение на позиции при ИСТИННОМ предшественнике) показала
                # рост втрое, а приёмка в бою упала: teacher forcing прячет то, что путь
                # разъезжается с первой же ошибки.
                if _t.is_grad_enabled() is False and os.environ.get("PRIYOMKA", "1") == "1":
                    from dflash2.otbor import отобрать_по_префиксу
                    ток = отобрать_по_префиксу(ч.в["candidate_selector.predecessor_codebook"],
                                               ч.в["candidate_selector.successor_codebook"],
                                               канд[None], унар[None], h_r[None], цель_ток[:1])[0]
                    _k = int(os.environ.get("SPEC_K", "3"))
                    совп = (ток[:_k] == цель_ток[:_k])
                    длина = 1
                    for _b in совп.tolist():
                        if _b: длина += 1
                        else: break
                    сум += длина
                    continue
                есть = (канд == цель_ток[:, None]); м2 = есть.any(dim=-1)
                if м2.any():
                    поз = есть.float().argmax(dim=-1)
                    p_ид = _t.cat((_t.zeros(1, dtype=_t.long, device=dev), поз[:-1]))
                    р = рёбра[0].gather(1, p_ид[:, None, None].expand(-1, 1, рёбра.shape[3])).squeeze(1)
                    сум += (канд.gather(1, р.argmax(-1)[:, None]).squeeze(1) == цель_ток).float().mean().item()
                continue
            if дист:
                зн = _t.stack([ряд_вал[i + j + 1][1][0][-1] for j in range(блок)]).to(dev, _t.float32)
                ид = _t.stack([ряд_вал[i + j + 1][1][1][-1] for j in range(блок)]).to(dev, _t.long)
                сум += _t.nn.functional.kl_div(_t.log_softmax(_t.gather(логиты, 1, ид), dim=-1),
                                               _t.softmax(зн, dim=-1), reduction="batchmean").item()
            else:
                м = _t.tensor([ряд_вал[i + j + 1][1].reshape(-1)[0].item() for j in range(блок)],
                              dtype=_t.long, device=dev)
                сум += _t.nn.functional.cross_entropy(логиты, м).item()
    return сум / проб

СЕЛЕКТОР = os.environ.get("SEL", "0") == "1"
# [ПРОВЕРКА ХАРНЕССА, 08.09] Базовый черновик даёт в бою 54-77 % на позиции 0, а в
# этом харнессе -- 1 %. Значит харнесс воспроизводит НЕ ТУ задачу. Первый подозреваемый
# -- сдвиг меток на шаг. Рычаг позволяет проверить это замером, а не рассуждением.
_СДВИГ_МЕТОК = int(os.environ.get("DF_SHIFT", "1"))

# [ВЕСА ПОЗИЦИЙ, 08.09] Потеря считалась РАВНОМЕРНО по всем 8 позициям блока, а
# приёмка так не устроена: позиция j вообще участвует, только если приняты все до
# неё. Замер на боевом (проза): приёмка по позициям 0.56 / 0.34 / 0.14, то есть
# вероятность ДОЙТИ до позиции падает вчетверо каждые две позиции. Производная
# приёмки по позициям даёт доли 0.62 / 0.26 / 0.11 -- позиция 0 весит впятеро
# больше третьей, а восьмая не весит почти ничего.
# Учить их поровну -- значит тратить ёмкость черновика на то, что почти не влияет.
_ВЕС_ПОЗИЦИЙ = os.environ.get("DF_POS_W", "0") == "1"
_ВЕСА = [1.0, 0.56, 0.34, 0.14, 0.06, 0.025, 0.01, 0.004]   # вероятность дойти

def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    ряд_весь = корпус()
    # ВАЛИДАЦИЯ: последняя десятая часть корпуса не участвует в обучении. Без неё падение
    # loss ничего не говорит -- ровно это и случилось в §95: loss упал втрое, а приёмка нет,
    # и отличить переобучение от неверной нормировки было НЕЧЕМ.
    _гр = int(len(ряд_весь) * 0.9)
    ряд, ряд_вал = ряд_весь[:_гр], ряд_весь[_гр:]
    print(f"  пар в корпусе: {len(ряд_весь)} (обучение {len(ряд)}, проверка {len(ряд_вал)})")
    from safetensors.torch import load_file
    веса = load_file(os.path.join(ЧЕРН, "model.safetensors"))
    конф = json.load(open(os.path.join(ЧЕРН, "config.json")))
    # Тело в fp32. В fp16 обучение даёт NaN на первом же шаге: чекпойнт черновика хранится
    # в BF16 (его config: dtype bfloat16), а у bf16 шире порядок -- при переносе в fp16 поток
    # переполняется. Этот же закон записан у нас для целевой сети («bf16-чекпойнт не ложится
    # в fp16»), и здесь он повторился в точности.
    ч = ЧерновикDFlash2(веса, конф, устройство=dev, тип=torch.float32)
    # При SEL=1 учим ТОЛЬКО веса селектора. Первый прогон (1500 шагов) пустил градиент рёбер
    # во ВСЕ 81 тензор -- «совпало» выросло 0->15%, а приёмка в бою УПАЛА 1.88 -> 1.44, и
    # позиция 0 просела 42% -> 35.8%. Причина прямая: целевая функция рёбер не следит за
    # качеством распределения тела, а градиент в тело течёт. Тело замораживаем.
    for и, т in ч.в.items():
        if hasattr(т, "requires_grad_"):
            т.requires_grad_(и.startswith("candidate_selector.") if СЕЛЕКТОР else True)
    учимые = [т for т in ч.в.values() if getattr(т, "requires_grad", False)]
    print(f"  обучаемых тензоров: {len(учимые)}")
    W = tenzor_celi('lm_head.weight', dev)                     # ЗАМОРОЖЕНА: это голова ЦЕЛИ, её не трогаем
    global ВЛОЖЕНИЕ_МАСКИ
    _ключ = "model.language_model.embed_tokens.weight"
    _idx = json.load(open(os.path.join(ЦЕЛЬ, "model.safetensors.index.json")))
    if _ключ not in _idx["weight_map"]:
        _ключ = next(n for n in _idx["weight_map"] if n.endswith("embed_tokens.weight"))
    # Вложение приводим к типу ТЕЛА: голова хранится в fp16 ради памяти, а тело
    # считает в fp32 -- смешивать в одном умножении нельзя.
    ВЛОЖЕНИЕ_МАСКИ = stroka_celi(_ключ, ч.маска_id, dev).to(ч.тип)
    print(f"  вложение маски взято у цели: {_ключ}")
    print(f"  голова цели: {tuple(W.shape)} (заморожена)")
    # SGD вместо AdamW: у Adam два состояния на параметр, и на 81 тензоре это
    # утраивает память под веса. Для дообучения на малом корпусе SGD достаточен.
    opt = torch.optim.SGD(учимые, lr=float(os.environ.get("LR","1e-6")), momentum=0.9)
    блок = ч.блок
    t0 = time.time()
    # БАТЧ: примеры накапливаются и градиент усредняется. С батчем 1 шум таков, что по
    # десяткам шагов ничего не видно -- это и наблюдалось в первом запуске.
    БАТЧ = int(os.environ.get("BATCH", "8"))
    окно_loss, окно_точн = [], []
    for шаг in range(ШАГОВ):
        opt.zero_grad(set_to_none=True)
        сум_loss = 0.0; сум_точн = 0.0
        for _ in range(БАТЧ):
            i = torch.randint(КОНТЕКСТ, len(ряд) - блок - 1, (1,)).item()
            сост = torch.stack([ряд[j][0][-1] for j in range(i - КОНТЕКСТ, i)]).to(dev, ч.тип)
            # ДВЕ СХЕМЫ ОБУЧЕНИЯ. По токенам (метка -- тензор) обучение выходит на плато:
            # один правильный ответ на позицию несёт слишком мало сигнала (записка 25 §94d).
            # По ЛОГИТАМ цели (метка -- кортеж top-k значений и индексов) учим KL-дивергенцией:
            # черновик обязан воспроизводить РАСПРЕДЕЛЕНИЕ цели, а не угадывать один токен.
            дистилляция = isinstance(ряд[i + 1][1], tuple)
            if дистилляция:
                зн = torch.stack([ряд[i + j + 1][1][0][-1] for j in range(блок)]).to(dev, torch.float32)
                ид = torch.stack([ряд[i + j + 1][1][1][-1] for j in range(блок)]).to(dev, torch.long)
                метки = None
            else:
                метки = torch.tensor([ряд[i + j + _СДВИГ_МЕТОК][1].reshape(-1)[0].item() for j in range(блок)],
                                     dtype=torch.long, device=dev)
            поз_к = torch.arange(КОНТЕКСТ, device=dev)
            K, V = ч.kv_контекста(сост, поз_к)
            kv = [(K[l], V[l], поз_к) for l in range(ч.слоёв)]
            вл = ВЛОЖЕНИЕ_МАСКИ[None].repeat(блок, 1)
            поз_б = torch.arange(КОНТЕКСТ, КОНТЕКСТ + блок, device=dev)
            скрытое = ч.проход_блока(вл, поз_б, kv_внеш=kv)
            логиты = (скрытое.to(torch.float16) @ W.T).float()
            # [ОБУЧЕНИЕ СЕЛЕКТОРА -- SEL=1]
            # §95a показал: переобучения нет, но приёмка не растёт. Причина в том, ЧТО мы учим:
            # приёмку решает не близость распределений, а СОВПАЛ ЛИ ВЫБРАННЫЙ токен. Между
            # распределением и выбором стоит `candidate_selector`, и его веса в loss по логитам
            # почти не участвуют. Здесь целевая функция -- рёбра селектора: сам выбор (Виттерби)
            # недифференцируем, но РЁБРА дифференцируемы, и учить надо их.
            if СЕЛЕКТОР:
                from dflash2.otbor import оценить_рёбра
                унар, канд = логиты.topk(16, dim=-1)                       # [L, K]
                h_r = скрытое @ ч.в["candidate_selector.hidden_projection.weight"].T
                як = ид[:, 0][:1] if дистилляция else метки[:1]            # якорь = первый токен
                if os.environ.get("DIAG"):
                    print("  DIAG codebook.rg:", ч.в["candidate_selector.predecessor_codebook"].requires_grad,
                          "hp.rg:", ч.в["candidate_selector.hidden_projection.weight"].requires_grad,
                          "hidden.rg:", скрытое.requires_grad, "h_r.gf:", h_r.grad_fn is not None)
                рёбра = оценить_рёбра(ч.в["candidate_selector.predecessor_codebook"],
                                      ч.в["candidate_selector.successor_codebook"],
                                      канд[None], унар[None], h_r[None], як)
                цель_ток = ид[:, 0] if дистилляция else метки               # [L]
                # Правильный класс = позиция целевого токена среди кандидатов; если его там нет,
                # позиция пропускается (учить нечему -- кандидат не предложен вовсе).
                есть = (канд == цель_ток[:, None])
                маска = есть.any(dim=-1)
                if not маска.any():
                    continue
                if True:
                    поз = есть.float().argmax(dim=-1)
                    # Предшественник берётся ИСТИННЫЙ (teacher forcing), а не max по оси p:
                    # в бою путь строится слева направо от уже принятого токена, значит цена
                    # перехода должна учиться при том предшественнике, который цель и выбрала.
                    # Позиция 0 -- особая: у неё предшественник единственный (якорь).
                    p_ид = torch.cat((torch.zeros(1, dtype=torch.long, device=поз.device), поз[:-1]))
                    # Позиция годна, только если и её токен, и предыдущий есть среди кандидатов.
                    маска = маска & torch.cat((torch.ones(1, dtype=torch.bool, device=маска.device),
                                               маска[:-1]))
                    if not маска.any():
                        continue
                    р = рёбра[0].gather(1, p_ид[:, None, None].expand(-1, 1, рёбра.shape[3])).squeeze(1)
                    loss = torch.nn.functional.cross_entropy(р[маска], поз[маска]) / БАТЧ
                    эталон = цель_ток
                    loss.backward()
                    сум_loss += loss.item() * БАТЧ
                    with torch.no_grad():
                        сум_точн += (канд[маска].gather(1, р[маска].argmax(-1)[:, None]).squeeze(1)
                                     == цель_ток[маска]).float().mean().item()
                continue
            if дистилляция:
                # KL по top-k: берём те же позиции словаря, что назвала цель, и сводим
                # распределения. Нормируем цель по своему же срезу -- сумма вне top-k мала.
                наши = torch.gather(логиты, 1, ид)
                loss = torch.nn.functional.kl_div(
                    torch.log_softmax(наши, dim=-1),
                    torch.softmax(зн, dim=-1), reduction="batchmean") / БАТЧ
                эталон = ид[:, 0]
            else:
                if _ВЕС_ПОЗИЦИЙ:
                    _w = torch.tensor(_ВЕСА[:логиты.shape[0]], device=логиты.device,
                                      dtype=логиты.dtype)
                    _w = _w / _w.mean()          # среднее 1: масштаб потери не едет
                    _пот = torch.nn.functional.cross_entropy(логиты, метки, reduction="none")
                    loss = (_пот * _w).mean() / БАТЧ
                else:
                    loss = torch.nn.functional.cross_entropy(логиты, метки) / БАТЧ
                эталон = метки
            loss.backward()
            сум_loss += loss.item() * БАТЧ
            with torch.no_grad():
                _сов = (логиты.argmax(-1) == эталон).float()
                сум_точн += _сов.mean().item()
                # ПО ПОЗИЦИЯМ: среднее по блоку прячет главное. Приёмку решает
                # позиция 0, и если харнесс не воспроизводит боевые ~54 % на ней,
                # то он учит не ту задачу -- и любое обучение в нём бессмысленно.
                _поз = globals().setdefault("_ТОЧН_ПОЗ", [0.0]*8)
                _чис = globals().setdefault("_ЧИСЛО_ПОЗ", [0]*8)
                for _j in range(min(len(_сов), 8)):
                    _поз[_j] += _сов[_j].item(); _чис[_j] += 1
        torch.nn.utils.clip_grad_norm_(учимые, 1.0)
        opt.step()
        окно_loss.append(сум_loss / БАТЧ); окно_точн.append(сум_точн / БАТЧ)
        if шаг % 20 == 0 or шаг == ШАГОВ - 1:
            _п = globals().get("_ТОЧН_ПОЗ"); _ч = globals().get("_ЧИСЛО_ПОЗ")
            if _п and _ч and _ч[0]:
                print("    по позициям: " + "  ".join(
                    f"{_j}={_п[_j]/max(_ч[_j],1)*100:.0f}%" for _j in range(8)), flush=True)
            n = min(len(окно_loss), 20)
            вал = _проверка(ч, W, ряд_вал, блок, dev) if ряд_вал else float("nan")
            print(f"  шаг {шаг:>5}: loss {sum(окно_loss[-n:])/n:.4f}  проверка {вал:.4f}  "
                  f"совпало {sum(окно_точн[-n:])/n*100:.1f}%", flush=True)
    print(f"  {ШАГОВ} шагов за {time.time()-t0:.1f} с")
    # СОХРАНЕНИЕ: веса кладём рядом с исходными, отдельным файлом -- исходный чекпойнт
    # не трогаем, иначе откат станет невозможен.
    путь = os.environ.get("SOHRANIT", "")
    if путь:
        torch.save({и: т.detach().cpu() for и, т in ч.в.items()}, путь)
        print(f"  веса сохранены: {путь}")

if __name__ == "__main__":
    main()
