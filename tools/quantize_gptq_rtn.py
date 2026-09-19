"""КВАНТОВАТЕЛЬ BF16 -> int8 В ФОРМАТЕ GPTQ, БЕЗ КАЛИБРОВОЧНОГО НАБОРА.

ЧТО ЭТО И ЧЕМ НЕ ЯВЛЯЕТСЯ. Формат на выходе -- ровно тот, что читает движок (`quant_method: gptq`,
bits=8, group_size=128, sym=false, desc_act=false, checkpoint_format=gptq). АЛГОРИТМ выбора целых --
НЕ GPTQ: здесь округление к ближайшему с подбором обрезки (RTN+clip), без гессиана и без корпуса
калибровки. Это осознанный размен, и он записан здесь, а не спрятан:
  * настоящий GPTQ требует корпуса и часов на картах, а библиотеки его в окружении нет;
  * на ВОСЬМИ разрядах разрыв между RTN и GPTQ мал -- шаг сетки 1/255 размаха группы;
  * цена всё равно ЗАМЕРЯЕТСЯ потом правдоподобием против BF16-оригинала (tools/nll_w12.py),
    то есть вердикт выносит прибор, а не это рассуждение.
Если замер покажет неприемлемое -- escalate к настоящему GPTQ, путь для этого открыт.

ПОДБОР ОБРЕЗКИ СОБРАН, НО ЗАМЕР СКАЗАЛ «НЕ НАДО» -- и это записано, а не выкинуто. Довод был
обычный: голый минмакс отдаёт сетку группы одному выбросу. Проверка на трёх боевых формах (с
намеренно вставленными выбросами x8) дала ошибку, СОВПАДАЮЩУЮ до третьей значащей цифры с обрезкой
и без. Причина считается на бумаге: обрезать выброс до 0.8 размаха стоит ~2.6 sigma^2 на одном весе,
а выигрыш на остальных 127 -- около 0.015 sigma^2. На ВОСЬМИ разрядах шаг уже мельче, чем цена
обрезки; она живёт на трёх-четырёх. Поэтому рычаг есть, умолчание -- ВЫКЛЮЧЕНО (иначе это x5 счёта
на 55 ГБ ради нуля).

УСЛОВНОСТЬ НУЛЕВОЙ ТОЧКИ ПРОВЕРЕНА ЗАМЕРОМ, А НЕ ПРОЧТЕНА (tools/gptq_uslovnost.py):
    в файл кладётся (z - 1), движку объявляется checkpoint_format="gptq"  -> relL2 5.90e-03
    при поле самого квантования 5.84e-03. Неверная пара даёт 2.33e-02 -- это ровно ОДИН квант
    смещения на каждом весе, то есть вдвое грубее самого int8, и оно не падает и не сигналит.

ЧТО НЕ КВАНТУЕТСЯ -- список повторён ДОСЛОВНО за боевым слепком Qwen3.6-27B-INT8, потому что он
проверен эксплуатацией: зрение, MTP, lm_head/embed, весь слой 0, `linear_attn.in_proj_{a,b}` и
`self_attn.{q,k,v}_proj`. Последние остаются BF16 НАМЕРЕННО: их берёт на себя рычаг FA2SM70_W12,
который кладёт их в 12 разрядов на подъёме (втрое точнее BF16 при 0.75 байт).

    python tools/quantize_gptq_rtn.py <исходный> <целевой> [--карта 0] [--без-обрезки]
"""
import argparse
import json
import os
import re
import shutil
import sys

import torch

ИСКЛЮЧЕНИЯ = (
    r"visual", r"^mtp", r"\.mtp", r"^lm_head", r"embed_tokens",
    r"\.layers\.0\.",                       # весь нулевой слой
    r"linear_attn\.in_proj_a", r"linear_attn\.in_proj_b",
    r"self_attn\.q_proj", r"self_attn\.k_proj", r"self_attn\.v_proj",
)
_ИСКЛ = re.compile("|".join(ИСКЛЮЧЕНИЯ))
ДОЛИ = (1.0, 0.95, 0.90, 0.85, 0.80)        # доли размаха для подбора обрезки


def квантовать_gptq(w_nk, бит=8, гр=128, обрезка=True):
    """w_nk: [N, K] BF16/fp32 (как лежит в слепке) -> qweight, qzeros, scales, g_idx, ошибка.

    ГРУППА ИДЁТ ПО ОСИ СВЁРТКИ K -- масштаб выносится из ЧАСТИЧНОЙ суммы группы. GPTQ хранит вес
    k-мажорно, поэтому первым делом транспонируем: [N, K] -> [K, N].
    """
    lvl = (1 << бит) - 1
    w = w_nk.t().contiguous().float()                       # [K, N]
    K, N = w.shape
    г = w.view(K // гр, гр, N)
    мин0, макс0 = г.amin(dim=1), г.amax(dim=1)
    центр = (макс0 + мин0) / 2
    полу = (макс0 - мин0) / 2

    лучш_s = лучш_z = None
    лучш_ош = None
    for доля in (ДОЛИ if обрезка else (1.0,)):
        мин = центр - полу * доля
        макс = центр + полу * доля
        s = ((макс - мин) / lvl).clamp_min(1e-12)
        z = torch.round(-мин / s).clamp(0, lvl)
        q = torch.round(г / s.unsqueeze(1) + z.unsqueeze(1)).clamp(0, lvl)
        ош = ((q - z.unsqueeze(1)) * s.unsqueeze(1) - г).pow(2).sum(dim=1)   # [K/гр, N]
        if лучш_ош is None:
            лучш_s, лучш_z, лучш_ош = s, z, ош
        else:
            м = ош < лучш_ош                                 # ВЫБОР ПОГРУППНЫЙ, а не потензорный
            лучш_s = torch.where(м, s, лучш_s)
            лучш_z = torch.where(м, z, лучш_z)
            лучш_ош = torch.where(м, ош, лучш_ош)
    s, z = лучш_s, лучш_z
    q = torch.round(г / s.unsqueeze(1) + z.unsqueeze(1)).clamp(0, lvl).view(K, N).to(torch.int64)

    отн = ((((q.view(K // гр, гр, N) - z.unsqueeze(1)) * s.unsqueeze(1) - г).norm())
           / г.norm().clamp_min(1e-20)).item()

    qw = torch.zeros(K // 4, N, dtype=torch.int64, device=w.device)
    for i in range(4):                                       # 4 подряд идущих k в одном слове
        qw |= (q[i::4] & 0xFF) << (8 * i)
    zf = (z - 1).clamp(0, lvl).to(torch.int64)               # УСЛОВНОСТЬ v1, проверена замером
    qz = torch.zeros(K // гр, N // 4, dtype=torch.int64, device=w.device)
    for i in range(4):                                       # 4 подряд идущих n в одном слове
        qz |= (zf[:, i::4] & 0xFF) << (8 * i)

    def к32(t):
        return torch.where(t >= (1 << 31), t - (1 << 32), t).to(torch.int32)

    g_idx = (torch.arange(K, dtype=torch.int32) // гр)
    return к32(qw).cpu(), к32(qz).cpu(), s.to(torch.float16).cpu(), g_idx, отн


def годен(имя, форма, гр=128):
    if _ИСКЛ.search(имя) or not имя.endswith(".weight") or len(форма) != 2:
        return False
    N, K = форма
    return K % гр == 0 and K % 4 == 0 and N % 4 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("исходный")
    ap.add_argument("целевой")
    ap.add_argument("--бит", type=int, default=8)
    ap.add_argument("--группа", type=int, default=128)
    ap.add_argument("--карта", default="0")
    ap.add_argument("--обрезка", action="store_true",
                    help="подбор обрезки: замерено, что на 8 разрядах не даёт ничего")
    ap.add_argument("--кусок-гб", type=float, default=4.0)
    a = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    dev = f"cuda:{a.карта}" if torch.cuda.is_available() else "cpu"
    инд = json.load(open(os.path.join(a.исходный, "model.safetensors.index.json")))["weight_map"]
    os.makedirs(a.целевой, exist_ok=True)

    # По ФАЙЛАМ, а не по именам: каждый исходный кусок открывается ровно один раз.
    пофайлам = {}
    for имя, ф in инд.items():
        пофайлам.setdefault(ф, []).append(имя)

    новый_инд, буфер, буфер_байт, кусков = {}, {}, 0, 0
    было = стало = 0
    кв = проп = 0
    худшая = 0.0
    предел = a.кусок_гб * 1e9

    def сброс():
        nonlocal буфер, буфер_байт, кусков
        if not буфер:
            return
        кусков += 1
        имяф = f"model-{кусков:05d}.safetensors"
        save_file(буфер, os.path.join(a.целевой, имяф), metadata={"format": "pt"})
        for k in буфер:
            новый_инд[k] = имяф
        print(f"  [кусок {имяф}: тензоров {len(буфер)}, {буфер_байт/1e9:.2f} ГБ]", flush=True)
        буфер, буфер_байт = {}, 0

    for ф in sorted(пофайлам):
        with safe_open(os.path.join(a.исходный, ф), framework="pt") as f:
            for имя in sorted(пофайлам[ф]):
                t = f.get_tensor(имя)
                было += t.numel() * t.element_size()
                if not годен(имя, tuple(t.shape), a.группа):
                    буфер[имя] = t
                    буфер_байт += t.numel() * t.element_size()
                    стало += t.numel() * t.element_size()
                    проп += 1
                else:
                    qw, qz, sc, gi, отн = квантовать_gptq(
                        t.to(dev), a.бит, a.группа, a.обрезка)
                    осн = имя[:-len(".weight")]
                    for суф, тен in (("qweight", qw), ("qzeros", qz), ("scales", sc),
                                     ("g_idx", gi)):
                        буфер[f"{осн}.{суф}"] = тен
                        б = тен.numel() * тен.element_size()
                        буфер_байт += б
                        стало += б
                    кв += 1
                    худшая = max(худшая, отн)
                    if кв % 40 == 1:
                        print(f"  {осн[:62]:<62s} {str(tuple(t.shape)):>15s} ош {отн:.2e}", flush=True)
                    del qw, qz, sc
                del t
                if буфер_байт >= предел:
                    сброс()
    сброс()

    json.dump({"metadata": {"total_size": стало}, "weight_map": новый_инд},
              open(os.path.join(a.целевой, "model.safetensors.index.json"), "w"), indent=1)

    # --- слепок целиком: без токенизатора и шаблона он не сеть, а мешок чисел -------------------
    for имяф in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "generation_config.json", "chat_template.jinja", "preprocessor_config.json",
                 "video_preprocessor_config.json", "LICENSE", "README.md"):
        и = os.path.join(a.исходный, имяф)
        if os.path.exists(и):
            shutil.copy2(и, os.path.join(a.целевой, имяф))

    cfg = json.load(open(os.path.join(a.исходный, "config.json")))
    cfg["quantization_config"] = {
        "quant_method": "gptq", "bits": a.бит, "group_size": a.группа,
        "desc_act": False, "sym": False, "lm_head": False,
        "checkpoint_format": "gptq",           # v1: в файле лежит (z-1). ПРОВЕРЕНО ЗАМЕРОМ.
        "dynamic": {"-:.*visual.*": {}, "-:.*\\.self_attn\\.qkv_proj$": {},
                    "-:.*\\.linear_attn\\.in_proj_ba$": {}, "-:.*\\.layers\\.0\\..*": {},
                    "-:.*mtp.*": {}, "-:^lm_head$": {}},
        "modules_to_not_convert": [
            "visual", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "model.layers.0.", "model.language_model.layers.0.", "mtp"],
    }
    json.dump(cfg, open(os.path.join(a.целевой, "config.json"), "w"), indent=1,
              ensure_ascii=False)

    print(f"\nквантовано тензоров {кв}, оставлено как есть {проп}")
    print(f"ИТОГО: {было/1e9:.2f} -> {стало/1e9:.2f} ГБ  (x{было/стало:.3f} меньше), "
          f"худшая относительная ошибка веса {худшая:.2e}")
    print(f"записано -> {a.целевой} ({кусков} кусков)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
