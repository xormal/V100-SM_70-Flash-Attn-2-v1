# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ================================================================================================
# СВОЙ DFlash2: ТЕЛО БЛОЧНОГО ЧЕРНОВИКА. Пишем сами, а не заплатками поверх чужого.
# ================================================================================================
#
# ПОЧЕМУ СВОЁ, А НЕ ПОРТ. Заплаточный путь дал ЧЕТЫРНАДЦАТЬ ступеней несовместимости подряд, каждая
# порождала следующую; конструкция получалась хрупкой -- любое обновление форка её ломает, а
# владения стыком нет. Поправка владельца: «не упрощай работу, это против доктрины». Своё тело
# ценно ВЛАДЕНИЕМ стыком, а не маржой.
#
# ЧТО ЭТО ЗА АЛГОРИТМ (arXiv 2602.06036). Черновик выдаёт БЛОК из k токенов ЗА ОДИН проход, а не k
# последовательных: T_draft перестаёт зависеть от k. Наш замер назвал именно последовательность
# проходов главной статьёй (14.3 мс НА ПРОХОД), поэтому ход бьёт в цель.
#
# РАСКЛАДКА ВЕСОВ СНЯТА С ФАЙЛА (81 тензор, BF16), а не взята из статьи:
#   fc                       [5120, 25600]   25600 = 5 x 5120: слияние признаков ПЯТИ слоёв цели
#   hidden_norm              [5120]
#   layers.N.self_attn       q [4096,5120] (32 головы x 128), k/v [1024,5120] (8 KV-голов), o [5120,4096]
#   layers.N.self_attn.{q,k}_norm [128]      нормировка ПО ГОЛОВЕ
#   layers.N.mlp             gate/up [17408,5120], down [5120,17408]
#   layers.N.{attention,mlp}_conv  base_kernel [2,2,5120] + kernel_projection [1280,5120]
#                                            -- ДВУХТАКТНАЯ ДИНАМИЧЕСКАЯ СВЁРТКА: ядро не задано
#                                            весами, а ПРЕДСКАЗЫВАЕТСЯ из входа проекцией
#   norm                     [5120]
#   candidate_selector       hidden_projection [256,5120] + два кодбука [248320,256]
#                                            -- оценка пар (предшественник, преемник): по ним
#                                            выбирается СВЯЗНЫЙ путь через кандидатов позиций
# LM head в весах ОТСУТСТВУЕТ -- он общий с целевой моделью (и у нас на него уже есть своё ядро).
#
# ЧТО ЗДЕСЬ СЕЙЧАС: разбор и проверка раскладки весов -- то, без чего тело писать нельзя.
# Гейт этого шага: все 81 тензор опознаны, формы сошлись с конфигом, ничего лишнего не осталось.
# ================================================================================================
from __future__ import annotations

import json
import os


class РаскладкаЧерновика:
    """Опись весов черновика: что есть, какой формы и куда пойдёт.

    Пишется ДО тела намеренно: «сперва алгоритм, потом разложение на примитивы». Пока опись не
    сошлась с конфигом ДО ЕДИНОГО тензора, писать вычисление рано -- иначе несоответствие всплывёт
    внутри ядра, где его уже не отличить от ошибки счёта.
    """

    def __init__(self, путь: str):
        self.путь = путь
        with open(os.path.join(путь, "config.json"), encoding="utf-8") as ф:
            self.конфиг = json.load(ф)
        d = self.конфиг["dflash_config"]
        self.слоёв = self.конфиг["num_hidden_layers"]
        self.скрытая = self.конфиг["hidden_size"]
        self.голов = self.конфиг["num_attention_heads"]
        self.kv_голов = self.конфиг["num_key_value_heads"]
        self.размер_головы = self.конфиг["head_dim"]
        self.промежуточная = self.конфиг["intermediate_size"]
        self.окно = self.конфиг["sliding_window"]
        self.блок = d["block_size"]
        self.ядро_свёртки = d["conv_kernel_size"]
        self.группа_свёртки = d["conv_group_size"]
        self.маска = d["mask_token_id"]
        self.слои_цели = tuple(d["target_layer_ids"])
        self.ранг_селектора = d["selector_rank"]
        self.верхушка_селектора = d["selector_top_k"]
        self.словарь = self.конфиг["vocab_size"]

    def ожидаемые(self) -> dict[str, tuple[int, ...]]:
        """Формы, которых мы ЖДЁМ по конфигу -- прежде чем смотреть в файл."""
        H, I, hd = self.скрытая, self.промежуточная, self.размер_головы
        q, kv = self.голов * hd, self.kv_голов * hd
        ожид = {
            "fc.weight": (H, H * len(self.слои_цели)),
            "hidden_norm.weight": (H,),
            "norm.weight": (H,),
            "candidate_selector.hidden_projection.weight": (self.ранг_селектора, H),
            "candidate_selector.predecessor_codebook": (self.словарь, self.ранг_селектора),
            "candidate_selector.successor_codebook": (self.словарь, self.ранг_селектора),
        }
        for n in range(self.слоёв):
            ожид.update({
                f"layers.{n}.self_attn.q_proj.weight": (q, H),
                f"layers.{n}.self_attn.k_proj.weight": (kv, H),
                f"layers.{n}.self_attn.v_proj.weight": (kv, H),
                f"layers.{n}.self_attn.o_proj.weight": (H, q),
                f"layers.{n}.self_attn.q_norm.weight": (hd,),
                f"layers.{n}.self_attn.k_norm.weight": (hd,),
                f"layers.{n}.mlp.gate_proj.weight": (I, H),
                f"layers.{n}.mlp.up_proj.weight": (I, H),
                f"layers.{n}.mlp.down_proj.weight": (H, I),
                f"layers.{n}.input_layernorm.weight": (H,),
                f"layers.{n}.post_attention_layernorm.weight": (H,),
                f"layers.{n}.attention_conv.base_kernel": (self.ядро_свёртки, self.ядро_свёртки, H),
                f"layers.{n}.mlp_conv.base_kernel": (self.ядро_свёртки, self.ядро_свёртки, H),
                f"layers.{n}.attention_conv.kernel_projection.weight": (None, H),
                f"layers.{n}.mlp_conv.kernel_projection.weight": (None, H),
            })
        return ожид

    def сверить(self) -> tuple[bool, list[str]]:
        """Сверка описи с файлом. Возвращает (сошлось, замечания)."""
        from safetensors import safe_open

        замечания: list[str] = []
        ожид = self.ожидаемые()
        with safe_open(os.path.join(self.путь, "model.safetensors"), framework="pt") as ф:
            есть = {k: tuple(ф.get_slice(k).get_shape()) for k in ф.keys()}
        for имя, форма in ожид.items():
            if имя not in есть:
                замечания.append(f"НЕТ в файле: {имя}")
                continue
            если_форма = есть[имя]
            for i, (ж, ф_) in enumerate(zip(форма, если_форма)):
                if ж is not None and ж != ф_:
                    замечания.append(f"ФОРМА {имя}: ждали {форма}, в файле {если_форма}")
                    break
        лишние = sorted(set(есть) - set(ожид))
        for имя in лишние:
            замечания.append(f"ЛИШНЕЕ (не опознано): {имя} {есть[имя]}")
        return (not замечания), замечания
