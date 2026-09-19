# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ================================================================================================
# СВОЙ DFlash2: СБОРКА ТЕЛА ЧЕРНОВИКА ЦЕЛИКОМ.
# ================================================================================================
#
# ГЛАВНОЕ ОТКРЫТИЕ ПОРТА, ради которого стоило читать эталон, а не статью: **черновик НЕ прогоняет
# контекст через свои слои вообще**. Признаки цели (пять слоёв, слитые через fc) превращаются
# СРАЗУ в K/V всех пяти слоёв черновика одной проекцией и ВПРЫСКИВАЮТСЯ в его KV-кэш. Собственный
# проход черновика идёт ТОЛЬКО по k позициям блока, у которых Q берётся из вложений маск-токенов.
#
# Почему это меняет расклад для нас (а не просто «так у них написано»):
#   * стоимость черновика перестаёт зависеть от длины контекста -- он читает KV, но не строит его;
#   * слитая проекция -- ОДНО умножение [T, 5120] x [5120, 5*2*1024] вместо десяти мелких, то есть
#     ровно та амортизация M, которую мы уже замерили на префилле ([[fa2-sm70-chank-amortiziruet-dekvant]]);
#   * впрыск идёт через ту же точку `do_kv_cache_update`, что и у нас в бэкенде -- значит стык НАШ,
#     переходников не нужно;
#   * скользящее окно 2048 у черновика означает, что при 250K контекста он читает 2048 ключей, а не
#     250K. Это прямо бьёт в «каждая позиция спекуляции стоит 30 мс» -- у DFlash2 её цена НЕ РАСТЁТ.
#
# ПОРЯДОК ОДНОГО ХОДА (собран здесь):
#   1. признаки цели [T, 5*5120] --fc--> [T, 5120]
#   2. hidden_norm -> слитая KV-проекция 5 слоёв -> k_norm по голове -> RoPE -> в KV-кэш (ВПРЫСК)
#   3. вход блока: вложения [маска]*k (масштаб input_embedding_scale), позиции = хвост+1..+k
#   4. пять слоёв: норма -> свёртка.подготовить -> ВНИМАНИЕ(окно, блок двунаправлен) ->
#      свёртка.завершить -> норма_после -> свёртка.подготовить -> MLP -> свёртка.завершить
#   5. norm -> lm_head -> top_k=16 кандидатов на позицию
#   6. ОТБОР связного пути (Виттерби по низкоранговым рёбрам) -> k токенов черновика
# ================================================================================================
from __future__ import annotations

import torch

from .otbor import отобрать
from .sloy import ПараСвёрток, внимание_черновика, мlp, норма_rms


class ЧерновикDFlash2:
    """Тело черновика на общих операциях: эталон СВОЕЙ реализации.

    Здесь намеренно нет ни одного вызова наших ядер -- этот файл служит ЭТАЛОНОМ, против которого
    гейтится нативная версия. Наш же порядок работ («корректно, потом быстро») запрещает менять
    обе стороны сразу: иначе расхождение не отличить от ускорения.
    """

    def __init__(self, веса: dict, конфиг: dict, устройство="cpu", тип=torch.float32):
        d = конфиг["dflash_config"]
        self.к = конфиг
        self.блок = int(d["block_size"])
        self.маска_id = int(d["mask_token_id"])
        self.окно = int(конфиг["sliding_window"])
        self.слоёв = int(конфиг["num_hidden_layers"])
        self.голов = int(конфиг["num_attention_heads"])
        self.kv_голов = int(конфиг["num_key_value_heads"])
        self.hd = int(конфиг["head_dim"])
        self.eps = float(конфиг["rms_norm_eps"])
        self.масштаб_вложений = float(d.get("input_embedding_scale", 1.0))
        self.групп = конфиг["hidden_size"] // int(d["conv_group_size"])
        self.размер_группы = int(d["conv_group_size"])
        self.тактов = int(d["conv_kernel_size"])
        self.в = {и: т.to(device=устройство, dtype=тип) if hasattr(т, "to") else т
                  for и, т in веса.items()}
        self.устройство, self.тип = устройство, тип
        θ = float(конфиг["rope_parameters"]["rope_theta"])
        половина = self.hd // 2
        self.инв = 1.0 / (θ ** (torch.arange(0, половина, device=устройство).to(тип) / половина))

    # -- шаг 1-2: признаки цели -> K/V всех слоёв (ВПРЫСК) ------------------------------------
    def слить_признаки(self, признаки: torch.Tensor) -> torch.Tensor:
        """[T, 5*5120] -> [T, 5120]. Одно умножение, как в эталоне."""
        return признаки @ self.в["fc.weight"].T

    def kv_контекста(self, состояния: torch.Tensor, позиции: torch.Tensor):
        """Слитая проекция K/V ВСЕХ слоёв разом: [T,5120] -> ([L,T,nkv,hd], [L,T,nkv,hd]).

        Слитость -- не украшение: одно умножение с M=T против 2L мелких. На нашей машине
        линейный слой связан ЧТЕНИЕМ ниже M=128, поэтому склейка по выходу -- бесплатный выигрыш.
        """
        T = состояния.shape[0]
        x = норма_rms(состояния, self.в["hidden_norm.weight"], self.eps)
        части = []
        for i in range(self.слоёв):
            части.append(self.в[f"layers.{i}.self_attn.k_proj.weight"])
            части.append(self.в[f"layers.{i}.self_attn.v_proj.weight"])
        W = torch.cat(части, dim=0)                       # [L*2*kv, H]
        плоско = x @ W.T                                   # ОДНО умножение
        kv = плоско.view(T, self.слоёв, 2, self.kv_голов, self.hd).permute(2, 1, 0, 3, 4)
        K, V = kv[0].contiguous(), kv[1].contiguous()      # [L, T, nkv, hd]
        # k_norm ПО ГОЛОВЕ, вес свой на каждом слое.
        # СОБИРАЕМ СПИСКОМ, а не пишем в срез: присваивание `K[i] = ...` -- операция НА МЕСТЕ,
        # и autograd на ней рвётся («variable needed for gradient has been modified by an
        # inplace operation»). Для вывода это было безразлично, для ДООБУЧЕНИЯ -- нет.
        # Значение то же до бита.
        K = torch.stack([норма_rms(K[i], self.в[f"layers.{i}.self_attn.k_norm.weight"], self.eps)
                         for i in range(self.слоёв)])
        угол = позиции.to(self.тип)[:, None] * self.инв[None, :]
        c, s = угол.cos()[None, :, None, :], угол.sin()[None, :, None, :]
        # [RoPE ПО-NEOX, 08.09] Бой крутит `get_rope(is_neox_style=True)` -- пара измерений
        # (i, i+hd/2). Здесь стояло ЧЕРЕДОВАНИЕ (2i, 2i+1): та же частота, но на других
        # измерениях. Ошибка МОЛЧАЛИВАЯ (о ней предупреждает и сам upstream), а цена -- вся
        # офлайн-приёмка: эталон считал не ту функцию, что бой.
        л, пр = K[..., :K.shape[-1] // 2], K[..., K.shape[-1] // 2:]
        K = torch.cat((л * c - пр * s, л * s + пр * c), dim=-1)
        return K, V

    # -- шаг 3-5: собственный проход ТОЛЬКО по блоку -------------------------------------------
    def проход_блока(self, вложения: torch.Tensor, позиции: torch.Tensor,
                     kv_внеш=None) -> torch.Tensor:
        """kv_внеш -- список по слоям [(K, V, позиции)] или None (прежнее поведение до бита).

        Нужен для ДООБУЧЕНИЯ: в выводе контекст приходит впрыском из признаков цели, и учить
        надо ровно на этом, а не на прогоне контекста самим черновиком.
        """
        x = вложения * self.масштаб_вложений
        for i in range(self.слоёв):
            п = f"layers.{i}."
            пара_вн = ПараСвёрток(self.в[п + "attention_conv.base_kernel"],
                                  self.в[п + "attention_conv.kernel_projection.weight"],
                                  self.блок, self.групп, self.размер_группы, self.тактов)
            пара_мл = ПараСвёрток(self.в[п + "mlp_conv.base_kernel"],
                                  self.в[п + "mlp_conv.kernel_projection.weight"],
                                  self.блок, self.групп, self.размер_группы, self.тактов)
            ост = x
            h = норма_rms(x, self.в[п + "input_layernorm.weight"], self.eps)
            h, коэф = пара_вн.подготовить(h)
            угол = позиции.to(self.тип)[:, None] * self.инв[None, :]
            h = внимание_черновика(h, self.в[п + "self_attn.q_proj.weight"],
                                   self.в[п + "self_attn.k_proj.weight"],
                                   self.в[п + "self_attn.v_proj.weight"],
                                   self.в[п + "self_attn.o_proj.weight"],
                                   self.в[п + "self_attn.q_norm.weight"],
                                   self.в[п + "self_attn.k_norm.weight"],
                                   позиции, self.голов, self.kv_голов, self.hd,
                                   self.окно, self.блок, self.eps, угол.cos(), угол.sin(),
                                   None if kv_внеш is None else kv_внеш[i])
            x = ост + пара_вн.завершить(h, коэф)
            ост = x
            h = норма_rms(x, self.в[п + "post_attention_layernorm.weight"], self.eps)
            h, коэф = пара_мл.подготовить(h)
            h = мlp(h, self.в[п + "mlp.gate_proj.weight"], self.в[п + "mlp.up_proj.weight"],
                    self.в[п + "mlp.down_proj.weight"])
            x = ост + пара_мл.завершить(h, коэф)
        return норма_rms(x, self.в["norm.weight"], self.eps)

    # -- шаг 6: кандидаты и связный путь --------------------------------------------------------
    def выдать_блок(self, скрытое: torch.Tensor, lm_head: torch.Tensor, якорь: torch.Tensor,
                    top_k: int = 16) -> torch.Tensor:
        """[k, 5120] -> [k] токенов черновика, СВЯЗНЫХ между собой."""
        логиты = скрытое @ lm_head.T                        # [k, V]
        унарные, кандидаты = логиты.topk(top_k, dim=-1)      # [k, K]
        h_r = скрытое @ self.в["candidate_selector.hidden_projection.weight"].T
        return отобрать(self.в["candidate_selector.predecessor_codebook"],
                        self.в["candidate_selector.successor_codebook"],
                        кандидаты[None], унарные[None], h_r[None], якорь[None])[0]
