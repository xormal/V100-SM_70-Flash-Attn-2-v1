# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ЧАНКОВЫЙ СКАН GATED-DELTA-NET НА НАШИХ ЯДРАХ -- ЗАМЕНА fla.chunk_gated_delta_rule.
#
# Подпись и семантика ДОСЛОВНО повторяют
# vllm/model_executor/layers/fla/ops/chunk.py::chunk_gated_delta_rule, потому что подмена делается
# на уровне ИМЕНИ в модуле-потребителе (см. integrations/vllm_sm70/fa2_gdn_patch), а не правкой
# исходников vLLM. Любое расхождение в подписи здесь -- это отказ на боевом пути, а не тест.
#
# ЧТО ОСТАЁТСЯ ОТ TRITON И ПОЧЕМУ. Из шести шагов цепочки заменяются пять. Первый --
# l2norm + chunk_local_cumsum -- НЕ ЗАМЕНЯЕТСЯ: он идёт 0.195 мс на слой, то есть 0.8% скана,
# и на нём нет ни одного умножения матриц, ради которого существуют тензорные ядра. Замена дала бы
# в лучшем случае десятые доли процента, а рисковала бы точностью нормировки, от которой зависит
# ВСЁ остальное. Правило "сперва алгоритм, потом примитивы": оптимизируется цепочка, а не свободные
# слоты в ней.
#
# ГРАНИЦЫ ПРИМЕНИМОСТИ -- ЯВНЫЕ, И ПРИ ВЫХОДЕ ЗА НИХ ВЫЗОВ УХОДИТ К TRITON, А НЕ ПАДАЕТ.
# Ядра инстанцированы под BT=64 и головы 128, потому что это единственная живая форма Qwen3.5.
# Всё, что вне: другой head_dim, batch>1 без cu_seqlens, отсутствие initial_state, fp32-вход.
# Это НЕ заглушка на будущее: без такого отката один неучтённый режим (например, профилировочный
# прогон vLLM с пустыми метаданными) ронял бы сервер на старте.
import os

import torch

from ._ext import gdn_ext

BT = 64                      # размер чанка; на него рассчитаны все пять ядер
_HEAD_DIM = 128


# ------------------------------------------------------------------------------------------------
# Индексы чанков. Форма ДОСЛОВНО как у fla (ops/index.py), но в int32: наши ядра читают их
# скалярной загрузкой на блок, и int64 удвоил бы этот трафик ни за что.
# Кэш -- по (адрес данных, длина, устройство). Ключом НЕЛЬЗЯ брать сам тензор: cu_seqlens
# пересоздаётся каждый шаг, и кэш по объекту не попадал бы ни разу.
# ------------------------------------------------------------------------------------------------
# Кэш ПО ТОЖДЕСТВУ ОБЪЕКТА, а не по содержимому -- ровно как @tensor_cache у fla (utils.py:32,
# сравнение `a is b`). Два соображения, и оба обязательные:
#   * КОРРЕКТНОСТЬ. Ключ по data_ptr НЕЛЬЗЯ: vLLM переиспользует постоянные буферы под
#     query_start_loc, и по одному адресу в следующем шаге лежат ДРУГИЕ границы. Ключ по
#     тождеству + СИЛЬНАЯ ссылка в записи кэша делает переиспользование id невозможным пока
#     запись жива.
#   * СКОРОСТЬ. Построение индексов требует значений на хосте, то есть СИНХРОНИЗАЦИИ с GPU.
#     Скан вызывается 48 раз за проход с ОДНИМ И ТЕМ ЖЕ объектом cu_seqlens -- без кэша это 48
#     синхронизаций на проход. У fla ровно та же конструкция ровно по той же причине.
_idx_cache: list = []
_IDX_CACHE_MAX = 8


def _chunk_meta(cu_seqlens: torch.Tensor):
    """-> (cu_i32, chunk_indices_i32 [NT,2], chunk_offsets_i32 [N+1], NT)

    Индексы берутся ИЗ fla (prepare_chunk_indices / prepare_chunk_offsets), а не строятся заново.
    Это не лень: любое расхождение в раскладке чанков дало бы ответ, отличающийся от образца по
    причине, не имеющей отношения к ядрам, и искать её пришлось бы в двух местах сразу.
    Наше здесь -- только приведение к int32: ядра читают эти индексы скалярной загрузкой на блок,
    и int64 удвоил бы этот трафик ни за что.
    """
    for i, (ref, out) in enumerate(_idx_cache):
        if ref is cu_seqlens:
            if i:                                   # LRU: свежий -- в конец
                _idx_cache.append(_idx_cache.pop(i))
            return out
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    ci = prepare_chunk_indices(cu_seqlens, BT).to(torch.int32).contiguous()
    co = prepare_chunk_offsets(cu_seqlens, BT).to(torch.int32).contiguous()
    cu32 = cu_seqlens.to(torch.int32).contiguous()
    out = (cu32, ci, co, int(ci.shape[0]))
    _idx_cache.append((cu_seqlens, out))
    while len(_idx_cache) > _IDX_CACHE_MAX:
        _idx_cache.pop(0)
    return out


def supported(q, k, v, g, beta, initial_state, cu_seqlens) -> bool:
    """Ровно те условия, при которых инстанцированы ядра. Проверяется ДО любой работы."""
    if cu_seqlens is None or initial_state is None:
        return False
    # Развёртка бывает и fp16, и bf16 (этот сервер поднимается с --dtype float16). Обе годны:
    # при fp16 конверсия входа не делается ВООБЩЕ, при bf16 -- слитым проходом внутри расширения.
    if q.dtype != k.dtype or v.dtype != q.dtype or beta.dtype != v.dtype:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if q.shape[0] != 1 or q.shape[-1] != _HEAD_DIM or v.shape[-1] != _HEAD_DIM:
        return False
    H, Hg = v.shape[-2], q.shape[-2]
    if H % Hg != 0 or (H // Hg) not in (1, 2, 3, 4, 6, 8):
        return False
    if g.dtype != torch.float32:
        return False
    # Начальное состояние ОБЯЗАНО быть fp32. У vLLM это значение по умолчанию ("GDN recurrent
    # state needs float32"), но mamba_ssm_cache_dtype его переопределяет -- и тогда наше ядро
    # прочло бы bf16-буфер КАК float, то есть выдало бы мусор БЕЗ ЕДИНОЙ ошибки. Проверяем тип,
    # а не полагаемся на умолчание.
    if initial_state.dtype != torch.float32:
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(q.device)[0] != 7:
        return False
    return True


def chunk_gated_delta_rule_volta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    out_dtype: torch.dtype,
):
    """Все пять ядер за один вызов расширения. q,k -- УЖЕ fp16 (их выдаёт l2norm), g -- fp32."""
    cu32, ci, co, _NT = _chunk_meta(cu_seqlens)
    o, ht = gdn_ext().gdn_chunk_scan(
        q, k, v.contiguous(), g.contiguous(), beta.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
        cu32, ci, co, float(scale), bool(out_dtype == torch.bfloat16),
        bool(output_final_state))
    # Контракт образца: при output_final_state=False возвращается None, а не пустой тензор.
    return o, (ht if output_final_state else None)


# ------------------------------------------------------------------------------------------------
# ПОШАГОВАЯ СВЕРКА. Держится РЯДОМ с боевой функцией намеренно: когда цепочка из пяти ядер разойдётся
# с образцом, вопрос "на каком шаге" обязан отвечаться одной командой, а не бисекцией по коду.
# ------------------------------------------------------------------------------------------------
def steps(q, k, v, g, beta, initial_state, cu_seqlens, scale, out_bf16=False):
    """Возвращает словарь всех промежуточных величин цепочки (A, Ai, w, u, h, v_new, ht, o)."""
    ext = gdn_ext()
    cu32, ci, co, NT = _chunk_meta(cu_seqlens)
    H = v.shape[-2]
    v16 = v.to(torch.float16).contiguous()
    b16 = beta.to(torch.float16).contiguous()
    A = ext.gdn_kkt(k, b16, g, cu32, ci, H)
    Ai = ext.gdn_solve_tril(A, cu32, ci)
    w, u = ext.gdn_wu(k, v16, b16, g, Ai, cu32, ci)
    h, v_new, ht = ext.gdn_h(k, w, u, g, initial_state, cu32, co, NT, True)
    o = ext.gdn_o(q, k, v_new, h, g, cu32, ci, float(scale), bool(out_bf16))
    return dict(A=A, Ai=Ai, w=w, u=u, h=h, v_new=v_new, ht=ht, o=o)


def enabled() -> bool:
    """Выключатель для A/B. Гейт врезки проверяется ПОВЕДЕНИЕМ, а не этой строкой."""
    return os.environ.get("FA2SM70_GDN", "1") != "0"
