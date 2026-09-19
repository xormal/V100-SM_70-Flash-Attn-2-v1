# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec



# [FA2/SM70 25.08] ФАЗОМЕР СТРОИТЕЛЯ (FA2SM70_GDN_PHASE=N). Замер показал: этот строитель стоит
# 9.9 мс НА ШАГ при спекуляции -- больше, чем всё остальное вне прохода. Надо назвать УЧАСТОК.
import os as _os
import time as _time

_ГФ = {}
_ГС = [0]


def _гф(имя, т0):
    if not int(_os.environ.get("FA2SM70_GDN_PHASE", "0")):
        return
    _ГФ[имя] = _ГФ.get(имя, 0.0) + (_time.perf_counter() - т0) * 1e3


def _гпечать():
    н = int(_os.environ.get("FA2SM70_GDN_PHASE", "0"))
    if not н:
        return
    _ГС[0] += 1
    if _ГС[0] % н == 0:
        c = _ГС[0]
        стр = "  ".join(f"{и}={в / c:.2f}" for и, в in sorted(_ГФ.items(), key=lambda x: -x[1]))
        print(f"[gdn ФАЗЫ СТРОИТЕЛЯ, мс/вызов, {c}] {стр}", flush=True)

class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # --- РЕЖИМ 'all' (префикс-кэш ДЛЯ РЕКУРРЕНТНЫХ СЛОЁВ), задача 194 ---------------------
    # В режимах 'none'/'align' на запрос приходится ОДНО состояние: индекс одномерный, и
    # восстановить состояние в середине последовательности нечем -- поэтому со спекуляцией
    # 'align' портит выход (проверено гейтом тождественности 19.08, research/23).
    # В режиме 'all' состояние сохраняется НА КАЖДОЙ ГРАНИЦЕ БЛОКА, поэтому:
    #   * non_spec_state_indices_tensor становится ДВУМЕРНЫМ (запрос x блоки);
    #   * block_idx_* -- указатели В ЭТУ таблицу: откуда взять начальное состояние
    #     (последний ПОСЧИТАННЫЙ токен) и куда положить промежуточные и финальное.
    # Механизм и имена -- ровно как у Mamba2 (mamba_attn.py), чтобы два тела не разошлись.
    # ВСЕ ЧЕТЫРЕ -- ПО НЕ-СПЕКУЛЯТИВНЫМ ЗАПРОСАМ ЦЕЛИКОМ (сперва декоды, затем префиллы), а
    # не по одним префиллам, как у Mamba2. Причина в устройстве GDN: когда в батче есть хоть
    # один префилл, чанковый скан обрабатывает ВЕСЬ не-спекулятивный батч разом, и декодные
    # запросы идут тем же вызовом. Срез «только префиллы» дал бы им чужие блоки -- отказа не
    # будет, будет тихая порча.
    # [FA2/SM70 25.08] Длины нужны, чтобы пересобрать индексы состояний под таблицу блоков ДРУГОЙ
    # группы (см. update_block_table): у Mamba2 они в метаданных есть, у GDN их не было.
    seq_lens_для_обновления: torch.Tensor | None = None
    block_idx_last_computed_token: torch.Tensor | None = None
    block_idx_last_scheduled_token: torch.Tensor | None = None
    block_idx_first_scheduled_token: torch.Tensor | None = None
    num_computed_tokens_ns: torch.Tensor | None = None
    mamba_block_size: int = 0
    # Копии на стороне процессора -- ТОЛЬКО для префилла: чтобы разрезать вызов скана по
    # границам блоков, нужны сами числа, а не тензоры на карте. Стоят одной синхронизации на
    # шаг префилла (десятки миллисекунд работы) и НЕ трогаются в декоде, где синхронизация
    # ломала бы полный граф.
    num_computed_tokens_ns_cpu: torch.Tensor | None = None
    non_spec_query_start_loc_cpu: torch.Tensor | None = None


# [FA2/SM70 25.08] ОБЩИЙ СЧЁТ ГРУПП. У модели ДЕСЯТЬ отдельных KV-групп GDN (по 4-5 слоёв), и
# строитель метаданных зовётся по разу на группу: замерено 6300 вызовов на 640 шагов = ~10 за шаг,
# 9.9 мс -- больше, чем всё остальное вне прохода. При этом входные метаданные у групп РАЗЛИЧАЮТСЯ
# ТОЛЬКО таблицей блоков и слотами: маски спекуляции, счётчики, индексы токенов и начала запросов
# совпадают по построению. Считаем их ОДИН раз на шаг и раздаём; блоко-зависимые тензоры каждая
# группа по-прежнему считает СВОИ (иначе состояние читалось бы из чужих блоков -- тихая порча).
# Ключ -- тождество объектов входа (у групп это буквально один и тот же тензор).
_ОБЩЕЕ = {"ключ": None, "знач": None}


def _общий_ключ(m, num_accepted_tokens):
    """Ключ шага. ВНИМАНИЕ: id() безопасен ТОЛЬКО пока объект жив.

    Python переиспользует адреса после сборки мусора, поэтому новый объект может получить id
    старого -- и кэш отдаст ЧУЖИЕ тензоры. Отказа не будет, будет тихая порча. Лечение: вместе с
    ключом держим ЖИВЫЕ ССЫЛКИ на те самые объекты (`_ОБЩЕЕ["якорь"]`), тогда их адреса не могут
    быть переиспользованы, пока запись в кэше действительна.
    """
    # РАЗЛИЧИТЕЛЬ ШАГА ОБЯЗАТЕЛЕН. Раннер переиспользует ОДНИ И ТЕ ЖЕ буферы каждый шаг, поэтому
    # id() совпадает между шагами, а `num_actual_tokens`/`num_reqs` в установившемся декоде
    # постоянны -- кэш отдал бы УСТАРЕВШИЕ индексы блоков (тихая порча, не отказ). `max_seq_len`
    # растёт на каждом шаге декода и одинаков у всех групп внутри шага -- это и есть нужный ключ.
    return (id(m.query_start_loc), id(m.seq_lens), m.num_actual_tokens,
            m.num_reqs, int(m.max_seq_len), id(num_accepted_tokens))


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        # РЕЖИМ 'all': на запрос приходится не один слот, а СТОЛБЕЦ блоков, поэтому
        # постоянный буфер под CUDA-граф обязан быть двумерным. Если оставить его
        # одномерным и подсовывать графу свежий тензор, граф запомнит ЧУЖОЙ адрес --
        # отказа не будет, будет тихая порча (тот же класс, что мы уже ловили гейтом).
        self.mamba_all = vllm_config.cache_config.mamba_cache_mode == "all"
        self.mamba_max_blocks = (
            cdiv(vllm_config.model_config.max_model_len, kv_cache_spec.block_size)
            if self.mamba_all
            else 1
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.mamba_max_blocks)
            if self.mamba_all
            else (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.block_idx_last_computed_token = torch.zeros(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.block_idx_last_scheduled_token = torch.zeros(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    # [FA2/SM70 25.08] ВЫКЛЮЧЕНО ПО ПРОВАЛУ ГЕЙТА. Механизм написан (см. update_block_table ниже) и
    # даёт правильную форму, но гейт «17*23» вернул «39» вместо «391» -- ТИХАЯ ПОРЧА, а не отказ.
    # Значит пересборка индексов состояний под чужую таблицу расходится с `build` в чём-то, что
    # формой не ловится (подозрение: паддинг постоянных буферов и ветка non_spec). Включать только
    # после ПОБИТОВОЙ сверки метаданных обеих дорог на живом шаге, а не «по виду».
    supports_update_block_table: bool = (
        _os.environ.get("FA2SM70_GDN_UPDATE", "0") == "1"
    )  # включается только вместе со сверкой FA2SM70_META_CHECK

    def _индексы_состояний(
        self,
        block_table_tensor,
        aligned_block_table,
        spec_sequence_masks,
        mamba_all: bool,
        только_спекуляция: bool,
    ):
        """ЕДИНОЕ тело для build и update_block_table. Два тела здесь уже стоили подъёма."""
        if spec_sequence_masks is None:
            spec_state = None
            non_spec_state = (
                block_table_tensor if mamba_all else block_table_tensor[:, 0]
            )
            return spec_state, non_spec_state
        _src = aligned_block_table if mamba_all else block_table_tensor
        if только_спекуляция:
            return _src[:, : self.num_spec + 1], None
        return (
            _src[spec_sequence_masks, : self.num_spec + 1],
            block_table_tensor[~spec_sequence_masks]
            if mamba_all
            else block_table_tensor[~spec_sequence_masks, 0],
        )

    def update_block_table(self, metadata, blk_table, slot_mapping):
        """Пересборка ТОЛЬКО блоко-зависимых полей под таблицу другой группы.

        ЗАЧЕМ. У модели ДЕСЯТЬ KV-групп GDN, и без этого движок звал `build` по разу на группу --
        замерено 10 вызовов за шаг, 9.9 мс (наш строитель внимания -- 0.07). Всё, кроме индексов
        состояний, у групп совпадает: маски, счётчики, индексы токенов, начала запросов и индексы
        блоков (они зависят от ДЛИН, а не от таблицы). Механизм штатный -- ровно так делает
        Mamba2 (`mamba_attn.py`), у GDN он просто не был реализован.
        """
        import copy as _copy

        новое = _copy.copy(metadata)
        режим = self.vllm_config.cache_config.mamba_cache_mode
        mamba_all = режим == "all"
        seq_lens = metadata.seq_lens_для_обновления
        block_table_tensor = mamba_get_block_table_tensor(
            blk_table, seq_lens, self.kv_cache_spec, режим
        )
        aligned = (
            mamba_get_block_table_tensor(blk_table, seq_lens, self.kv_cache_spec, "align")
            if mamba_all
            else None
        )
        только_спек = metadata.num_prefills == 0 and metadata.num_decodes == 0
        spec_state, non_spec_state = self._индексы_состояний(
            block_table_tensor, aligned, metadata.spec_sequence_masks,
            mamba_all, только_спек,
        )
        # ПОСТОЯННЫЕ БУФЕРЫ -- СВОИ У ЭТОЙ ГРУППЫ (её граф захватил именно эти адреса).
        # ЗАКОН, ДОБЫТЫЙ ОШИБКОЙ (25.08): СОВПАДЕНИЕ ЗНАЧЕНИЙ НЕ ЗНАЧИТ ПРАВИЛЬНОСТИ ПРИ ГРАФАХ.
        # Первая редакция копировала только индексы состояний, а остальные поля отдавала ЧУЖИЕ
        # (буферы соседней группы). Побитовая сверка 378 шагов показала НОЛЬ расхождений по
        # значениям -- и всё равно гейт вернул «39» вместо «391»: граф читает АДРЕСА своих
        # буферов, а их никто не заполнил. Поэтому копируем ВСЕ поля, которые заполняет `build`.
        # Условия и добивка PAD_SLOT_ID повторяют ветку `build` ОДИН В ОДИН: разойдись они --
        # получим тихую порчу на паддинге, а не отказ.
        if (
            self.use_full_cuda_graph
            and metadata.num_prefills == 0
            and metadata.num_decodes == 0
            and metadata.num_spec_decodes > 0
            and metadata.num_spec_decodes <= self.decode_cudagraph_max_bs
            and metadata.num_spec_decode_tokens <= self.decode_cudagraph_max_bs
            and spec_state is not None
            and metadata.spec_state_indices_tensor is not None
        ):
            n = metadata.num_spec_decodes
            batch_size = metadata.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[:n].copy_(spec_state[:n], non_blocking=True)
            spec_state = self.spec_state_indices_tensor[:batch_size]
            spec_state[n:].fill_(PAD_SLOT_ID)

            # ОСТАЛЬНЫЕ ПОЛЯ -- В СВОИ БУФЕРЫ, порядок и добивка как в `build`.
            _м = metadata
            if _м.spec_sequence_masks is not None:
                self.spec_sequence_masks[:n].copy_(_м.spec_sequence_masks[:n], non_blocking=True)
                новое.spec_sequence_masks = self.spec_sequence_masks[:batch_size]
                новое.spec_sequence_masks[n:].fill_(False)
            if _м.non_spec_token_indx is not None:
                _к = _м.non_spec_token_indx.size(0)
                self.non_spec_token_indx[:_к].copy_(_м.non_spec_token_indx, non_blocking=True)
                новое.non_spec_token_indx = self.non_spec_token_indx[:_к]
            if _м.spec_token_indx is not None:
                _к = _м.spec_token_indx.size(0)
                self.spec_token_indx[:_к].copy_(_м.spec_token_indx, non_blocking=True)
                новое.spec_token_indx = self.spec_token_indx[:_к]
            if _м.spec_query_start_loc is not None:
                self.spec_query_start_loc[: n + 1].copy_(
                    _м.spec_query_start_loc[: n + 1], non_blocking=True
                )
                _хвост = _м.spec_query_start_loc[n]
                новое.spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
                новое.spec_query_start_loc[n + 1 :].fill_(_хвост)
            if _м.num_accepted_tokens is not None:
                self.num_accepted_tokens[:n].copy_(_м.num_accepted_tokens[:n], non_blocking=True)
                новое.num_accepted_tokens = self.num_accepted_tokens[:batch_size]
                новое.num_accepted_tokens[n:].fill_(1)
        # ВТОРАЯ ВЕТКА БУФЕРОВ -- ЧИСТЫЙ ДЕКОД (без спекуляции). Её пропуск дал ПУСТЫЕ ОТВЕТЫ в
        # конфигурации боевого: значения совпадали, но граф читал незаполненные буферы группы.
        # Зеркалим `build` один в один, включая добивку.
        if (
            self.use_full_cuda_graph
            and metadata.num_prefills == 0
            and metadata.num_spec_decodes == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and non_spec_state is not None
        ):
            _д = metadata.num_decodes
            _bs = (metadata.non_spec_state_indices_tensor.shape[0]
                   if metadata.non_spec_state_indices_tensor is not None else _д)
            self.non_spec_state_indices_tensor[:_д].copy_(non_spec_state[:_д], non_blocking=True)
            non_spec_state = self.non_spec_state_indices_tensor[:_bs]
            non_spec_state[_д:].fill_(PAD_SLOT_ID)

            if mamba_all and metadata.block_idx_last_computed_token is not None:
                self.block_idx_last_computed_token[:_д].copy_(
                    metadata.block_idx_last_computed_token[:_д], non_blocking=True)
                новое.block_idx_last_computed_token = self.block_idx_last_computed_token[:_bs]
                новое.block_idx_last_computed_token[_д:].fill_(0)
                self.block_idx_last_scheduled_token[:_д].copy_(
                    metadata.block_idx_last_scheduled_token[:_д], non_blocking=True)
                новое.block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[:_bs]
                новое.block_idx_last_scheduled_token[_д:].fill_(0)

            if metadata.non_spec_query_start_loc is not None:
                self.non_spec_query_start_loc[: _д + 1].copy_(
                    metadata.non_spec_query_start_loc[: _д + 1], non_blocking=True)
                _хв = metadata.non_spec_query_start_loc[_д]
                новое.non_spec_query_start_loc = self.non_spec_query_start_loc[: _bs + 1]
                новое.non_spec_query_start_loc[_д + 1 :].fill_(_хв)

        новое.spec_state_indices_tensor = spec_state
        новое.non_spec_state_indices_tensor = non_spec_state
        return новое

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        _т0 = _time.perf_counter()
        # ОБЩИЙ КЛЮЧ ШАГА: у всех KV-групп GDN эти объекты -- одни и те же (различаются только
        # таблица блоков и слоты). Тождество объектов и есть признак «тот же шаг, та же группа дел».
        _клч0 = (id(m.query_start_loc), id(m.seq_lens), m.num_actual_tokens, m.num_reqs,
                 int(m.max_seq_len))  # см. пояснение в _общий_ключ: без длины ключ не различает ШАГИ
        _делить = _os.environ.get("FA2SM70_GDN_SHARE", "1") == "1"
        _общ0 = _ОБЩЕЕ.get("знач0") if (_делить and _ОБЩЕЕ.get("ключ0") == _клч0) else None
        if _общ0 is not None:
            context_lens_tensor = _общ0["ctx"]
        else:
            context_lens_tensor = m.compute_num_computed_tokens()
        _гф("0 compute_num_computed_tokens", _т0)
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        mamba_cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        mamba_all = mamba_cache_mode == "all"
        _т1 = _time.perf_counter()
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            mamba_cache_mode,
        )
        _гф("1 таблица блоков (режим)", _т1)
        # РЕЖИМ 'all' (задача 194). Здесь block_table_tensor приходит ПОЛНЫЙ (запрос x блоки),
        # и одномерные потребители (спекуляция, декод) его использовать не могут. Сведённую
        # таблицу берём ТОЙ ЖЕ функцией в режиме 'align': она отдаёт 1+num_spec последних
        # блоков каждого запроса -- ровно то, что нужно и спекуляции, и декоду. Собственного
        # второго выражения для этого НЕ пишем: расхождение двух тел уже стоило нам подъёма.
        aligned_block_table = None
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None
        block_idx_first_scheduled_token = None
        if mamba_all:
            _т2 = _time.perf_counter()
            aligned_block_table = mamba_get_block_table_tensor(
                m.block_table_tensor, m.seq_lens, self.kv_cache_spec, "align"
            )
            _гф("2 таблица блоков (align)", _т2)
            mamba_block_size = self.kv_cache_spec.block_size
            if _общ0 is not None:
                # Индексы блоков зависят ТОЛЬКО от длин (контекст/последовательность), а они у
                # всех групп общие -- считаем один раз на шаг. Блоко-зависимое ниже своё.
                block_idx_last_computed_token = _общ0["b_last_computed"]
                block_idx_first_scheduled_token = _общ0["b_first_sched"]
                block_idx_last_scheduled_token = _общ0["b_last_sched"]
            else:
                # Индексы блоков -- как у Mamba2 (mamba_attn._compute_prefix_caching_block_indices)
                block_idx_last_computed_token = torch.clamp(
                    cdiv(context_lens_tensor, mamba_block_size) - 1, min=0
                )
                block_idx_first_scheduled_token = (
                    cdiv(context_lens_tensor + 1, mamba_block_size) - 1
                )
                block_idx_last_scheduled_token = torch.clamp(
                    cdiv(m.seq_lens, mamba_block_size) - 1, min=0
                )
                if _делить:
                    _ОБЩЕЕ["ключ0"] = _клч0
                    _ОБЩЕЕ["якорь0"] = (m.query_start_loc, m.seq_lens)  # см. _общий_ключ
                    _ОБЩЕЕ["знач0"] = {
                        "ctx": context_lens_tensor,
                        "b_last_computed": block_idx_last_computed_token,
                        "b_first_sched": block_idx_first_scheduled_token,
                        "b_last_sched": block_idx_last_scheduled_token,
                    }
                    _общ0 = _ОБЩЕЕ["знач0"]
            _гф("3 индексы блоков", _т2)

        _тА = _time.perf_counter()
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = spec_sequence_masks_cpu.to(
                    query_start_loc.device, non_blocking=True
                )

        _гф("A маски", _тА)
        _тБ = _time.perf_counter()
        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            # В 'all' слой получает ВСЮ таблицу блоков (он сам возьмёт нужный столбец по
            # block_idx_*); в 'none'/'align' таблица уже сведена к одному блоку на запрос.
            non_spec_state_indices_tensor = (
                block_table_tensor if mamba_all else block_table_tensor[:, 0]
            )
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            non_spec_query_lens = query_lens[~spec_sequence_masks]
            # [FA2/SM70 25.08] СЧЁТ ВЕДЁТСЯ ПО ПРОЦЕССОРНЫМ КОПИЯМ, А НЕ ПО ТЕНЗОРАМ КАРТЫ.
            # Здесь стояли четыре `.item()` по GPU-тензорам, и каждый -- СИНХРОНИЗАЦИЯ с картой.
            # Замер фазомером: `GDNAttentionMetadataBuilder.build` стоил **9.86 мс НА ШАГ** при
            # спекуляции (наш собственный строитель -- 0.07 мс), то есть весь налог на включение
            # спекуляции (+16.7 мс к шагу) сидел здесь, а не в вычислении. Значения БИТ В БИТ те
            # же: `query_lens_cpu` -- это та же разность, посчитанная движком заранее.
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = int((non_spec_query_lens_cpu == 1).sum())
            num_prefills = non_spec_query_lens.size(0) - num_decodes
            num_decode_tokens = num_decodes
            num_prefill_tokens = int(non_spec_query_lens_cpu.sum()) - num_decode_tokens
            num_spec_decode_tokens = (
                int(query_lens_cpu.sum()) - num_prefill_tokens - num_decode_tokens
            )

            _клч = _общий_ключ(m, num_accepted_tokens)
            _общ = _ОБЩЕЕ["знач"] if _ОБЩЕЕ["ключ"] == _клч else None
            if _общ is not None and _os.environ.get("FA2SM70_GDN_SHARE", "1") == "1":
                (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens,
                 num_spec_decode_tokens, spec_token_indx, non_spec_token_indx,
                 spec_query_start_loc, non_spec_query_start_loc,
                 non_spec_query_start_loc_cpu) = _общ
                # БЛОКО-ЗАВИСИМОЕ -- СВОЁ У КАЖДОЙ ГРУППЫ.
                _spec_src = aligned_block_table if mamba_all else block_table_tensor
                if num_prefills == 0 and num_decodes == 0:
                    spec_state_indices_tensor = _spec_src[:, : self.num_spec + 1]
                    non_spec_state_indices_tensor = None
                else:
                    spec_state_indices_tensor = _spec_src[
                        spec_sequence_masks, : self.num_spec + 1
                    ]
                    non_spec_state_indices_tensor = (
                        block_table_tensor[~spec_sequence_masks]
                        if mamba_all
                        else block_table_tensor[~spec_sequence_masks, 0]
                    )
            elif num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    int(query_start_loc_cpu[-1]),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Спекуляция всегда работает СВЕДЁННОЙ таблицей: её токены идут за
                # последним запланированным, и им нужны 1+num_spec ПОСЛЕДНИХ блоков --
                # ровно то, что отдаёт режим 'align' (в 'all' полная таблица не годится:
                # её первые столбцы -- начало последовательности, а не её конец).
                _spec_src = aligned_block_table if mamba_all else block_table_tensor
                spec_state_indices_tensor = _spec_src[:, : self.num_spec + 1]
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                _spec_src = aligned_block_table if mamba_all else block_table_tensor
                spec_state_indices_tensor = _spec_src[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = (
                    block_table_tensor[~spec_sequence_masks]
                    if mamba_all
                    else block_table_tensor[~spec_sequence_masks, 0]
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks], dim=0, out=spec_query_start_loc[1:]
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            if _общ is None:
                _ОБЩЕЕ["ключ"] = _клч
                _ОБЩЕЕ["якорь"] = (m.query_start_loc, m.seq_lens, num_accepted_tokens)
                _ОБЩЕЕ["знач"] = (
                    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens,
                    num_spec_decode_tokens, spec_token_indx, non_spec_token_indx,
                    spec_query_start_loc, non_spec_query_start_loc,
                    non_spec_query_start_loc_cpu,
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            if mamba_all:
                # Те же указатели блоков -- в постоянные буферы, по той же причине.
                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=True
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :batch_size
                ]
                block_idx_last_computed_token[num_decodes:].fill_(0)
                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=True
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :batch_size
                ]
                block_idx_last_scheduled_token[num_decodes:].fill_(0)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        # РЕЖИМ 'all': индексы блоков приводим к тому же ПОДМНОЖЕСТВУ и тому же ПОРЯДКУ, что
        # и non_spec_state_indices_tensor (сперва декоды, затем префиллы), иначе состояние
        # запроса будет прочитано из блока соседа -- отказа не будет, будет тихая порча.
        _гф("Б ветка спекуляции + копии", _тБ)
        _тВ = _time.perf_counter()
        num_computed_tokens_ns = None
        num_computed_tokens_ns_cpu = None
        if mamba_all:
            if spec_sequence_masks is not None:
                # ДЛИНЫ СОГЛАСУЕМ ЯВНО. При ЗАХВАТЕ ГРАФА метаданные приходят ПАДДИРОВАННЫМИ
                # (движок сам это признаёт: `unpadded()` в backend.py с пометкой «drafter still
                # only uses piecewise cudagraphs ... does not want padded metadata»), поэтому
                # маска бывает ДЛИННЕЕ индексов блоков: замерено отказом 22.08 -- «mask [8] does
                # not match indexed tensor [2]», подъём падал целиком. Паддинг всегда в ХВОСТЕ,
                # реальные запросы идут первыми, поэтому срез по длине индексов берёт ровно
                # реальную часть. Если длины совпали -- поведение прежнее.
                if _общ0 is not None and "b_last_computed_m" in _общ0:
                    block_idx_last_computed_token = _общ0["b_last_computed_m"]
                    block_idx_last_scheduled_token = _общ0["b_last_sched_m"]
                    block_idx_first_scheduled_token = _общ0["b_first_sched_m"]
                    num_computed_tokens_ns = _общ0["ns"]
                else:
                    _n = block_idx_last_computed_token.shape[0]
                    _keep = ~(spec_sequence_masks[:_n] if spec_sequence_masks.shape[0] > _n
                              else spec_sequence_masks)
                    block_idx_last_computed_token = block_idx_last_computed_token[_keep]
                    block_idx_last_scheduled_token = block_idx_last_scheduled_token[_keep]
                    block_idx_first_scheduled_token = block_idx_first_scheduled_token[_keep]
                    num_computed_tokens_ns = context_lens_tensor[_keep]
                    if _делить and _общ0 is not None:
                        _общ0["b_last_computed_m"] = block_idx_last_computed_token
                        _общ0["b_last_sched_m"] = block_idx_last_scheduled_token
                        _общ0["b_first_sched_m"] = block_idx_first_scheduled_token
                        _общ0["ns"] = num_computed_tokens_ns
            else:
                num_computed_tokens_ns = context_lens_tensor
            if num_prefills > 0:
                # Синхронизация ТОЛЬКО когда в батче есть префилл (см. поле выше).
                num_computed_tokens_ns_cpu = num_computed_tokens_ns.to("cpu")

        _гф("В индексы блоков (mamba all)", _тВ)
        _тГ = _time.perf_counter()
        attn_metadata = GDNAttentionMetadata(
            seq_lens_для_обновления=m.seq_lens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            num_computed_tokens_ns=num_computed_tokens_ns,
            num_computed_tokens_ns_cpu=num_computed_tokens_ns_cpu,
            non_spec_query_start_loc_cpu=(
                non_spec_query_start_loc_cpu if mamba_all and num_prefills > 0 else None
            ),
            mamba_block_size=(self.kv_cache_spec.block_size if mamba_all else 0),
        )
        _гф("Г сборка объекта", _тГ)
        _гпечать()
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
