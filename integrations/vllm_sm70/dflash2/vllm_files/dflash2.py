# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ==================================================================================================
# [FA2/SM70 25.08] ПРЕДЛАГАТЕЛЬ DFlash2 -- СВОЙ КЛАСС, А НЕ НАСЛЕДНИК EagleProposer.
# ==================================================================================================
# ПОЧЕМУ НЕ НАСЛЕДОВАНИЕ (это не вкусовщина, а устройство алгоритма):
#
#   `EagleProposer.propose` СОДЕРЖИТ ЦИКЛ `for token_index in range(num_speculative_tokens - 1)`,
#   то есть k-1 ДОПОЛНИТЕЛЬНЫХ проходов тела черновика на каждый шаг. Ровно эта последовательность
#   проходов и была названа нашим замером главной статьёй расхода спекуляции: тело дёшево, а
#   проходов k. DFlash2 выдаёт блок ЗА ОДИН проход -- значит наследовать цикл означало бы
#   унаследовать то, ради устранения чего берётся этот метод. Замерено у нас на карте: проход
#   блока 11.6 мс при k=1 и 12.37 мс при k=16, то есть шестнадцатый токен стоит 0.06 мс.
#
#   Второе: у Eagle выдача -- поточечный argmax. У DFlash2 выдача -- СВЯЗНЫЙ ПУТЬ по кандидатам
#   (селектор, точный Виттерби). Это другой последний шаг, и он в цикл не встраивается.
#
# ЧТО ЗДЕСЬ СВОЁ, А ЧТО ВЗЯТО ГОТОВЫМ:
#   своё -- порядок хода, впрыск K/V, единственный проход, отбор пути, дозорные проверки;
#   готовым -- плита базового класса (буферы, загрузка модели, ключи графов) и triton-ядро
#   раскладки входа: это машинерия движка, переписывать её значило бы плодить свои копии чужого.
#
# ПОРЯДОК ХОДА:
#   1. раскладка входа: контекст (позиции+слоты) и запрос (1 бонусный + k маск-токенов);
#   2. слияние признаков цели через `fc` -> `hidden_norm` -> K/V ВСЕХ слоёв -> впрыск в кэш;
#   3. ОДИН проход тела по B*(1+k) позициям (граф допустим: форма постоянная);
#   4. верхушка словаря: top_k кандидатов на каждую из k позиций;
#   5. отбор связного пути (якорь = принятый токен) -> [B, k] токенов черновика.
# ==================================================================================================

import os
from dataclasses import replace

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    next_power_of_2,
)

logger = init_logger(__name__)


class DFlash2Proposer(SpecDecodeBaseProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash2"
        # Скрытые состояния цели в тело НЕ передаются: они уходят в K/V впрыском, а телу на вход
        # идут ВЛОЖЕНИЯ маск-токенов. Это ключевое отличие от eagle, и оно объявляется здесь.
        super().__init__(
            vllm_config, device, pass_hidden_states_to_model=False, runner=runner
        )

        конф = self.dflash_config
        self.маска_id = int(конф["mask_token_id"])
        self.top_k = int(конф["selector_top_k"])
        self.block_size = vllm_config.cache_config.block_size

        self._дозор = int(os.environ.get("FA2SM70_DFLASH2_DEBUG", "0"))
        self._только_блок = os.environ.get("FA2SM70_DFLASH2_ONLYBLOCK", "0") == "1"
        self._фазы = int(os.environ.get("FA2SM70_DFLASH2_PHASE", "0"))
        # ФАЛЬСИФИКАТОР: черновик НЕ считает ничего и отдаёт заглушку. Ответ заведомо мусор,
        # читается ТОЛЬКО время: разность с обычным прогоном = цена проверки целью при q=k+1.
        self._пустышка = os.environ.get("FA2SM70_DFLASH2_FAKE", "0") == "1"
        self._фаза_сумма = {}
        self._фаза_счёт = 0
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        # Запросных позиций на цепочку: бонусный токен + k масок.
        self.позиций_на_запрос = 1 + self.num_speculative_tokens
        self.max_query_tokens = self.max_batch_size * self.позиций_на_запрос
        self.max_padded_query_tokens = max(
            self.max_query_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )

        # Буферы контекста ОТДЕЛЬНЫЕ от запросных: адреса запросных обязаны быть неподвижны, иначе
        # захваченный граф будет читать не оттуда (наш форк уже ловил это на графах декода).
        self._слоты_контекста = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self._позиции_контекста = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self._слоты_запроса = torch.zeros(
            self.max_padded_query_tokens, dtype=torch.int64, device=device
        )
        self.positions = torch.zeros(
            self.max_padded_query_tokens, dtype=torch.int64, device=device
        )
        self.arange = torch.arange(
            self.max_num_tokens + self.max_padded_query_tokens + 1,
            device=device,
            dtype=torch.int32,
        )

        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

        # Внутри блока маска ДВУСТОРОННЯЯ -- иначе блочная выдача теряет смысл. У нашего ядра
        # признак причинности -- параметр, поэтому это берётся без правки ядра.
        self.причинный = not dflash_has_any_non_causal(self.draft_model_config.hf_config)

    @property
    def dflash_config(self):
        return getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}

    def _get_eagle3_use_aux_hidden_state_from_config(self):
        # Признаки берутся с ПЯТИ слоёв цели (target_layer_ids) -- без них `fc` нечем кормить.
        return self.dflash_config.get("use_aux_hidden_state", True)

    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        arch = base.model_config.model_arch_config
        if arch.is_mm_prefix_lm:
            base.model_config.model_arch_config = replace(arch, is_mm_prefix_lm=False)
        return replace(
            base,
            attention_config=replace(
                base.attention_config, use_non_causal=not self.причинный
            ),
        )

    def _warn_if_multimodal(self):
        pass

    def load_model(self, target_model: torch.nn.Module) -> None:
        from vllm.model_executor.models.qwen3_dflash import (
            dflash_target_rope_is_neox_style,
        )

        конф = self.speculative_config.draft_model_config
        assert конф is not None
        # Черновик обязан вращать Q/K ТАК ЖЕ, как его цель: иначе позиции разойдутся тихо, и это
        # проявится не падением, а падением приёмки -- худший вид дефекта.
        neox = dflash_target_rope_is_neox_style(target_model)
        if neox is not None:
            конф.hf_config.is_neox_style = neox
        super().load_model(target_model)

    # ---------------------------------------------------------------------------------------
    # 1. Раскладка входа: контекст и запрос в РАЗНЫЕ буферы
    # ---------------------------------------------------------------------------------------
    def _разложить_вход(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ):
        батч = cad.batch_size()
        контекста = target_token_ids.shape[0]
        запросных = батч * self.позиций_на_запрос

        индексы_выборки = torch.empty(
            батч * self.num_speculative_tokens, dtype=torch.int32, device=self.device
        )
        макс_на_запрос = cad.max_query_len + self.позиций_на_запрос
        БЛОК = min(256, next_power_of_2(макс_на_запрос))
        сетка = (батч, (макс_на_запрос + БЛОК - 1) // БЛОК)
        есть_отказы = num_rejected_tokens_gpu is not None

        copy_and_expand_dflash_inputs_kernel[сетка](
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._позиции_контекста,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._слоты_контекста,
            out_query_slot_mapping_ptr=self._слоты_запроса,
            out_token_indices_ptr=индексы_выборки,
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(num_rejected_tokens_gpu if есть_отказы else 0),
            parallel_drafting_token_id=self.маска_id,
            block_size=self.block_size,
            num_query_per_req=self.позиций_на_запрос,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=контекста,
            BLOCK_SIZE=БЛОК,
            HAS_NUM_REJECTED=есть_отказы,
        )

        длины = cad.seq_lens
        if есть_отказы:
            # В паддированном режиме seq_lens ещё содержит отвергнутые -- вычитаем, иначе внимание
            # увидит хвост, которого в цепочке уже нет (тихая порча, не падение).
            длины = длины - num_rejected_tokens_gpu

        # ПОЛЯ БЕРУТСЯ ИЗ НАШЕГО ФОРКА, А НЕ ИЗ UPSTREAM. У upstream есть `seq_lens_cpu_upper_bound`
        # (верхняя оценка длин на процессоре, чтобы не синхронизироваться с картой); у нас его нет.
        # Выдумывать поле нельзя, а молча пропускать -- значит потерять смысл: вместо него ставим
        # _seq_lens_cpu=None, что и означает «оценки нет, считайте по месту».
        if self._только_блок:
            # ФАЛЬСИФИКАТОР СНЯТИЕМ ФАЗЫ: длины урезаны до самого блока, то есть внимание НЕ читает
            # впрыснутый контекст. Ответ заведомо неверен -- читается только КОНЕЧНОСТЬ чисел:
            # если NaN пропал, он родился на впрыске/чтении кэша, а не в теле.
            длины = torch.zeros_like(длины)
        новый = CommonAttentionMetadata(
            query_start_loc=self.arange[: батч + 1] * self.позиций_на_запрос,
            seq_lens=длины + self.позиций_на_запрос,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: батч + 1]).clone()
                * self.позиций_на_запрос
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            num_reqs=cad.num_reqs,
            num_actual_tokens=запросных,
            max_query_len=self.позиций_на_запрос,
            max_seq_len=cad.max_seq_len + self.позиций_на_запрос,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=self._слоты_запроса[:запросных],
            causal=self.причинный,
        )
        return запросных, контекста, индексы_выборки, новый

    # ---------------------------------------------------------------------------------------
    # 2-5. Один проход + отбор связного пути
    # ---------------------------------------------------------------------------------------
    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata=None,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings=None,
    ) -> torch.Tensor:
        батч = common_attn_metadata.batch_size()
        k = self.num_speculative_tokens
        if self._пустышка:
            return torch.zeros(батч, k, dtype=torch.int32, device=self.device)
        if self._фазы:
            import time as _t

            def _от():
                torch.cuda.synchronize()
                return _t.perf_counter()

            def _до(имя, т0):
                torch.cuda.synchronize()
                self._фаза_сумма[имя] = self._фаза_сумма.get(имя, 0.0) + (_t.perf_counter() - т0) * 1e3
            _т_всё = _от()
        else:
            _от = lambda: 0.0
            _до = lambda имя, т0: None
            _т_всё = 0.0

        # ШАГ 1. Слияние признаков ПЯТИ слоёв цели -> [T, hidden]. Без него `fc` не выполнен, и
        # впрыск пойдёт от неверной размерности -- проверяем ЯВНО, а не полагаемся на форму.
        # Приведение типа НА НАШЕМ СТЫКЕ: у цели остаток (residual) живёт в fp32 -- это её право
        # и её точность, но веса черновика fp16, и `fc` падает на разнотипном умножении. Кастуем
        # ЗДЕСЬ, а не у цели: менять точность чужого пути ради своего черновика -- значит тихо
        # тронуть качество боевой сети. Потери нет: признаки всё равно уходят в fp16-умножение.
        if target_hidden_states.dtype != self.dtype:
            target_hidden_states = target_hidden_states.to(self.dtype)
        _т = _от()
        состояния = self.model.combine_hidden_states(target_hidden_states)
        if self._дозор:
            def _оп(имя, t):
                if t is None: return f"{имя}=нет"
                ф = torch.isfinite(t)
                return (f"{имя}[{tuple(t.shape)} {str(t.dtype).split('.')[-1]}]"
                        f" max={t[ф].abs().max().item() if ф.any() else float('nan'):.4g}"
                        f" неконечных={int((~ф).sum())}")
            print(f"[dflash2 СТАДИИ] {_оп('признаки_цели', target_hidden_states)} | "
                  f"{_оп('после_fc', состояния)}", flush=True)
            # ВЕСА: `base_kernel` и кодбуки заведены как nn.Parameter(torch.empty(...)). Если
            # загрузчик их не заполнил, там НЕИНИЦИАЛИЗИРОВАННАЯ память -- и это даёт ровно NaN,
            # причём молча. Проверяем ЧИСЛОМ, а не наличием имени в дереве.
            сл = self.model.model.layers[0]
            for имя, т in (("attention_conv.base_kernel", сл.attention_conv.base_kernel),
                           ("attention_conv.kernel_projection", сл.attention_conv.kernel_projection.weight),
                           ("mlp_conv.base_kernel", сл.mlp_conv.base_kernel),
                           ("q_proj", сл.self_attn.qkv_proj.weight if hasattr(сл.self_attn, "qkv_proj") else сл.self_attn.q_proj.weight),
                           ("codebook_пред", self.model.model.candidate_selector.predecessor_codebook),
                           ("selector.hidden_projection", self.model.model.candidate_selector.hidden_projection.weight),
                           ("fc", self.model.model.fc.weight)):
                ф = т.isfinite()
                print(f"    ВЕС {имя}: {tuple(т.shape)} max={т[ф].abs().max().item() if ф.any() else float('nan'):.4g}"
                      f" среднее|.|={т[ф].abs().float().mean().item() if ф.any() else float('nan'):.4g}"
                      f" неконечных={int((~ф).sum())}", flush=True)
        _до("слияние fc", _т)
        assert состояния.shape[-1] == self.hidden_size, (
            f"DFlash2: после fc ждали {self.hidden_size}, получили {состояния.shape[-1]}"
        )

        _т = _от()
        запросных, контекста, индексы_выборки, cad = self._разложить_вход(
            target_token_ids, next_token_ids, target_positions,
            common_attn_metadata, num_rejected_tokens_gpu,
        )

        _т = _от()
        _до("раскладка входа", _т)
        строитель = self.attn_metadata_builder or self._get_attention_metadata_builder()
        мета = строитель.build_for_drafting(common_attn_metadata=cad, draft_index=0)
        по_слоям = {имя: мета for имя in self.attn_layer_names}

        _до("метаданные", _т)
        _, поперёк_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=запросных, num_tokens_padded=запросных
        )
        режим_графа, опись = self.cudagraph_dispatcher.dispatch(запросных)
        входных = опись.num_tokens
        if поперёк_dp is not None:
            поперёк_dp[self.dp_rank] = входных

        # ШАГ 2. ВПРЫСК: признаки цели -> K/V всех слоёв -> кэш черновика. Идёт ВНЕ графа: форма
        # контекста меняется от шага к шагу, а запросная часть -- нет, потому граф ей и достаётся.
        _т = _от()
        self.model.model.precompute_and_store_context_kv(
            состояния,
            self._позиции_контекста[:контекста],
            self._слоты_контекста[:контекста],
        )

        if self._дозор:
            # ПОСТРОЧНО: NaN уже во вложениях или родился в теле? Вызов симметричен на всех рангах
            # (внутри вложения идёт коллектив TP), поэтому лишнего барьера не создаёт.
            _вл = self.model.embed_input_ids(self.input_ids[:запросных])
            _ф = _вл.isfinite().all(dim=-1)
            print(f"[dflash2 ВХОД] вложения по строкам конечны={_ф.tolist()} "
                  f"ids={self.input_ids[:запросных].tolist()} "
                  f"max={_вл[_вл.isfinite()].abs().max().item():.4g}", flush=True)

        _до("ВПРЫСК KV", _т)
        # ШАГ 3. ОДИН проход. Никакого цикла по k -- в этом весь метод.
        _т = _от()
        with set_forward_context(
            по_слоям,
            self.vllm_config,
            num_tokens=входных,
            num_tokens_across_dp=поперёк_dp,
            cudagraph_runtime_mode=режим_графа,
            slot_mapping=self._get_slot_mapping(входных, cad.slot_mapping),
            additional_kwargs=self._get_forward_context_additional_kwargs(),
        ):
            выход = self.model(
                input_ids=self.input_ids[:входных],
                positions=self._get_positions(входных),
                inputs_embeds=None,
            )
        _до("ПРОХОД тела", _т)
        скрытое = выход[0] if isinstance(выход, tuple) else выход
        if self._дозор:
            ф = torch.isfinite(скрытое)
            print(f"[dflash2 ВЫХОД] строки конечны={скрытое.isfinite().all(dim=-1).tolist()}", flush=True)
            print(f"[dflash2 СТАДИИ] причинность_меты={getattr(мета, 'causal', 'НЕТ ПОЛЯ')} "
                  f"скрытое_черновика[{tuple(скрытое.shape)}] "
                  f"max={скрытое[ф].abs().max().item() if ф.any() else float('nan'):.4g} "
                  f"неконечных={int((~ф).sum())} из {скрытое.numel()}", flush=True)

        # ШАГ 4. Верхушка словаря на k позициях маски (бонусный токен пропущен -- он уже принят).
        _т = _от()
        выбранные = скрытое[индексы_выборки]                      # [B*k, H]
        унарные, кандидаты = self.model.compute_candidates(выбранные)
        _до("верхушка словаря", _т)
        if унарные is None:  # не последний ранг TP: токенов не отдаём
            return torch.zeros(батч, k, dtype=torch.int32, device=self.device)

        # ШАГ 5. Связный путь. Якорь -- принятый токен: путь пришит к реальности, а не к догадке.
        _т = _от()
        токены = self.model.выбрать_путь(
            выбранные.view(батч, k, -1),
            кандидаты.view(батч, k, self.top_k),
            унарные.view(батч, k, self.top_k),
            next_token_ids.view(батч),
        )
        if self._дозор:
            # ДОЗОР: печатаем ПЕРВЫЕ несколько ходов и умолкаем. Печать в горячем цикле уже вешала
            # у нас сервер (1.3 млн строк за 40 минут), поэтому счётчик, а не флаг.
            self._дозор -= 1
            with torch.no_grad():
                поточечный = кандидаты.view(батч, k, self.top_k)[:, :, 0]
                print(
                    f"[dflash2 ДОЗОР] якорь={next_token_ids.view(-1)[:2].tolist()} "
                    f"путь={токены[0].tolist()} "
                    f"поточечный={поточечный[0].tolist()} "
                    f"унар0={унарные.view(батч, k, self.top_k)[0, 0, :3].tolist()} "
                    f"контекста={контекста} запросных={запросных} "
                    f"вход={self.input_ids[:min(запросных, 10)].tolist()} "
                    f"позиции={self._get_positions(запросных)[:min(запросных, 10)].tolist()}",
                    flush=True,
                )
        _до("ОТБОР пути", _т)
        if self._фазы:
            _до("ВЕСЬ propose", _т_всё)
            self._фаза_счёт += 1
            if self._фаза_счёт % self._фазы == 0:
                н = self._фаза_счёт
                стр = "  ".join(f"{и}={в/н:.2f}" for и, в in sorted(
                    self._фаза_сумма.items(), key=lambda x: -x[1]))
                print(f"[dflash2 ФАЗЫ, мс/вызов, {н} вызовов] {стр}", flush=True)
        return токены.to(torch.int32)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings=None,
    ) -> None:
        """Прогрев/профиль: ОДИН проход, как в бою (иначе профиль соврёт в свою пользу)."""
        from vllm.config.compilation import CUDAGraphMode

        запросных = min(num_tokens, self.max_query_tokens)
        if use_cudagraphs:
            размеры = getattr(self.compilation_config, "cudagraph_capture_sizes", None)
            подходящие = [x for x in sorted(размеры or []) if x >= запросных]
            входных = подходящие[0] if подходящие else запросных
            режим = CUDAGraphMode.PIECEWISE
        else:
            входных, режим = запросных, CUDAGraphMode.NONE

        if (
            self.attn_layer_names
            and slot_mappings is not None
            and next(iter(self.attn_layer_names)) in slot_mappings
        ):
            слоты = self._get_slot_mapping(входных)
        else:
            слоты = slot_mappings or {}

        self.model.model.precompute_and_store_context_kv(
            self.hidden_states[:num_tokens], self._позиции_контекста[:num_tokens]
        )
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=входных,
            num_tokens_across_dp=None,
            cudagraph_runtime_mode=режим,
            slot_mapping=слоты,
        ):
            self.model(
                input_ids=self.input_ids[:входных],
                positions=self._get_positions(входных),
                inputs_embeds=None,
            )
