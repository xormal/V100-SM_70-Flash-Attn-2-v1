# vLLM patches needed to run an int8 (GPTQ W8A16) checkpoint of Qwen3.5/3.6 on `sm_70`

Two small patches against this 1Cat-vLLM fork. Both are needed; with either missing the server still
**boots and serves**, and emits pure garbage (`!!!!!!`) — which is why each one below records the
symptom, not just the change.

Apply from the site-packages `vllm/` directory:

```bash
cd <venv>/lib/python3.12/site-packages/vllm
patch -p0 < 01-qwen3_5-gptq-load.patch
patch -p0 < 02-gptq-modules-to-not-convert.patch
```
(The originals ship with CRLF line endings; the patches are generated with `--strip-trailing-cr`, and
applying them leaves the files LF. That is cosmetic.)

## 01 — `qwen3_5.py`: a quantised parameter with no output dim

`load_weights` splits a tuple-shard (`in_proj_qkvz` is fed by `in_proj_qkv` + `in_proj_z`) along
`param.output_dim`. GPTQ's `g_idx` is one value per **input** row, so it has no output dim and is
shared by every output shard — it must be loaded whole. Without this the load dies on
`AttributeError: 'RowvLLMParameter' object has no attribute 'output_dim'`, raised from inside a
`logger.debug()` argument, so it fails even with debug logging off.

## 02 — `gptq.py`: carry `modules_to_not_convert` through

The AWQ config exposes `modules_to_not_convert`; the GPTQ config dropped it. That attribute is not
used to decide what to quantise (`dynamic` does that) — but **models read it**.
`qwen3_5.py::_uses_split_gdn_input_projections()` asks whether `linear_attn.in_proj_a/b` were left in
fp16, and builds either SPLIT `in_proj_qkvz` + `in_proj_ba` modules, or one fused BF16-style
`in_proj_qkvz`. With the attribute absent, GPTQ always looked "unsplit", so the model expected a
single fused `in_proj_qkvz.weight` that a split quantised checkpoint does not contain. The only
outward sign was one warning line per GDN layer:

```
Parameter layers.N.linear_attn.in_proj_qkvz.weight not found in params_dict, skip loading
```

and then 47 of the 48 GDN layers ran on **uninitialised** weights.

## Checkpoint side

The checkpoint's `quantization_config` must carry BOTH:
* `dynamic` with `-:<regex>` skips naming the **fused** module names as vLLM sees them
  (`self_attn.qkv_proj`, `linear_attn.in_proj_ba`) — naming the source modules (`q_proj`, `in_proj_a`)
  never matches, because vLLM fuses before asking whether a layer is quantised;
* `modules_to_not_convert` in the AWQ spelling (source module names), which is what patch 02 forwards
  to the model.

See `../quantize/` for the converter.

## 03 — `v1/attention/backends/flash_attn_v100.py`: int8-склад ТОЛЬКО там, где внимание наше

`do_kv_cache_update` направляет запись в наш CUDA-склад по одному условию — `kv_cache_dtype`
начинается с `fp8`. А вот **внимание** идёт по нашему пути по ДРУГОМУ, более узкому условию:
`_supports_flash_v100_path()` требует, среди прочего, `sliding_window == (-1, -1)`.

Пока модель без скользящего окна (Qwen3.5/3.6) условия совпадают и разницы нет. На **Gemma-4** они
расходятся: 40 слоёв из 48 — скользящие, их внимание откатывается на Triton, а склад по-прежнему наш.
При `FA2SM70_KV_I8=1` это означает, что в 40 слоёв лягут **int8-байты с масштабом на позицию**, а
прочитает их Triton **как e4m3**. Симптом — не «чуть хуже точность», а мусор на выходе, причём
модель поднимется и будет отвечать.

Патч передаёт складу `allow_i8=self._supports_flash_v100_path()`. Тогда int8 стоит ровно там, где все
читатели наши, а слои со скользящим окном остаются e4m3 — их и пишет наше же ядро, потому что Triton
на Volta e4m3 не умеет писать вовсе. Пул при этом СМЕШАННЫЙ, и это уже поддержано: шим отмечает
формат на каждый кэш (`_FMT[ck]`), а сборка, префилл и декод читают именно отметку.

Правило, которое стоит за патчем: **приватный формат законен ровно до тех пор, пока каждый его
читатель наш.** Появился один чужой — формат обязан вернуться к общему.

## 05 — mamba/align: НЕ брать блок, закэшированный ДРУГОЙ заявкой В ТОМ ЖЕ ШАГЕ

Чинит боевой отказ «забор из восклицаний»: ответ на 100% состоит из `!` (токен 0) на всю длину
`max_tokens`, с первого же токена и без восстановления. Захвачено 6 случаев на боевом за день.

**Регистр.** У гибридной модели (16 слоёв внимания + 48 gated-delta-net) префикс-кэш принудительно
идёт в режиме `align`: `Qwen3_5ForConditionalGeneration` не объявляет `supports_mamba_prefix_caching`,
поэтому и `--mamba-cache-mode none`, и `all` схлопываются в `align`
(`model_executor/models/config.py`), а апстрим печатает про него EXPERIMENTAL. Отключить режим
через флаг НЕЛЬЗЯ — только вместе с префикс-кэшем.

**Корень.** Блок публикуется в хэш-карту префикс-кэша в `KVCacheManager.allocate_slots` —
то есть в `schedule()`, ДО `execute_model`, на токены, которые ещё не посчитаны. Для внимания это
безопасно: внутри слоя `reshape_and_cache` пишет KV всей партии до того, как ядро внимания её читает.
Для mamba — нет: рекуррентное состояние пишется только по итогам прохода, а читается ДО прохода
(`gpu_model_runner` -> `mamba_utils.preprocess_mamba`, индекс `(num_computed_tokens-1)//block_size`).
Значит вторая заявка с тем же префиксом, попавшая в тот же шаг планировщика, копирует себе
состояние из блока, который первая только что опубликовала и ещё НЕ заполнила. Внимание у неё
корректное, а состояние 48 слоёв — чужой мусор.

Межшаговый вариант этой же гонки апстрим уже закрыл (при `mamba_cache_mode != "none"` принудительно
выключается async scheduling, `config/vllm.py`). Внутришаговый — не закрыт.

**Лечение** (взято из более новой ветки vLLM, `1Cat-vLLM-fa2-work`,
`v1/core/single_type_kv_cache_manager.py`): `MambaManager` помнит хэши блоков, закэшированных в
текущем шаге, и если попадание опирается на такой блок — возвращает `num_gpu_blocks + 1` из
`get_num_blocks_to_allocate`. Планировщик считает, что места нет, и берёт заявку СЛЕДУЮЩИМ шагом,
когда состояние уже записано. Патч добавляет хук `new_step_starts()` по цепочке
scheduler -> KVCacheManager -> coordinator -> типовые менеджеры, чтобы чистить учёт на каждом шаге.

**Замер (зеркало на картах 0-1, пара БАЙТ-В-БАЙТ одинаковых заявок одновременно, холодный кэш):**

| | было | стало |
|---|---|---|
| промпт из захвата 12:10 | 18/20 пар | **0/20** |
| промпт из захвата 14:31 | 18/20 пар | **0/20** |

Префикс-кэш продолжает работать: доля попаданий 64.5%, повтор заявки 11.8 с -> 1.3 с. То есть
исправление ОТКЛАДЫВАЕТ заявку на один шаг, а не отключает кэш.

Контроли, снятые до исправления (каждый — отдельный подъём, факт проверялся поведением, а не
баннером): у каждой заявки своя соль (общего префикса нет) 0/15; префикс-кэш выключен 0/15;
`--enforce-eager` 0/10; наш скан GDN выключен 5/15 (то есть дефект НЕ наш); плита gptq `fold`
выключена 10/20; `USE_SHIM=0` + fp16-пул, ни одного нашего ядра — 13/20 и 18/20. Последнее и
доказывает, что корень в vLLM, а не в наших ядрах.
