#!/usr/bin/env bash
# Launch the abliterated Qwen3.6-27B (AWQ 4-bit) on :8085 using OUR fa2_sm70 attention
# kernels instead of the incumbent flash_attn_v100 package.
#
# HOW THE SUBSTITUTION WORKS
#   vLLM(V1) on sm_70 selects AttentionBackendEnum.FLASH_ATTN_V100, whose backend lazily does
#   `from flash_attn_v100 import ...`. We simply put our shim package FIRST on PYTHONPATH, so
#   that import resolves to  <repo>/integrations/vllm_sm70/flash_attn_v100 .
#   No vLLM source is patched. Unset PYTHONPATH to fall back to the incumbent.
#   (PYTHONSAFEPATH=1 is kept, so the shim MUST come via PYTHONPATH, not the cwd.)
#
# MEASUREMENT DISCIPLINE (read before trusting any number from this script)
#   Two earlier rounds of end-to-end numbers turned out to be measuring the INCUMBENT, not us: once
#   because the kill switches had been lost in a `git checkout` so FA2SM70_SHIM_DECODE=0 was silently
#   ineffective, and once because a launch timed out during prewarm and left an older incumbent
#   server holding the port. So: check the "[fa2_sm70] shim:" banner in the log AND verify output
#   quality BEFORE reading any speed number, and re-run the A/B with USE_SHIM=0 in the same session.
#   The end-to-end table that used to live here is deleted rather than corrected -- it was measured
#   against a mixed path (our prefill, incumbent decode/store) that no longer exists.
#
#   Kernel microbenchmarks are not a substitute: an early decode harness fed CONTIGUOUS caches while
#   the server passes strided unbind(1) views, and the harness therefore did not contain the
#   dominant cost (a 640 MB copy per call, 3.77 ms vs 0.048 ms). Trust the server.
#
# MEASURED A/B (2026-07-25, this script, USE_SHIM=1 vs USE_SHIM=0, unique prompts, TP=2, fp8_e4m3):
#                 prefill tok/s              decode tok/s
#    prompt    ours   incumbent  ratio    ours  incumbent  ratio
#      4119     860       832    1.03     41.8      35.3   1.18
#     16368     800       761    1.05     36.8      30.0   1.23
#     32743     793       541    1.47     26.5      26.2   1.01
#     65456     687       362    1.90     23.7      20.1   1.18
#    131027     562       223    2.52     19.2      13.6   1.41
#   Ahead on BOTH paths at every size; the prefill margin grows with context.
#
#   NOTE ON PREFILL EXPECTATIONS: prefill here is NOT attention-bound below ~100K -- at 4K attention
#   is negligible yet TTFT is 4.9 s, so the model's LINEAR part caps prefill at ~855 tok/s no matter
#   what attention does. A "9677 tok/s" prefill on this model would be 520 TFLOP/s against a 2xV100
#   peak of 250; that number was a prefix-cache hit. Benchmark with UNIQUE prompts only.
#
# KV DTYPE / MEMORY (2x32GB, TP=2, AWQ weights ~10.5 GB/GPU, ~2.5 GB activations):
#   fp16 KV (64 KB/token) -> ~471K tokens of capacity: does NOT reach 512K.
#   8-bit KV (32 KB/token) -> ~942K: 512K fits comfortably.
#   So we default to fp8_e4m3 with the offline-calibrated per-layer scales.
#
# EVERY KV PATH IS NOW OURS -- store, decode, dense prefill AND paged (prefix/chunked) prefill.
# Nothing delegates to the incumbent any more, so a measurement here is unambiguously ours; the
# unsupported cases raise instead of silently falling back. This matters because a launcher that
# died before its start line once left an older incumbent server listening and several numbers were
# attributed to us that were not.
#
# WHY PAGED PREFILL IS NOW ENABLED (VLLM_FLASH_V100_ENABLE_PAGED_PREFILL=1):
#   Without a `flash_attn_prefill_paged` entry the backend gathers the paged KV itself, and its fast
#   CUDA gather is guarded by `key_cache.dtype != uint8`. An fp8_e4m3 pool IS uint8, so 8-bit KV
#   silently took a PYTHON double loop with a .item() sync per 16-token block. Measured cost of that
#   loop, per layer per chunk: 64 ms at 16K context, 203 ms at 65K, 449 ms at 147K -- against 0.23 /
#   0.83 / 1.82 ms for our gather kernel, i.e. ~250x. Ours also emits straight to [Hkv, T, d] so the
#   prefill kernel needs no transpose. Correctness vs a dequantized reference: relL2 3.6e-4 (the
#   e4m3 quantization floor) at 1K/4K/16K/40K context.
# GPUs: 2,3 (GPU 0-1 left free for kernel development).
set -euo pipefail

# РАЗЫМЕНОВЫВАЕМ СИМЛИНК. Этот скрипт вызывают и напрямую из репозитория, и по ссылке из
# $VLLM_ROOT/run (так его зовёт start_tmux_int8_nvlink.sh). Без readlink -f SCRIPT_DIR указывал бы
# на каталог ССЫЛКИ, и FA2SM70_REPO ниже уезжал бы на два уровня вверх от run/, то есть мимо
# репозитория -- подъём падал с "shim missing". Ссылка появилась потому, что копия в run/ отстала
# на пять дней и молча поднимала конфигурацию без сегодняшних фич; ссылка это исключает, но требует
# разыменования здесь.
SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
VLLM_ROOT="${VLLM_ROOT:?set VLLM_ROOT to the vLLM workspace}"
PY="${PYTHON_BIN:-$VLLM_ROOT/.venv-1cat-vllm/bin/python}"
MODEL="${MODEL:-$VLLM_ROOT/Huihui-Qwen3.6-27B-abliterated-AWQ}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$VLLM_ROOT/run/chat_template_qwen36_tools_fixed.jinja}"
GENERATION_CONFIG="${GENERATION_CONFIG:-$VLLM_ROOT/run/generation_config_qwen36_chat_eos_only}"
FA2SM70_REPO="${FA2SM70_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

OVERRIDE_GEN_CFG='{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.05,"repetition_penalty":1.1}'

# KV pool MUST be pinned (the 1Cat-vLLM auto-trim bug collapses KV to ~784 tokens/seq if omitted).
# TP=2 on 32GB cards: AWQ weights ~10.5 GB/GPU, so ~15 GB/GPU of KV is safe.
# fp16 KV for this model is 64 KB/token total (16 full-attn layers x 4 kv-heads x 256 x 2 x 2B),
# i.e. 32 KB/token/GPU at TP=2 -> 15 GB/GPU ~= 490K tokens of capacity.
# ВНИМАНИЕ: умолчание 15 ГиБ рассчитано на веса AWQ 4-бит (~10.5 ГиБ/карту). У INT8-модели веса
# 15.5 ГиБ/карту, и с этим умолчанием ПЕРВЫЙ ЖЕ ПОДЪЁМ ПАДАЕТ В OOM (проверено). Для INT8 задавать
# явно: KV_CACHE_BYTES=6442450944 (6 ГиБ) и MAX_MODEL_LEN=131072 -- на этом снята вся серия A/B
# по скану. Умолчание оставлено под AWQ, потому что MODEL по умолчанию тоже AWQ (строка 65);
# если меняете MODEL на INT8 -- меняйте и это.
KV_CACHE_BYTES="${KV_CACHE_BYTES:-16106127360}"    # 15 GiB per GPU (AWQ!)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"           # 524288 works with VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
# ФРАГМЕНТАЦИЯ ПУЛА -- ОТКАЗ ПО ПАМЯТИ НА ДЛИННОМ ПРОМПТЕ (дефект #169). Замерено 11.08.2026:
# без этого игла на 64000 слов роняет движок с OOM при 2.49 ГиБ "reserved but unallocated";
# с ним -- 6/6 игл, ноль OOM, сервер жив. ИМЯ ВАЖНО: torch 2.9 читает ТОЛЬКО
# PYTORCH_CUDA_ALLOC_CONF (проверено `strings libc10_cuda.so`), новое имя PYTORCH_ALLOC_CONF в
# этой сборке НЕ читается вовсе -- то есть выглядит выставленным и не действует.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# NOTE: assign this with a plain `if unset` test, never `${LIMIT_MM:-{...}}` -- brace expansion ends
# at the first '}' inside the default, which silently appends a stray '}' and yields invalid JSON.
if [ -z "${LIMIT_MM:-}" ]; then LIMIT_MM='{"image":20,"video":0}'; fi
export LIMIT_MM
# e4m3 needs offline-calibrated per-layer scales; this file is the 27B's (our model).
export KV_SCALES_FILE="${KV_SCALES_FILE:-$VLLM_ROOT/run/kv_scales_awq.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
# NVLink (NV2, 2x25.78 GB/s) was fitted between GPU2 and GPU3 on 2026-07-25, so P2P must be ON --
# with it disabled NCCL falls back to host staging and the bridge is simply unused. Measured
# all-reduce on this pair, PCIe-with-P2P-off vs NVLink:
#     10 KB (one decode all-reduce)   0.096 ms -> 0.070 ms
#     44 MB (one 4K-prefill all-reduce) 22.86 ms -> 1.258 ms   (3.5 GB/s -> 65 GB/s, 18x)
# That matters more than any kernel here: the model does 2 all-reduces per layer x 64 layers = 128
# per forward, which was 63% of prefill wall and 40% of decode.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN_V100}"
export VLLM_SM70_ENABLE_LM_HEAD_FASTPATH="${VLLM_SM70_ENABLE_LM_HEAD_FASTPATH:-1}"
# ОБНУЛЕНИЕ C У GPTQ-УМНОЖЕНИЙ: 236 заливок на токен -> ОДИН узел графа (q_gemm.cu).
# Декодная ветка gptq копит в C через atomicAdd по gridDim.z = k/128, поэтому C обязан
# приходить нулевым; но модель делает 236 таких умножений за токен (63 gate_up + 63 down +
# 47 in_proj_qkvz + 47 out_proj + 16 o_proj -- сходится ровно), и каждая заливка это ~1.9 мкс
# ядра + ~0.95 мкс зазора между узлами графа при 2.5-17 КБ полезной работы, т.е. почти чистая
# цена существования запуска. Режим fold отдаёт C декодных умножений из постоянной плиты и
# зануляет её ОДНИМ cudaMemsetAsync на токен.
#
# ЗАМЕРЕНО СКВОЗНЯКОМ 2026-07-30 (карты 0-1, TP=2, gptq int8, три подъёма в одной серии,
# уникальные промпты, ignore_eos, greedy, медиана 9 и 5 повторов; разброс декода 0.008 ток/с):
#              декод 1.6К    декод 52К    префилл 52К
#   default      39.612       34.922        1383.9 ток/с
#   never*       40.603       35.653        1380.4     * ПОТОЛОК ветки, численно НЕВЕРЕН
#   fold         40.460       35.530        1373.0
#   fold даёт x1.0214 / x1.0174 -- это 86% и 83% потолка (0.61 мс из 0.62 мс на токен).
#
# ПЕРЕСЧИТАНО НЕЗАВИСИМОЙ ПРОВЕРКОЙ -- цифры выше верны, но переводить их надо иначе:
#   ВЫИГРЫШ АБСОЛЮТЕН И ОТ КОНТЕКСТА НЕ ЗАВИСИТ: 0.51 +- 0.02 мс на токен (1.6К и 52К дают
#   0.530 и 0.490 мс). В долях БОЕВОГО токена 31.10 мс это 1.64% (x1.0167), а НЕ 2.1%:
#   2.1% -- аналитический потолок ветки (236 x 2.84 мкс), измеренный потолок 0.59-0.62 мс = 1.9-2.0%.
#   ЧАСТОТЫ НЕ БЫЛИ ЗАФИКСИРОВАНЫ, и лог замера это показывает: GPU1 1530 -> 1485 -> 1380 МГц по
#   ходу серии. Заявленный разброс 0.008 ток/с относится к ряду 1.6К; на 52К разброс ВНУТРИ режима
#   1.44-2.47%, то есть больше измеряемого эффекта 1.74%. Медиана спасла, а снос по частоте бьёт по
#   позднему fold, то есть ЗАНИЖАЕТ выигрыш. Повторять замер только с nvidia-smi -lgc.
#   ОСТАТОК ДО ПОТОЛКА ОБЪЯСНЯЛСЯ НЕВЕРНО В 9 РАЗ: never - fold = 87-97 мкс/токен, а не ~10.
#   Откатов при B=1 РОВНО НОЛЬ, значит never и fold отличаются двумя вещами: одним узлом memset
#   (~10 мкс) и АДРЕСАМИ выходов. Остальные ~80 мкс платятся ЛОКАЛЬНОСТЬЮ: 236 выходов переехали с
#   горстки переиспользуемых горячих адресов на 236 холодных в плите, а между обнулением и
#   употреблением протекает ~11.7 ГБ весов gptq и вымывает L2; сама плита 8 МБ > 6 МБ L2.
#   СЛЕДСТВИЕ ДЛЯ VLLM_GPTQ_FOLD_MB: рычаг смотрит в МЕНЬШУЮ плиту (5-6 МБ, при B=1 нужно 4.06 МБ и
#   это влезает в L2), а НЕ в 40 -- memset зануляет всю плиту КАЖДЫЙ токен, поэтому 40 МБ стоят
#   +37 мкс на любом размере пакета, включая B=1, то есть 7% собственного выигрыша.
#   VLLM_GPTQ_ZERO_C=0 -- ПРЕЖНЕЕ УМОЛЧАНИЕ (обнулять только декодные, как было до этой правки).
#   NB: после смены умолчания на fold всякий скрипт, считающий "default = не задавать переменную",
#   меряет fold против fold. Задавать явно.
# Префилл НЕ меняется (x0.99): он идёт веткой reconstruct -> cublasHgemm с beta=0, где C и в
# default не обнуляется. Выигрыш ЦЕЛИКОМ декодный, как и предсказывала опись.
#
# ЧИСЛЕННО ЧИСТ. Побайтовая сверка с default: при B=1 и при B=4 совпадает везде, КРОМЕ тех
# промптов, на которых default расходится САМ С СОБОЙ между двумя подъёмами (символ 384 и 488
# -- ровно те же индексы). Проверять надо было именно так: два РАЗНЫХ процесса default дают
# те же расхождения, что и default против fold, т.е. это собственный шум ветки (atomicAdd в
# fp16 по 40..136 срезам k не ассоциативен), а не порча. Вырождения нет (ноль '!' в выводе).
# VLLM_GPTQ_ZERO_C=1 -- вернуть безусловное обнуление, =never -- потолок (МУСОР на выходе).
#
# ЗАДАЧА #92 (закрыта в рабочем дереве, в целевое ещё не перенесена; см. внешний отчёт о
# валидации). Три вещи, которые надо знать про этот вентиль:
#  1) ЕДИНИЦА -- ПРОХОД (токен), а не граф. В отгруженном сюда бинарнике границу прохода
#     определяет флаг "мы внутри захвата", и она верна только потому, что vLLM успевает
#     между двумя захватами сделать хотя бы одно умножение вне графа. Захвати он два
#     графа подряд -- второй не получит узла обнуления: ошибка 1.0 на втором
#     воспроизведении, 2.0 на третьем, БЕЗ падения. Воспроизведено стендом.
#  2) ПЛИТА НЕ ПРИВЯЗАНА К КАРТЕ. Вызов с другой карты в том же процессе (черновая модель
#     спекулятивного декода) сворачивается по чужому указателю -- воспроизведено:
#     SIGSEGV. В рабочем дереве это откат, а не падение.
#  3) Ниже -- переменные, которые понимает ТОЛЬКО исправленный бинарник; старый их
#     игнорирует, поэтому строки безопасны и при откате на прежнее дерево.
# VLLM_GPTQ_FOLD_EXPECT -- длина прохода в умножениях. Для этой модели 236 (16 полных
#   слоёв x 3 + 47 линейных x 4; слой 0 не квантован, qkv_proj и in_proj_ba исключены
#   правилом dynamic). Закрепление включает ЖЁСТКИЙ инвариант с первого же прохода:
#   проход другой длины -> ОТКАЗ, а не тихо неверные числа.
# VLLM_GPTQ_FOLD_MB -- размер плиты. ЗАМЕРЕНО (карта V100, 236 умножений боевой
#   раскладки, 11.38 ГБ весов): узел обнуления идёт на полной полосе, 1.21 мкс на МБ.
#   Исправленный бинарник зануляет ТОЛЬКО занятый проходом префикс (4.06 МБ при B=1),
#   поэтому размер плиты при B=1 больше НИЧЕГО не стоит: 8 и 40 МБ дают 14.898 мс против
#   14.880 у потолка. Значит плиту надо брать БОЛЬШУЮ (40 МБ покрывает B=8 целиком и
#   снимает 177 откатов из 236). В СТАРОМ бинарнике всё наоборот -- он зануляет всю
#   плиту, и 40 МБ стоят +37 мкс на КАЖДОМ токене, включая B=1.
export VLLM_GPTQ_ZERO_C="${VLLM_GPTQ_ZERO_C:-fold}"
export VLLM_GPTQ_FOLD_EXPECT="${VLLM_GPTQ_FOLD_EXPECT:-236}"
# ЧАНКОВЫЙ СКАН GATED-DELTA-NET на наших тензорных ядрах вместо fla/Triton. Врезается через
# sitecustomize.py в каталоге шима (он попадает в PYTHONPATH ниже) -- то есть встаёт и в
# spawn-воркерах, где скан и считается. Замерено на боевой форме (T=16384, Hv=24, Hk=8, d=128,
# fp16, чередование, медиана из 15): Triton 20.66 мс -> наше 3.88 мс на слой-проход, x5.33
# (независимая перепроверка; заявка автора x5.40 воспроизвелась). Выигрыш ЗАВИСИТ ОТ T и заголовочную
# цифру переносить нельзя: при max-num-batched-tokens 4096 это x3.81, при 2048 -- x3.15.
# ТОЧНОСТЬ. Три РАЗНЫЕ величины, их нельзя смешивать (я смешивал):
#   наше   против независимого эталона (fused_recurrent, fp32):  4.7e-4
#   Triton против того же эталона:                               5.2e-4   <- ЗАМЕНЯЕМОЕ ХУЖЕ НАШЕГО
#   наше против Triton (замерено, реальные гейты A_log/dt_bias): 6.84e-4
# ПОСЛЕДНЕЕ -- НЕ НАША ОШИБКА, а расстояние между ДВУМЯ приближениями: квадратура двух независимых
# ошибок даёт hypot(4.7, 5.2) = 7.01e-4 против замеренных 6.84e-4, сходится в 2%. Такое совпадение
# и есть подтверждение НЕЗАВИСИМОСТИ округлений (будь они коррелированы, расстояние вышло бы меньше).
# СЛЕДСТВИЕ, важное для любой будущей замены: совпасть с Triton точнее ~7e-4 НЕВОЗМОЖНО В ПРИНЦИПЕ,
# если не воспроизводить его порядок округлений. Целиться надо в расстояние до ИСТИНЫ (планка 5.2e-4
# -- столько стоит заменяемое), а не в попарное расстояние до Triton.
# Ноль NaN; накопления по чанкам НЕТ (6.84e-4 -> 6.97e-4 при росте контекста вчетверо).
# ЗАМЕРЕНО СКВОЗНЯКОМ (парно, чередование ветвей в ОДНОМ подъёме через файловый переключатель,
# медиана 7 пар, один и тот же промпт в паре, префикс-кэш разведён cache_salt -- "из кэша 0"):
#   промпт  чанк   TTFT наш   TTFT triton   r
#     4126  16384   2.486 c     2.698 c    1.088
#    16695  16384  10.009 c    10.901 c    1.089
#    66558  16384  46.638 c    50.049 c    1.073
#   ПРЕФИЛЛ x1.07-1.09 НА ВСЁМ СПЕКТРЕ. Отрицательный контроль: при FA2SM70_GDN=0 тот же жгут даёт
#   r = 0.983/0.998 -- эффект исчезает вместе с подменой.
# ДОЛЯ СКАНА ОПРОВЕРГНУТА В БОЛЬШУЮ СТОРОНУ: обратным счётом из TTFT она 10.9% при 4К, 10.0-10.5%
# при 16К, 8.4-8.9% при 64К -- вдвое больше прежних 5.4-6.0% (те верны лишь для очень длинного
# префилла, откуда и брались 204.5 с). Абсолютно скан у Triton: 0.29 с (4К), 1.10 с (16К), 4.20 (64К).
# ДЕКОД НЕ ТРОНУТ, подтверждено тремя выборками (1.0016 / 1.0108 / 0.9786 -- разброс внутри ветки
# перекрывает разницу): он идёт fused_recurrent_*, а не чанковым путём.
# ЧАНК: стендовое падение x5.33 -> x3.81 при 16384 -> 4096 ПОДТВЕРЖДЕНО как факт стенда, но сквозняком
# это 1.089 -> 1.084 (0.5 п.п.), потому что скан и так ~10%. Держать 16384: чанк 4096 замедляет
# сервер целиком в ОБЕИХ ветвях (10.90 -> 11.22 с у triton).
# КАЧЕСТВО: 6/7 совпадений токенов, |dlogp| медиана 3.4e-05. ОГОВОРКА: сервер НЕ ДЕТЕРМИНИРОВАН сам
# с собой -- Triton против Triton даёт 9.4e-06 и тоже одно расхождение из 7, причём с |dlogp| 1.3e-01,
# то есть БОЛЬШЕ нашего. Причина известна: неассоциативный atomicAdd в gptq. Наше расхождение всего
# в ~3.5 раза выше собственного шума ветки.
# FA2SM70_GDN=0 -> контроль (идёт Triton). Это ось A/B по скану.
export FA2SM70_GDN="${FA2SM70_GDN:-1}"
# ТИХИЙ ЛОГ НА КЛИЕНТСКИХ ОШИБКАХ. vLLM на заявку длиннее контекста возвращает корректный HTTP 400,
# но ПЕРЕД этим печатает 25-строчный ERROR-трейсбек (serving.py: logger.exception -> create_error_response).
# Здесь это не косметика: лог сервера -- доказательная база приёмки (verify_all.sh грепает
# "Error|Traceback|not supported", чтобы отличить "наши ядра встали" от "молча откатились"), и
# клиентский трейсбек даёт ЛОЖНЫЙ красный. Схлопываем ТОЛЬКО VLLMValidationError и ТОЛЬКО трейсбек:
# текст сообщения (в нём все числа) сохраняется, уровень становится WARNING, любая другая ошибка
# идёт нетронутой вместе с трейсбеком. Врезка -- обёртка logging.config.dictConfig из sitecustomize
# (не VLLM_LOGGING_CONFIG_PATH: тот ЗАМЕНЯЕТ конфиг целиком и разъедется при обновлении vLLM).
# Подробности и границы -- fa2sm70_logquiet.py. =0 -> вернуть штатные трейсбеки.
export FA2SM70_QUIET_CLIENT_ERR="${FA2SM70_QUIET_CLIENT_ERR:-1}"
export PYTHONSAFEPATH=1
# All ours by default. DECODE requires STORE (our decode reads the ||k|| bound the store maintains;
# without it the bound is zero and the output is garbage -- that combination once produced a
# "working" server that emitted nonsense).
export FA2SM70_SHIM_DECODE="${FA2SM70_SHIM_DECODE:-1}"
export FA2SM70_SHIM_STORE="${FA2SM70_SHIM_STORE:-1}"
export FA2SM70_SHIM_PREFILL="${FA2SM70_SHIM_PREFILL:-1}"
# --- ФОРМАТ KV-ПУЛА: int8 + масштаб НА ПОЗИЦИЮ (FA2SM70_KV_I8=1) ------------------------------
# Движку по-прежнему говорится --kv-cache-dtype fp8_e4m3: это ЕГО БУХГАЛТЕРИЯ (в списке форматов
# vLLM int8 нет вовсе), а пул e4m3 -- это uint8, то есть просто байты. Записываем в них мы, читаем
# тоже мы, поэтому при KV_I8=1 в тех же байтах лежит int8 с масштабом на позицию, а масштабы -- в
# побочной таблице по слоту. Точность при том же байте втрое выше (стенд: relL2 2.5e-2 -> 7.0e-3,
# x3.4-3.8 против НЕЗАВИСИМОГО fp32-эталона), сборка префилла -- перекладка байт вместо разворота
# в fp16, а d=256 идёт ядром ролей (volta_fwd_ws).
# ВСЕ ЧЕТЫРЕ МЕСТА (запись/сборка/префилл/декод) сверяют признак формата и ПАДАЮТ при
# несовпадении -- полурабочий формат хуже, чем никакой.
# ПО УМОЛЧАНИЮ ВЫКЛЮЧЕНО: пул, записанный int8, нельзя прочитать e4m3-путём, поэтому переключение
# формата -- это ПЕРЕЗАПУСК с пустым кэшем, а не горячая правка.
# ДЕФЕКТ, НАЙДЕННЫЙ И ЗАКРЫТЫЙ 30.07. Симптом: на пуле e4m3 (FA2SM70_KV_I8=0) ПЕРВЫЙ ДЛИННЫЙ
# запрос после свежего подъёма выдавал стену "!" -- NaN в логитах, -- и следующие несколько тоже,
# после чего сервер "выздоравливал". Разложено опытами: (1) мусор уже в ПЕРВОМ токене, значит порча
# в префилле, а не в декоде; (2) короткий прогрев ДО длинного запроса лечит; (3) при безусловном
# обнулении GPTQ (VLLM_GPTQ_ZERO_C=1) отказ остаётся -- свёртка обнулений НИ ПРИ ЧЁМ.
# ПРИЧИНА: площадки сборки страниц выделялись через torch.empty и ВПРИТЫК (cap == need), а ядра
# внимания читают плитками и захватывают позиции за пределом Sk. На первом запросе за концом лежал
# мусор (на fp16 -- NaN/Inf); на последующих ёмкость успевала удвоиться, и дочитывались валидные
# числа прошлого запроса -- отсюда и "выздоровление".
# ЛЕЧЕНИЕ (flash_attn_v100/__init__.py, _TILE_PAD): площадки НУЛЕВЫЕ и с запасом в 256 позиций на
# голову. Дочитанный ноль эквивалентен маске, а запас убирает выход за выделение. Проверено на том
# же воспроизводителе: было 2 отказа из 2 подъёмов, стало 0 из 2; int8 без изменений.
# ПОЧЕМУ int8 ВСЁ РАВНО ОСТАЁТСЯ БОЕВЫМ: на int8-маршруте НАШЕ ВСЁ -- запись кладёт int8 со своей
# таблицей масштабов, сборщик перекладывает байты как есть, ядро ролей ест их напрямую. На e4m3
# формат пула и k_scale/v_scale принадлежат движку, и наши ядра лишь ИСТОЛКОВЫВАЮТ чужие значения;
# именно там неконтролируемое и входит. Путь, который контролируется целиком, дешевле в отладке.
export FA2SM70_KV_I8="${FA2SM70_KV_I8:-0}"
# Чем читать int8-пул на префилле: 1 -- ядро ролей по байтам (d=256), 0 -- точный разворот q*s в
# fp16 и обычное ядро. Ось A/B «эффект формата против эффекта ядра»; ответ один и тот же.
export FA2SM70_KV_I8_BYTES="${FA2SM70_KV_I8_BYTES:-1}"
# Route the prefix/chunked prefill into our flash_attn_prefill_paged (see above).
export VLLM_FLASH_V100_ENABLE_PAGED_PREFILL="${VLLM_FLASH_V100_ENABLE_PAGED_PREFILL:-1}"

# --- the substitution -------------------------------------------------------
# our shim package dir first, then the repo itself (so `import fa2_sm70` resolves).
# USE_SHIM=0 leaves it out entirely -> vLLM loads the incumbent flash_attn_v100 (A/B baseline).
if [ "${USE_SHIM:-1}" = "1" ]; then
  export PYTHONPATH="$FA2SM70_REPO/integrations/vllm_sm70:$FA2SM70_REPO${PYTHONPATH:+:$PYTHONPATH}"
else
  echo "[fa2_sm70] USE_SHIM=0 -> running the INCUMBENT flash_attn_v100 (baseline)" >&2
fi
export FA2SM70_PATH="$FA2SM70_REPO"
# JIT build cache for our CUDA extensions (first boot compiles for a few minutes).
export FA2SM70_BUILD_DIR="${FA2SM70_BUILD_DIR:-$VLLM_ROOT/run/.fa2sm70_build}"
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
export CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}" CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
# ninja is needed by torch's JIT extension loader and is not in the vLLM venv.
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
mkdir -p "$FA2SM70_BUILD_DIR"
# Clear stale JIT build locks. torch's extension loader takes a lock file per extension and waits on
# it in a sleep loop with NO timeout; if a previous server was killed mid-load the lock survives, and
# every future worker then blocks forever on it. That is not a hypothetical: it hung two boots here,
# and the symptom is maximally misleading -- the engine simply stops during CUDA-graph capture, with
# no traceback, no error, and a worker parked in hrtimer_nanosleep. Safe to remove at launch time,
# since no worker exists yet.
find "$FA2SM70_BUILD_DIR" -name lock -type f -delete 2>/dev/null || true

[ -x "$PY" ] || { echo "venv python missing" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "model dir missing: $MODEL" >&2; exit 1; }
[ -d "$FA2SM70_REPO/integrations/vllm_sm70/flash_attn_v100" ] || { echo "shim missing" >&2; exit 1; }

echo "[fa2_sm70] shim: $FA2SM70_REPO/integrations/vllm_sm70/flash_attn_v100" >&2
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits >&2 || true

RUNDIR="$VLLM_ROOT/run/.rundir_fa2sm70"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

# --- СЕРИАЛИЗАЦИЯ СЛОТА КАРТ ------------------------------------------------------------------
# Эта модель при TP=2 занимает минимум ~18.1 ГиБ на карту (веса 15.52 + контекст CUDA 0.6 +
# активации ~1.5 + графы 0.5), а этот скрипт по умолчанию просит util=0.90. Значит на паре 32-ГиБ
# карт помещается РОВНО ОДИН экземпляр, и два параллельных запуска не просто конкурируют -- они
# ОБА ПАДАЮТ: между проверкой "свободно ли" и стартовым контролем внутри vLLM проходит ~40 с
# загрузки весов, и кто первый дошёл до контроля, тот и взял, а остальные получают
# "Free memory on device cuda:N is less than desired GPU memory utilization".
# Наблюдалось живьём: три подъёма одного замера сорваны подряд, и один раз ТРИ сервера
# одновременно грузили веса в одну карту, отбирая память друг у друга. Договорённостями это не
# лечится -- слот достаётся тому, кто чаще опрашивает, а не тому, кому нужнее. Поэтому замок
# берётся ЗДЕСЬ, в общем инструменте, а не в скрипте каждого желающего.
# Держателем пишется $$ -- он переживает exec ниже, поэтому владельцем замка становится САМ
# СЕРВЕР, и замок снимается автоматически при его смерти (проверка kill -0). Забытых замков нет.
# GPU_SLOT_LOCK=0 отключает (боевой сервер на 2,3 сериализовать не нужно -- он там один).
if [ "${GPU_SLOT_LOCK:-1}" = "1" ] && [ "$CUDA_VISIBLE_DEVICES" != "2,3" ]; then
  export GPU_SLOT_NAME="gpu$(echo "$CUDA_VISIBLE_DEVICES" | tr -d ',')"
  "$SCRIPT_DIR/gpu_slot.sh" acquire \
    "${GPU_SLOT_OWNER:-$(basename "${0}")-$$}" \
    "${GPU_SLOT_WHY:-порт ${PORT:-8085}, util ${GPU_UTIL:-0.90}}" \
    "${GPU_SLOT_TTL:-45}" "$$" || {
      echo "[fa2_sm70] слот карт $CUDA_VISIBLE_DEVICES не освободился -- НЕ поднимаюсь" >&2
      exit 1; }
fi
# CUSTOM ALL-REDUCE: KEEP DISABLED. With NVLink fitted it is tempting to enable vLLM's own
# all-reduce (it targets P2P and beats NCCL on the small payloads decode uses), and it did MEASURE
# slightly faster -- decode 37.4 vs 35.1 tok/s at 4K. But the output is CORRUPT: "The capital of
# France is" -> " Paris!!!!!!!!!" -- starts right, then degenerates. Those speed numbers are void.
# DISABLE_CAR=" " re-enables it if someone wants to retest after a vLLM upgrade.
# (NB: a comment must NOT live inside the backslash-continued argument list below -- doing that
# silently TRUNCATES the command at that point, and the server then comes up on default port 8000
# with default memory settings, which looks like a working server and is not the one you asked for.)
exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name huihui-qwen3.5-35b-a3b-abliterated-q8_0 gpt-4-0125-preview Qwen3.6-27B-AWQ \
  --trust-remote-code \
  --quantization "${QUANT:-awq}" \
  --dtype float16 \
  --tensor-parallel-size 2 \
  ${DISABLE_CAR:---disable-custom-all-reduce} \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --kv-cache-memory-bytes "$KV_CACHE_BYTES" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
  --enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-16384}" \
  --mm-processor-cache-gb 1 \
  --limit-mm-per-prompt "$LIMIT_MM" \
  --generation-config "$GENERATION_CONFIG" \
  --override-generation-config "$OVERRIDE_GEN_CFG" \
  --chat-template "$CHAT_TEMPLATE" \
  --chat-template-content-format openai \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8]}' \
  ${EXTRA_ARGS:-} \
  --host 0.0.0.0 --port "${PORT:-8085}"
