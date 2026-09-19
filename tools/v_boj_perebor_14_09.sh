#!/bin/bash
# Постановка рычагов перебора 14.09 в БОЕВОЕ дерево -- ТОЛЬКО с разрешения владельца (боевой перезапускается им).
# Что делает: копирует три файла в /mnt/d1/alex/VLLM_fa2_prod и печатает строки для пускача. Сам пускач и боевой НЕ трогает.
# После копирования первый подъём боевого пересоберёт prefill (+attn_fwd_cutlass/sparse_select) и tm8 (~10 мин, прогрев вторым процессом есть в пускаче).
set -e
DEV=/mnt/d1/alex/VLLM_fa2/solutions/fa2_sm70_cutlass_grade
PROD=/mnt/d1/alex/VLLM_fa2_prod/fa2_sm70_cutlass_grade
for f in fa2_sm70/csrc/sparse_select.cu fa2_sm70/csrc/tm8_gemm.cu fa2_sm70/csrc/attn_fwd_cutlass.cu fa2_sm70/csrc/split_decode.cu fa2_sm70/csrc/decode_hmma.cuh integrations/vllm_sm70/fa2sm70_w8.py integrations/vllm_sm70/backends/fa2_sm70.py; do
  cp -v "$PROD/$f" "$PROD/$f.pered_perebor_14_09" 2>/dev/null || true
  cp -v "$DEV/$f" "$PROD/$f"
done
cat <<'TXT'
Добавьте в /mnt/d1/alex/VLLM_ch/run/serve_boevoy_qwen38abl.sh (рядом с блоком FA2SM70_SPARSE_LAYER_MAX):
  export FA2SM70_SEL_GEMM="${FA2SM70_SEL_GEMM:-1}"            # зонды отбора матрицами: −10.5 с ядер на 250K, лесенка 32/32
  export FA2SM70_TM8_RECON_MMIN="${FA2SM70_TM8_RECON_MMIN:-1024}"  # тела при M>=1024 через reconstruct+cuBLAS: x1.21, точность та же
  export FA2SM70_MAXBATCH_FORCE="${FA2SM70_MAXBATCH_FORCE:-2048}" # чанк 2048: держит ёмкость 262 144 вместе с разреженным декодом (замер 15.09)
  export FA2SM70_SPARSE_MINSQ="${FA2SM70_SPARSE_MINSQ:-2048}"     # последний (неполный) чанк плотный: лесенка 32/32 и 12/12 на 250K БЕЗ плотных слоёв (LAYER_MAX не нужен)
  export FA2SM70_SEL_ROWNORM="${FA2SM70_SEL_ROWNORM:-1}"         # свёртка зондов с нормировкой по строке: отбор верен для всех строк
  export FA2SM70_SPARSE_MASS="${FA2SM70_SPARSE_MASS:-0.85}"       # граница без потерь по согласию с плотным: 0.85 (128/128, |Δlp|<=0.014), 0.80 уже расходится; стенд 250K: 0.85 -- 285 с, 0.90 -- 302, 0.98 -- 341
Ожидание на боевом (NVLink): ~208 с БЕЗ ПОТЕРЬ (стенд 285 с против 418 у базы) -- вместо нынешних 269 с с потерей различения на 32 иглах.
Пускач ставит FA2SM70_SPARSE_MINSQ через `:-0` (строка ~234): внешнее значение из окружения действует (проверено стендом), либо поменяйте умолчание там же на 3072.
n-граммы для копирующих задач (списки, таблицы, правка кода): export FA2SM70_MTP_NGRAM=1 -- источник черновика выбирается по факту приёмки, на пересказе автоматически не берётся.
Разреженный ДЕКОД (по решению отдельно, бюджет KV урезать НЕ нужно: хватает чанка 2048, ёмкость 262 144 держится):
  export FA2SM70_SPARSE_DEC=1 FA2SM70_SPARSE_DEC_TOPF=0.15 FA2SM70_SPARSE_DEC_GRP=0   # 12/12 игл на 250K, внимание+отбор 9 мс/шаг вместо 28
  export FA2SM70_SEQLENS_LAZY=1                                                     # без неявной синхронизации длин (черновик звал её 4 раза за шаг): +12 % декода на 250K
Бэкенд fa2_sm70.py в venv уже правлен (рычаги выкл = прежнее поведение); backends/fa2_sm70.py в боевом дереве -- для sitecustomize-копии, если она там используется.
Проверка после подъёма: в журнале строки «[fa2_sm70] отбор: зонды счётом МАТРИЦАМИ» и «[fa2_sm70 TM8] RECON: тела при M>=1024 ...», ёмкость 262 144.
TXT
