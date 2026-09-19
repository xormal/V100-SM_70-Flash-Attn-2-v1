# [ВЫПОЛНЕНО 12.09 16:22 и 17:49] боевой: 12/12 за 276.8 с; второе окно 17:49 -- FA2SM70_EAGLE_SBROS=0: вопрос 40 -> 11.7 с (x3.4), 12/12 за 269.3 с, смешанный батч ok

# Процедура боевого окна (~35 мин) -- выполнять ТОЛЬКО по разрешению владельца

0. Снимок: `cp -a /mnt/d1/alex/VLLM_fa2_prod /mnt/d1/alex/VLLM_fa2_prod.pered_12_09` ; `cp serve_boevoy_qwen38abl.sh{,.pered_12_09}`
1. Ядра в prod (3 файла из стендового дерева, побайтовые гейты пройдены):
   cp /mnt/d1/alex/VLLM_fa2_stend/fa2_sm70_cutlass_grade/fa2_sm70/csrc/{decode_hmma.cuh,split_decode.cu,sparse_select.cu} \
      /mnt/d1/alex/VLLM_fa2_prod/fa2_sm70_cutlass_grade/fa2_sm70/csrc/
   (venv-бэкенд уже = dev: дамп/TAIL_DENSE выключены по умолчанию)
2. Пускач serve_boevoy_qwen38abl.sh: SPARSE_PCT=100, SEL_STRAT=8, SPARSE_B=64, SPARSE_MASS=0.95, MAX_SPLITS=160
   (KVDT=int8_per_token_head, ALPHA=0.02, GAMMA=0.5 -- как есть). Комментарий в §305-блоке обновить.
3. Перезапуск боевого ТОЛЬКО по PID из ss (kill -TERM), надзиратель поднимет; пересборка decode+prefill ~10 мин:
   ждать health=200 И строки «Starting vLLM API server», потом ещё прогрев.
4. Гейты: tools/smeshannyj_batch.py 8085 (декод+префилл одновременно -- ОБЯЗАТЕЛЬНО), tools/multi_igla_tekst.py 8085 250000 (12/12 и время 250K -- ЦЕЛЬ <200 с), гейт 3/3 коротко,
   декод коротко (>=54 ток/с), декод после 250K (ожидание: внимание 20->13 мс/шаг).
5. Если 12/12 и время >200 с: владелец выбирает MASS (0.95 точно; 0.93 -- быстрее, риск 11/12).
6. Откат: вернуть 3 файла из .pered_12_09 и пускач, перезапуск.
