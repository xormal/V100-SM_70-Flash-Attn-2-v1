# План продвижения в бой (после 12/12 на стенде) -- НЕ ПРИМЕНЕНО, ждёт разрешения владельца

## 1. Ядра (promote_to_prod из стендового дерева /mnt/d1/alex/VLLM_fa2_stend = prod + 3 файла)
- fa2_sm70/csrc/decode_hmma.cuh   -- PRMT-деквант, V3 (2 блока/SM), V4; побайтово = prod, x1.56 на 262K
- fa2_sm70/csrc/split_decode.cu   -- carveout shared 100 для двух блоков/SM
- fa2_sm70/csrc/sparse_select.cu  -- FA2SM70_SPARSE_MASS (порог по массе: 12/12 на Википедии при 0.95), STRAT_PAIR (не нужен), счётчик массы stat
- integrations/vllm_sm70/backends/fa2_sm70.py (venv = dev) -- дамп отбора FA2SM70_SEL_DUMP (прибор, выкл), TAIL_DENSE (не нужен, выкл)
Пересборка в бою: fa2sm70_decode и fa2sm70_prefill (~10 мин при подъёме; health=200 != готовность).

## 2. Пускач serve_boevoy_qwen38abl.sh (умолчания)
- FA2SM70_SPARSE_PCT=100            (вернуть разреженный: плотный <200 с невозможен по FLOP)
- FA2SM70_SEL_STRAT=8               (лучший в окне)
- FA2SM70_SPARSE_B=64               (блок отбора 64 -- пол по кратности BK)
- FA2SM70_SPARSE_MASS=0.95 (или 0.90, если и оно даёт 12/12 -- ради времени)
- FA2SM70_SEL_STRAT_PAIR=0
- FA2SM70_MAX_SPLITS=160            (химера: 2 волны при 2 блоках/SM)
- FA2SM70_SPARSE_ALPHA=0.02, GAMMA=0.5, KVDT=int8_per_token_head -- как сейчас

## 3. Гейты на боевом (окно владельца ~25 мин)
- мульти-игла Википедия 12/12 (tools/multi_igla_tekst.py 8085 250000), время префилла 250K
- гейт качества 3/3 коротко; декод коротко (>= 54 ток/с) и на 128K/250K (ожидание: внимание 20 -> 13 мс/шаг)
