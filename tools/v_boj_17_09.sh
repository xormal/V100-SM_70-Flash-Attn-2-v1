#!/bin/bash
# ПЕРЕВОД В БОЙ 17.09: лечение отказа памяти на одновременных запросах с картинками.
# Кладёт два файла в боевое дерево и печатает строки, которые надо добавить в пускач.
# Перезапуск боевого НЕ делает -- это отдельное решение владельца.
set -u
SRC=/mnt/d1/alex/VLLM_fa2/solutions/fa2_sm70_cutlass_grade/integrations/vllm_sm70
DST=/mnt/d1/alex/VLLM_fa2_prod/fa2_sm70_cutlass_grade/integrations/vllm_sm70
SEGODNYA=$(date +%d_%m)
for f in fa2sm70_mtp_w8.py fa2sm70_w8.py; do
  [ -f "$DST/$f" ] && cp -a "$DST/$f" "$DST/$f.pered_$SEGODNYA"
  cp "$SRC/$f" "$DST/$f"
  echo "положен $f  (md5 $(md5sum < $DST/$f | cut -c1-8))"
done
cat <<'TXT'

В пускач /mnt/d1/alex/VLLM_ch/run/serve_boevoy_qwen38abl.sh:
  export FA2SM70_MTP_W8_NSPLIT="${FA2SM70_MTP_W8_NSPLIT:-4}"   # широкое тело головы MTP -- 4 куска
  export FA2SM70_W8_OOM_RETRY="${FA2SM70_W8_OOM_RETRY:-1}"     # отказ памяти не роняет запрос
  export FA2SM70_TM8_RECON_MMIN="${FA2SM70_TM8_RECON_MMIN:-1024}"  # вернуть: к падению он не причастен
TXT
