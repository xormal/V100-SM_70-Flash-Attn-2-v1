#!/bin/bash
# ПОДЪЁМ БОЕВОГО ПОСЛЕ ПЕРЕЗАГРУЗКИ. Запускается службой boevoy-vllm.service внутри tmux
# (сессия `vllm`), чтобы к нему можно было подключиться: tmux attach -t vllm
#
# ЗАЧЕМ ОТДЕЛЬНЫЙ ФАЙЛ, А НЕ ПРЯМОЙ ВЫЗОВ АВТОПОДЪЁМЩИКА:
#   1. после перезагрузки драйвер поднимается НЕ МГНОВЕННО -- ждём, пока nvidia-smi увидит
#      все карты, иначе сборка и профилировка памяти падают на пустом списке устройств;
#   2. снимаем снимок ТЕПЛА И ТАКТОВ до старта -- он нужен, чтобы после перестановки карт
#      было с чем сравнить (у карты с горячим HBM тепловой сброс виден сразу);
#   3. чистим осиротевшие замки прошлого запуска (после жёсткой перезагрузки они остаются).
set -uo pipefail
RUNDIR=/mnt/d1/alex/VLLM_ch/run
LOG="$RUNDIR/logs/boot_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$RUNDIR/logs"
exec > >(tee -a "$LOG") 2>&1
say() { printf '[boot] %s %s\n' "$(date '+%F %T')" "$*"; }

say "жду драйвер и карты"
for i in $(seq 60); do
  n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  [ "${n:-0}" -ge 4 ] && { say "карт видно: $n"; break; }
  sleep 5
done
[ "${n:-0}" -ge 4 ] || { say "ОТКАЗ: карт видно $n из 4 -- не поднимаю"; exit 1; }

say "снимок карт до старта (ядро / память / такт / предел):"
nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory,clocks.sm,clocks.applications.graphics,power.limit \
           --format=csv,noheader | sed 's/^/[boot]   /'

# Замки прошлого запуска: после жёсткой перезагрузки владельца у них нет.
for f in "$RUNDIR"/.serve_*.lock; do
  [ -e "$f" ] || continue
  if [ -z "$(fuser "$f" 2>/dev/null)" ]; then rm -f "$f" && say "снят осиротевший замок $(basename "$f")"; fi
done
for l in $(find "$RUNDIR"/.fa2sm70_build_* "$HOME/.cache/torch_extensions" -name lock 2>/dev/null); do
  rm -f "$l" && say "снят сборочный замок $l"
done

say "передаю управление автоподъёмщику (он же перезапустит боевой при падении)"
exec bash "$RUNDIR/tmux_autostart_model.sh"
