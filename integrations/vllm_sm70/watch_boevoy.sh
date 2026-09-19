#!/usr/bin/env bash
# СТОРОЖ БОЕВОГО: телеметрия памяти + автоподъём после смерти.
#
# ЗАЧЕМ. 17.08 боевой умер по OOM в 21:31 и лежал 40 минут, пока человек не заметил. Занятость
# torch выросла с ~25 до 30.5 ГиБ за 4 часа при ровном пуле KV -- источник роста НЕ НАЗВАН.
# Сторож делает две вещи: пишет кривую памяти (следующая смерть оставит улику) и поднимает
# мёртвый сервер за минуты.
#
# ЧЕГО ОН НЕ ДЕЛАЕТ. Не перезапускает ЖИВОЙ сервер ни при каких показаниях памяти -- решение о
# профилактическом рестарте принимает человек по кривой. Не трогает карты 0-1. Не обходит гейты:
# подъём идёт через tmux_boevoy_qwen38abl.sh со всеми его проверками (занятость карт, замок,
# гейт генерацией) -- если человек в этот момент сам что-то поднимает, скрипт откажет, и сторож
# просто подождёт следующего цикла.
#
# Запуск:   tmux new-session -d -s watch8085 '/mnt/d1/alex/VLLM_ch/run/watch_boevoy.sh'
# Остановка: tmux kill-session -t watch8085
# Телеметрия: /mnt/d1/alex/VLLM_ch/run/boevoy_mem.csv (раз в минуту)
set -u

RUNDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8085
CSV="$RUNDIR/boevoy_mem.csv"
ZH="$RUNDIR/watch_boevoy.log"
COOLDOWN=900
last_restart=0
fails=0

say() { printf '%s [сторож] %s\n' "$(date +%F_%T)" "$*" >> "$ZH"; }
[ -f "$CSV" ] || echo "время,код,МиБ_карта2,МиБ_карта3,запросов_бежит" > "$CSV"
say "старт (порт $PORT, кулдаун ${COOLDOWN}с)"

while :; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 2>/dev/null | paste -sd,)
  run=$(sed 's/\x1b\[[0-9;]*m//g' "$(ls -t $RUNDIR/serve_${PORT}_*.log 2>/dev/null | head -1)" 2>/dev/null \
        | grep -oE "Running: [0-9]+ reqs" | tail -1 | grep -oE "[0-9]+")
  echo "$(date +%F_%T),$code,${mem:-,-},${run:-?}" >> "$CSV"

  if [ "$code" = "200" ]; then
    fails=0
  else
    fails=$((fails+1))
    say "порт не отвечает ($code), подряд: $fails"
    if [ "$fails" -ge 3 ]; then
      # Подъём только если сервера действительно НЕТ: ни процесса с этим портом, ни занятых карт.
      alive=0
      for p in $(ps -eo pid= -o args= | grep "[a]pi_server" | awk '{print $1}'); do
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -- "--port $PORT" && alive=1
      done
      busy=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | while IFS=, read -r pid uuid; do
        idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep -F "${uuid// /}" | cut -d, -f1 | tr -d ' ')
        { [ "$idx" = "2" ] || [ "$idx" = "3" ]; } && echo -n x
      done)
      now=$(date +%s)
      if [ "$alive" = "1" ] || [ -n "$busy" ]; then
        say "процесс или карты ещё заняты (alive=$alive busy='$busy') -- поднимается кто-то другой, жду"
      elif [ $((now - last_restart)) -lt "$COOLDOWN" ]; then
        say "кулдаун ($((now-last_restart))с из ${COOLDOWN}с) -- если падает так часто, поднимать бессмысленно, нужен человек"
      else
        say "СЕРВЕР МЁРТВ -- поднимаю через tmux_boevoy_qwen38abl.sh"
        last_restart=$now
        if "$RUNDIR/tmux_boevoy_qwen38abl.sh" >> "$ZH" 2>&1; then
          say "ПОДНЯТ, гейт генерации пройден"
        else
          say "подъём НЕ УДАЛСЯ (см. строки выше) -- следующая попытка после кулдауна"
        fi
        fails=0
      fi
    fi
  fi
  sleep 60
done
