#!/usr/bin/env bash
# Auto-start entry point for the tmux `vllm` session (wired in ~/.tmux.conf).
#
# Runs the PRODUCTION model -- Huihui-Qwen3.6-27B-abliterated-INT8, TP=2 on cards 2-3, :8085 --
# with a single-flight guard (never a second instance on a busy port) and crash-restart
# with backoff. Logs land in $ROOT/run/logs/ (last 20 kept).
#
# Manual use:  tmux attach -t vllm
# Override the model:  DEFAULT_MODEL_SCRIPT=/path/to/start_model_moe_awq.sh tmux
set -uo pipefail

ROOT=/mnt/d1/alex/VLLM_ch
# BOEVOY PUSKACH, a ne iyulskaya AWQ-sborka. Do 05.08.2026 zdes stoyal
# start_model_awq_e4m3.sh -- DRUGAYA model (AWQ) na TP=4 i VSE CHETYRE karty, hotya v boyu
# s int8-raboty stoit INT8 na TP=2 i kartakh 2-3. Posle perezagruzki mashina podnimala ne tu
# set i otbirala testovye karty 0-1. Puskach sam beryot PORT=8085 CARDS=2,3 po umolchaniyu.
# УМОЛЧАНИЕ -- БОЕВОЙ ПУСКАЧ Qwen3.8-ABL. Прежнее (serve_qwen36_int8.sh) пережило переезд
# боевого и 18.08 после сбоя питания молча подняло СТАРУЮ сеть со старым деревом ядер и без
# наших рычагов: сервер отвечал 200, и по нему это было НЕ ВИДНО. Откат к прежнему -- явным
# DEFAULT_MODEL_SCRIPT, а не по умолчанию.
START="${DEFAULT_MODEL_SCRIPT:-$ROOT/run/serve_boevoy_qwen38abl.sh}"
LOGDIR="$ROOT/run/logs"
PORT="${MODEL_PORT:-8085}"
MIN_HEALTHY_SEC=300    # an exit sooner than this counts as a "fast failure"
MAX_FAST_FAILS=3       # that many in a row -> stop retrying, keep the pane for triage

mkdir -p "$LOGDIR"
ls -1t "$LOGDIR"/model_*.log 2>/dev/null | tail -n +21 | xargs -r rm -f

port_busy() { ss -ltn "sport = :$PORT" 2>/dev/null | grep -q ":$PORT"; }

if port_busy; then
  echo "[autostart] :$PORT is already serving -- standing by, will NOT start a second instance."
  while port_busy; do sleep 30; done
  echo "[autostart] :$PORT freed up -- taking over."
fi

fails=0
while :; do
  LOG="$LOGDIR/model_$(date +%Y%m%d_%H%M%S).log"
  echo "[autostart] $(date '+%F %T') starting $START  (log: $LOG)"
  start_ts=$SECONDS
  "$START" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  dur=$(( SECONDS - start_ts ))
  echo "[autostart] $(date '+%F %T') exited rc=$rc after ${dur}s"

  if [ "$dur" -lt "$MIN_HEALTHY_SEC" ]; then fails=$((fails+1)); else fails=0; fi
  if [ "$fails" -ge "$MAX_FAST_FAILS" ]; then
    echo "[autostart] $fails fast failures in a row -- giving up."
    echo "[autostart] last log: $LOG ; rerun by hand with: $START"
    break
  fi
  sleep $(( 10 * fails ))
done

# Never let the pane vanish -- leave an interactive shell for triage.
exec bash -i
