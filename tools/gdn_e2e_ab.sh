#!/usr/bin/env bash
# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# СКВОЗНОЙ A/B ПО СКАНУ GDN: два подъёма сервера на одних и тех же картах, отличающиеся РОВНО
# одной переменной FA2SM70_GDN, и одинаковый замер TTFT на 4K/16K/64K.
#
# ПОЧЕМУ ДВА ПОДЪЁМА, А НЕ ПЕРЕКЛЮЧЕНИЕ НА ЛЕТУ. Подмена имени встаёт на импорте (см.
# fa2_gdn_shim): переключить её у живого сервера нельзя, а переменная окружения, прочитанная
# внутри обёртки, переключала бы только откат -- то есть мерила бы не то.
#
# ЧТО ЗДЕСЬ СДЕЛАНО ПРОТИВ ИЗВЕСТНЫХ ЛОВУШЕК ЭТОГО ПРОЕКТА:
#   * УНИКАЛЬНЫЕ ПРОМПТЫ на каждый замер (bench_server.py их и строит): повторный промпт попадает
#     в префикс-кэш и даёт физически невозможные цифры -- так в проекте родились "9677 ток/с";
#   * ПРОГРЕВ КАЖДОГО РАЗМЕРА до замера: первый запрос после подъёма платит захват графов и JIT
#     и занижает сам себя вдвое;
#   * УБИЙСТВО ТОЛЬКО ПО ЗАПИСАННОМУ PID. pkill -f по шаблону попадает в СОБСТВЕННУЮ командную
#     строку (exit 144) -- за сессию это случилось четыре раза;
#   * ПРОВЕРКА ОСМЫСЛЕННОСТИ вывода: в проекте был сервер, отвечавший " Paris!!!!!!!!!" -- быстрый
#     и полностью негодный. Скорость без этой проверки не значит ничего.
set -uo pipefail

W="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# ПОДЪЁМНИК берётся из РАБОЧЕГО дерева, а КОД -- из этого (FA2SM70_REPO=$W). Причина конкретна:
# у подъёмника в рабочем дереве есть незакоммиченный захват слота карт (gpu_slot.sh), которого
# в этой копии нет. Замок обязан держать САМ СЕРВЕР -- он пишет pid=$$ и делает exec, поэтому
# держателем становится долгоживущий процесс, и замок самоочищается при его смерти. Отдельный
# ждущий процесс на эту роль НЕ ГОДИТСЯ: acquire завершается сразу после взятия, его PID умирает,
# и следующий желающий законно снимает "мёртвый" замок.
LAUNCHER="${LAUNCHER:-$W/integrations/vllm_sm70/start_model_fa2sm70_gpu23.sh}"
OUT="${OUT:-/tmp/fa2sm70-gdn-e2e}"
PORT="${PORT:-8812}"
SIZES="${SIZES:-4096,16384,65536}"
REPS="${REPS:-3}"
export VLLM_ROOT="${VLLM_ROOT:?set VLLM_ROOT to the external vLLM checkout}"
PY="$VLLM_ROOT/.venv-1cat-vllm/bin/python"

boot() {                                    # $1 = 0|1 значение FA2SM70_GDN, $2 = метка
  local gdn="$1" tag="$2"
  echo "=== ПОДЪЁМ [$tag]: FA2SM70_GDN=$gdn, порт $PORT ===" >&2
  CUDA_VISIBLE_DEVICES=0,1 \
  QUANT=gptq \
  MODEL="${MODEL:?set MODEL to the external model directory}" \
  USE_SHIM=1 \
  FA2SM70_GDN="$gdn" \
  FA2SM70_GDN_LOG=1 \
  FA2SM70_REPO="$W" \
  FA2SM70_BUILD_DIR="$OUT/gdnbuild" \
  GPU_SLOT_OWNER="gdn88-$tag" \
  GPU_SLOT_WHY="задание 88: сквозной A/B скана GDN ($tag), порт $PORT" \
  GPU_SLOT_TTL=40 \
  CUDA_HOME="${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit}" \
  CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ \
  PORT="$PORT" \
  nohup "$LAUNCHER" > "$OUT/srv_$tag.log" 2>&1 &
  echo $! > "$OUT/srv_$tag.pid"
  local pid; pid=$(cat "$OUT/srv_$tag.pid")
  echo "  pid=$pid, ждём слота и готовности..." >&2
  # ДО 100 МИНУТ, и это не запас "на всякий случай": сюда входит ОЖИДАНИЕ ЧУЖОГО ЗАМКА, а у него
  # штатный TTL 45-95 минут. Короткий таймаут здесь означает не "быстро упадём", а "потеряем
  # замер из-за соседа", то есть переизмерение с нуля.
  for i in $(seq 1 1200); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
      echo "  готов через $((i*5)) с" >&2; return 0
    fi
    kill -0 "$pid" 2>/dev/null || { echo "  ПОДЪЁМ УПАЛ (см. $OUT/srv_$tag.log)" >&2; return 1; }
    sleep 5
  done
  echo "  НЕ ПОДНЯЛСЯ за 30 мин" >&2; return 1
}

stop() {                                    # только по ЗАПИСАННОМУ pid, никаких шаблонов
  local tag="$1" pid
  [ -f "$OUT/srv_$tag.pid" ] || return 0
  pid=$(cat "$OUT/srv_$tag.pid")
  echo "=== ОСТАНОВ [$tag] pid=$pid ===" >&2
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
  sleep 8
  rm -f "$OUT/srv_$tag.pid"
}

sanity() {                                  # ОСМЫСЛЕННОСТЬ, а не только скорость
  local tag="$1"
  local m; m=$(curl -s "http://localhost:$PORT/v1/models" | "$PY" -c \
      'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
  local ans; ans=$(curl -s "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$m\",\"prompt\":\"Question: What is the capital city of France? Answer:\",\"max_tokens\":24,\"temperature\":0}" \
      | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"].strip().replace(chr(10)," "))')
  echo "ОСМЫСЛЕННОСТЬ [$tag]: $ans"
  case "$ans" in
    *Paris*|*paris*) echo "  -> содержит Paris: ГОДНО" ;;
    *) echo "  -> НЕ содержит Paris: ПОДОЗРИТЕЛЬНО" ;;
  esac
}

run_bench() {
  local tag="$1" r
  for r in $(seq 1 "$REPS"); do
    echo "--- [$tag] прогон $r/$REPS ---"
    "$PY" "$W/integrations/vllm_sm70/bench/bench_server.py" "$PORT" "$tag-$r" "$SIZES" 2>&1
  done
}

main() {
  local tag
  for spec in "1:ours" "0:triton"; do
    tag="${spec#*:}"
    boot "${spec%%:*}" "$tag" || { stop "$tag"; echo "ПРОВАЛ подъёма $tag"; exit 1; }
    sanity "$tag"          | tee -a "$OUT/ab_$tag.txt"
    grep -c "подмена встала" "$OUT/srv_$tag.log" | sed "s/^/ПОДМЕНА ВСТАЛА в процессах: /" \
      | tee -a "$OUT/ab_$tag.txt"
    grep -c "откат к Triton" "$OUT/srv_$tag.log" | sed "s/^/ОТКАТОВ К TRITON: /" \
      | tee -a "$OUT/ab_$tag.txt"
    run_bench "$tag"       | tee -a "$OUT/ab_$tag.txt"
    stop "$tag"
  done
  echo "=== ГОТОВО. Итоги: $OUT/ab_ours.txt и $OUT/ab_triton.txt ==="
}

main "$@"
