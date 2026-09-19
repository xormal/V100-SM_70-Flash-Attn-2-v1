#!/usr/bin/env bash
# БОЕВОЙ В TMUX: Qwen3.8-27B-ABLITERATED, карты 2-3, порт 8085.
#
# ЗАЧЕМ TMUX, А НЕ nohup. Базовый пускач заканчивается конвейером `... | tee "$LOG"`, и без своей
# СЕССИИ он умирает вместе с породившей оболочкой -- 17.08 это уже стоило пустого подъёма: процесс
# исчез, в журнале одна строка, ошибки нет. tmux даёт сессию, переживающую разрыв, и живое окно,
# в которое можно заглянуть.
#
# ЧТО ЭТО НЕ ДЕЛАЕТ. Не решает за вас, можно ли гасить работающий боевой. Если карты заняты --
# ОТКАЗ с показом, кто их держит; гасить только намеренно, ключом --restart, и только по PID,
# опознанному снизу от nvidia-smi.
#
# Использование:
#   ./tmux_boevoy_qwen38abl.sh              поднять (откажет, если карты/порт заняты)
#   ./tmux_boevoy_qwen38abl.sh --restart    погасить занявшего карты 2-3 и поднять заново
#   ./tmux_boevoy_qwen38abl.sh --status     что запущено, где журнал, отвечает ли порт
#   ./tmux_boevoy_qwen38abl.sh --attach     войти в окно (выход: Ctrl-b d)
#   ./tmux_boevoy_qwen38abl.sh --log        напечатать путь к ТЕКУЩЕМУ журналу
#   ./tmux_boevoy_qwen38abl.sh --stop       погасить
set -uo pipefail

SESSION="${SESSION:-boevoy38abl}"
PORT="${PORT:-8085}"
CARDS="${CARDS:-2,3}"
RUNDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$RUNDIR/serve_boevoy_qwen38abl.sh"
WRAPLOG="$RUNDIR/boevoy_wrap_$(date +%Y%m%d_%H%M%S).log"

say() { printf '[tmux-боевой] %s\n' "$*"; }
die() { printf '[tmux-боевой] ОТКАЗ: %s\n' "$*" >&2; exit 1; }

# Журнал СЕРВЕРА заводит базовый пускач сам, с отметкой времени: serve_<порт>_<дата>_<время>.log.
# Здесь -- только способ его найти, а не второе место записи: два журнала одного и того же
# расходятся, и потом непонятно, какой настоящий.
current_log() { ls -t "$RUNDIR"/serve_${PORT}_*.log 2>/dev/null | head -1; }

# Кто держит карты. Опознание ТОЛЬКО снизу: PID -> gpu_uuid -> НОМЕР карты. По шаблону имени
# процесса боевой уже роняли трижды, поэтому имён здесь нет вовсе.
holders() {
  local out="" pid uuid idx
  while IFS=, read -r pid uuid; do
    pid="${pid// /}"; uuid="${uuid// /}"
    idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep -F "$uuid" | cut -d, -f1 | tr -d ' ')
    case ",$CARDS," in *",$idx,"*) out="$out $pid" ;; esac
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
  echo "$out"
}

# Корень процесса-сервера от воркера: вверх по PPid до api_server.
root_of() {
  local p="$1" up
  for _ in 1 2 3 4; do
    up=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -z "$up" ] && break
    tr '\0' ' ' < /proc/$up/cmdline 2>/dev/null | grep -q "api_server" && { echo "$up"; return; }
    [ "$up" = "1" ] && break
    p="$up"
  done
  echo "$1"
}

stop_server() {
  local pids roots k full killed=0
  pids="$(holders)"
  [ -z "${pids// /}" ] && { say "карты $CARDS свободны, гасить некого"; return 0; }
  roots=$(for p in $pids; do root_of "$p"; done | sort -u)
  for k in $roots; do
    full=$(tr '\0' ' ' < /proc/$k/cmdline 2>/dev/null)
    # ПОДТВЕРЖДЕНИЕ ПО ПОЛНОЙ СТРОКЕ. Обрезанная строка уже давала ложный отказ: "--port" стоит
    # в самом конце, и `head -c 200` его не захватывал.
    case "$full" in
      *"--port $PORT"*) say "гашу pid=$k (подтверждено: несёт --port $PORT)"; kill "$k"; killed=1 ;;
      *) say "НЕ ТРОГАЮ pid=$k: в полной строке запуска нет '--port $PORT'" ;;
    esac
  done
  [ "$killed" = 1 ] || die "на картах $CARDS есть процессы, но ни один не опознан как сервер порта $PORT"
  # ЖДЁМ НЕ ПАМЯТЬ КАРТ, А СМЕРТЬ ЦЕПОЧКИ И СНЯТИЕ ЗАМКА. Первая редакция ждала только
  # освобождения карт -- и это стоило простоя: воркеры уже отпустили память, а обёртка
  # (`bash serve_...sh`), сам api_server и `tee` были живы и держали .serve_<порт>.lock.
  # Базовый пускач законно отказался поднимать второй экземпляр, окно tmux закрылось,
  # сессия исчезла, боевой остался лежать. Признак готовности -- ПУСТОЙ замок, а не пустая карта.
  local lock="$RUNDIR/.serve_${PORT}.lock" left
  for _ in $(seq 1 45); do
    left=""
    for q in $(cat "$lock" 2>/dev/null); do ps -p "$q" >/dev/null 2>&1 && left="$left $q"; done
    for q in $(holders); do left="$left $q"; done
    [ -z "${left// /}" ] && { say "карты и замок освободились"; return 0; }
    sleep 2
  done
  say "ещё живы:$left"
  for q in $left; do kill -9 "$q" 2>/dev/null; done
  sleep 5
  left=""
  for q in $(cat "$lock" 2>/dev/null); do ps -p "$q" >/dev/null 2>&1 && left="$left $q"; done
  [ -z "${left// /}" ] || die "не удалось освободить: $left"
  say "освободили принудительно"
}

case "${1:-}" in
  --log)    l="$(current_log)"; [ -n "$l" ] && echo "$l" || die "журналов serve_${PORT}_*.log нет"; exit 0 ;;
  --attach) tmux attach -t "$SESSION" 2>/dev/null || die "сессии '$SESSION' нет"; exit 0 ;;
  --stop)   stop_server; tmux kill-session -t "$SESSION" 2>/dev/null; say "остановлено"; exit 0 ;;
  --status)
    tmux has-session -t "$SESSION" 2>/dev/null && say "сессия '$SESSION': ЕСТЬ" || say "сессия '$SESSION': нет"
    say "карты $CARDS держат pid:$(holders)"
    say "журнал сервера: $(current_log || echo 'нет')"
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    say "порт $PORT отвечает: ${code:-000} $([ "$code" = 200 ] && echo '(жив)' || echo '(не отвечает)')"
    [ "$code" = 200 ] && curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" \
      | python3 -c "import sys,json;[print('[tmux-боевой] альяс:',d['id']) for d in json.load(sys.stdin)['data']]" 2>/dev/null
    exit 0 ;;
  --restart) RESTART=1 ;;
  "") ;;
  *) die "неизвестный ключ '$1' (см. шапку файла)" ;;
esac

[ -f "$LAUNCHER" ] || die "нет пускача $LAUNCHER"
command -v tmux >/dev/null || die "tmux не установлен"

# ------------------------------------------------------------------ предполётные проверки
if [ "${RESTART:-0}" = "1" ]; then
  stop_server
  tmux kill-session -t "$SESSION" 2>/dev/null
else
  busy="$(holders)"
  [ -n "${busy// /}" ] && {
    say "карты $CARDS ЗАНЯТЫ процессами:$busy"
    for p in $busy; do
      printf '  pid=%s : %s\n' "$p" "$(tr '\0' ' ' < /proc/$(root_of "$p")/cmdline 2>/dev/null | cut -c1-110)"
    done
    die "поднять поверх нельзя. Погасить намеренно: $0 --restart"
  }
  tmux has-session -t "$SESSION" 2>/dev/null && die "сессия '$SESSION' уже есть (--status / --attach / --restart)"
fi

# Порт: занятый порт уже дважды приводил к тому, что клиент молча говорил со СТАРЫМ сервером
# и получал невозможные числа. Проверяем ПОВЕДЕНИЕМ, а не наличием слушателя.
code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
[ "$code" = "200" ] && die "порт $PORT УЖЕ ОТВЕЧАЕТ (кто-то жив) -- разобраться, прежде чем поднимать"

# Затирание залипших замков делает САМ боевой пускач (одно тело на оба пути:
# и ручной подъём отсюда, и автостарт из сессии vllm, который зовёт его напрямую).

# ------------------------------------------------------------------ подъём
say "сессия '$SESSION', карты $CARDS, порт $PORT"
tmux new-session -d -s "$SESSION" -n boevoy \
  "cd '$RUNDIR' && PORT='$PORT' CARDS='$CARDS' bash '$LAUNCHER' 2>&1 | tee '$WRAPLOG'"
sleep 3
tmux has-session -t "$SESSION" 2>/dev/null || die "сессия не создалась, смотреть $WRAPLOG"

say "поднимается. журнал обёртки: $WRAPLOG"
say "первый подъём дольше: строится склад упакованных весов .w12_qwen38abl_prod"
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && { say "ПОДНЯЛСЯ за ~$((i*12)) с"; break; }
  tmux has-session -t "$SESSION" 2>/dev/null || die "сессия умерла, смотреть $WRAPLOG и $(current_log)"
  sleep 12
done
say "журнал сервера: $(current_log || echo 'ещё не создан')"
say "войти в окно: $0 --attach   (выход Ctrl-b d)"
[ "$code" = "200" ] || die "порт не ответил за 12 минут -- смотреть журналы выше"

# Ответ 200 доказательством НЕ является: у нас есть класс отказа «поднялся и врёт» (забор из
# восклицаний при промахе шаблонов пропуска в config.json). Поэтому здесь -- проверка ГЕНЕРАЦИЕЙ.
otvet=$(curl -s -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"huihui-qwen3.5-35b-a3b-abliterated-q8_0","messages":[{"role":"user","content":"Сколько будет 17*23? Ответь одним числом."}],"max_tokens":60,"temperature":0}' \
  | python3 -c "import sys,json;m=json.load(sys.stdin)['choices'][0]['message'];print(((m.get('content') or '')+(m.get('reasoning_content') or '')).strip()[:40])" 2>/dev/null)
case "$otvet" in
  *391*) say "ГЕЙТ ГЕНЕРАЦИИ ПРОЙДЕН: 17*23 -> $otvet" ;;
  *)     say "ГЕЙТ ГЕНЕРАЦИИ НЕ ПРОЙДЕН: ответ '$otvet' -- сервер отвечает, но НЕВЕРНО. Разбираться." ; exit 1 ;;
esac
