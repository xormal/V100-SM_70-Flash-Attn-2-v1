#!/usr/bin/env bash
# КАКОЙ ПУСКАЧ СИДИТ НА ЗАДАННЫХ КАРТАХ. Единственный законный способ адресовать сервер к снятию.
#
# ПОЧЕМУ ИМЕННО СНИЗУ ВВЕРХ. 16.08.2026 я снял БОЕВОЙ сервер, взяв PID сверху -- `pgrep -f
# api_server | head -1` -- и «удостоверив» его проверкой FA2SM70_PATH в environ. Различитель был
# ложным ПО ПОСТРОЕНИЮ: боевой тоже работает на наших ядрах, признак у них общий. Имя, шаблон,
# переменные окружения -- всё это ОБЩЕЕ у серверов на этой машине и не различает ничего.
# Различает ровно одно: НА КАКОЙ КАРТЕ ЛЕЖИТ ЕГО ПАМЯТЬ. Это и берётся у железа.
#
#   tools/pid_na_kartah.sh 0,1      -> список PID пускачей, чья память на картах 0 и 1
set -u
want="${1:?укажите номера карт через запятую, например 0,1}"
declare -A uuid2idx
while IFS=, read -r idx uuid; do
  uuid2idx["$(echo "$uuid" | tr -d ' ')"]="$(echo "$idx" | tr -d ' ')"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

pids=""
while IFS=, read -r pid uuid; do
  pid="$(echo "$pid" | tr -d ' ')"; uuid="$(echo "$uuid" | tr -d ' ')"
  idx="${uuid2idx[$uuid]:-?}"
  case ",$want," in *",$idx,"*) pids="$pids $pid";; esac
done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)

# ОТ РАБОЧЕГО ПРОЦЕССА ВВЕРХ ДО ТОГО, КТО ДЕРЖИТ КАРТЫ. Имена промежуточных процессов
# ПЕРЕИМЕНОВАНЫ движком (VLLM::Worker_TP0, VLLM::EngineCore), поэтому искать по имени на каждом
# шаге нельзя -- надо подняться по всей цепи.
#
# ВОЗВРАЩАЕТСЯ `api_server`, А НЕ ОБОЛОЧКА ПУСКАЧА. Пускач кончается `exec ... | tee`, и `exec` в
# КОНВЕЙЕРЕ процесс НЕ заменяет: оболочка живёт рядом, но сервер ей не подчинён. ЗАМЕРЕНО 16.08:
# снятие оболочки (1889114) оставило сервер (1900295) и обоих рабочих живыми и с занятой памятью.
# Снимать надо ТО, ЧТО ДЕРЖИТ КАРТЫ.
for p in $pids; do
  cur="$p"; srv=""; wrap=""
  for _ in 1 2 3 4 5 6 7 8; do
    cmd="$(tr '\0' ' ' < "/proc/$cur/cmdline" 2>/dev/null)"
    case "$cmd" in *api_server*) [ -z "$srv" ] && srv="$cur";; esac
    case "$cmd" in *serve_*|*restart_*|*start_model*) wrap="$cur";; esac
    parent="$(awk '/^PPid:/{print $2}' "/proc/$cur/status" 2>/dev/null)"
    [ -z "$parent" ] && break
    [ "$parent" = "1" ] && break
    cur="$parent"
  done
  [ -n "$srv" ] && echo "$srv"                      # сервер -- ПЕРВЫМ, его и снимать
  [ -n "$wrap" ] && [ "$wrap" != "$srv" ] && echo "$wrap"   # оболочка -- следом, чтобы не осталась
done | sort -u
