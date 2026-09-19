#!/bin/bash
# ВХОД СЛУЖБЫ boevoy-vllm: поднять tmux-СЕРВЕР. Сессию `vllm` создаёт сам tmux по строке
# `new-session` из ~/.tmux.conf -- поэтому здесь НЕЛЬЗЯ создавать её вручную: получится
# гонка и "duplicate session" (проверено отказом 26.08, служба падала, хотя боевой вставал).
# Идемпотентно: если сервер уже поднят, ничего не делаем.
set -uo pipefail
if tmux has-session -t vllm 2>/dev/null; then
  echo "[start] сессия vllm уже есть -- не трогаю (tmux attach -t vllm)"
  exit 0
fi
tmux start-server 2>/dev/null
for i in $(seq 20); do
  tmux has-session -t vllm 2>/dev/null && { echo "[start] сессия vllm поднята tmux-сервером"; exit 0; }
  sleep 1
done
echo "[start] сессия vllm не появилась за 20 с -- смотрите tmux ls и ~/.tmux.conf"
exit 0
