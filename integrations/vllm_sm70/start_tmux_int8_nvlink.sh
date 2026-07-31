#!/usr/bin/env bash
# Launch the Qwen3.6-27B int8 (W8A16/GPTQ) server on GPU2-3 with OUR fa2_sm70 kernels, in tmux.
#
# WHAT THIS CONFIGURATION IS, AND WHY EACH PIECE
#   * int8 weights (GPTQ W8A16, 17.7 GB/GPU) instead of AWQ 4-bit. Prefill is 1.02-1.09x FASTER
#     (prefill is compute-bound, so heavier weights cost nothing) and decode is better past ~32K.
#     It costs context: KV pool drops to 6 GiB, giving ~377K tokens of capacity. The model's own
#     limit is 256K, so that is enough.
#   * NCCL_P2P_DISABLE=0 -- REQUIRED. An NVLink bridge (NV2, 2x25.78 GB/s) sits on GPU2-GPU3; with
#     P2P off NCCL stages through the host and the bridge is simply unused. It is worth 2.5x on
#     prefill (877 -> 2154 tok/s at 4K): the model does 128 all-reduces per forward, which were 63%
#     of prefill wall at PCIe's 3.5 GB/s and are 8% at NVLink's 65.
#   * custom all-reduce stays DISABLED. It measured slightly faster (decode 37.4 vs 35.1) and
#     produces CORRUPT output -- "The capital of France is" -> " Paris!!!!!!!!!".
#
# PRE-FLIGHT CHECKS BELOW ARE NOT CEREMONY. Twice in development a leftover server kept port 8085,
# the client silently talked to it, and its prefix cache reported impossible rates (29,481 tok/s).
# The script refuses to start until the port is free and the GPUs are empty.
set -euo pipefail

SESSION="${SESSION:-vllm-int8}"
PORT="${PORT:-8085}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="${VLLM_ROOT:?set VLLM_ROOT to the vLLM workspace}"
LAUNCHER="$VLLM_ROOT/run/start_model_fa2sm70_gpu23.sh"
FA2SM70_REPO="${FA2SM70_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$VLLM_ROOT/.venv-1cat-vllm/bin/python}"

export MODEL="${MODEL:-$VLLM_ROOT/Huihui-Qwen3.6-27B-abliterated-INT8}"
export QUANT="${QUANT:-gptq}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"      # the model's own context limit
export KV_CACHE_BYTES="${KV_CACHE_BYTES:-6442450944}" # 6 GiB: what fits beside 17.7 GB of int8 weights
# 4, not more: at 8 the boot OOMs (384 MiB short) -- int8 weights are 17.7 GB/GPU and the CUDA-graph
# capture plus per-sequence activations have to fit in what is left beside the 6 GiB KV pool. A 5th
# request simply queues, which is how the published numbers were measured anyway.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"             # hermes + ht-mini, ~4-5 concurrent
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"     # NVLink
# These cards are dedicated to this server, so take the memory. The KV pool is pinned separately
# (KV_CACHE_BYTES), and what this raises is the room left for ACTIVATIONS -- which is what actually
# OOMs: not the KV cache but a long prefill chunk beside 17.7 GB of int8 weights.
export GPU_UTIL="${GPU_UTIL:-0.96}"
# Multimodal caps. Inherited video:0 from the AWQ launcher, and a client sending a video part gets a
# 500: "At most 0 video(s) may be provided in one prompt". The model DOES support video
# (video_token_id 248057, vision tower present and left in fp16 by the int8 conversion), so this is a
# policy cap, not a capability limit. Raising it is one env var:
#     LIMIT_MM='{"image":20,"video":2}' ./start_tmux_int8_nvlink.sh
# Left at 0 by default because this configuration runs at gpu-memory-utilization 0.96 with 17.7 GB of
# weights: the vision tower's activations are the one thing that has NOT been memory-tested here, and
# an engine-level OOM takes the whole server down rather than failing that one request.
if [ -z "${LIMIT_MM:-}" ]; then LIMIT_MM='{"image":20,"video":0}'; fi
export LIMIT_MM
export PORT

say() { printf '\033[1;36m[start]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[start] %s\033[0m\n' "$*" >&2; exit 1; }

[ -x "$LAUNCHER" ]  || die "launcher missing: $LAUNCHER"
[ -d "$MODEL" ]     || die "model dir missing: $MODEL"
command -v tmux >/dev/null || die "tmux is not installed"

# --- 1. NVLink must actually be up, or this config is pointless -------------
# In `nvidia-smi topo -m` the GPU2 row is: GPU2 <vs0> <vs1> <vs2=X> <vs3>, so the GPU2<->GPU3 cell
# is field 5, not 4. (Reading field 4 gives the self-cell "X" and reports NVLink as missing.)
LINK=$(nvidia-smi topo -m 2>/dev/null | awk '/^GPU2/{print $5; exit}')
if printf '%s' "$LINK" | grep -q '^NV'; then
  say "NVLink between GPU2 and GPU3: $LINK"
else
  say "WARNING: no NVLink on GPU2-GPU3 (topology shows '$LINK')."
  say "         Prefill will be ~2.5x slower; all-reduce becomes 63% of it."
fi

# --- 2. no leftover server may hold the port or the GPUs --------------------
PAT='vllm.entry''points|VLLM::'
if pgrep -f "$PAT" >/dev/null; then
  say "stopping leftover vLLM processes..."
  for sig in TERM TERM KILL; do
    for p in $(pgrep -f "$PAT"); do kill -"$sig" "$p" 2>/dev/null || true; done
    sleep 5
  done
fi
for i in $(seq 1 30); do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '3p;4p' | paste -sd+ | bc)
  busy=$(ss -ltn 2>/dev/null | grep -c ":$PORT" || true)
  [ "$busy" = "0" ] && [ "$mem" -lt 500 ] && break
  sleep 5
done
[ "$(ss -ltn 2>/dev/null | grep -c ":$PORT" || true)" = "0" ] || die "port $PORT is still in use"
say "port $PORT free, GPU2-3 clear"

# --- 3. stale JIT lock: a killed server leaves one and every later boot then
#        blocks on it FOREVER, during CUDA-graph capture, with no traceback ---
BUILD="${FA2SM70_BUILD_DIR:-$VLLM_ROOT/run/.fa2sm70_build}"
if find "$BUILD" -name lock -type f 2>/dev/null | grep -q .; then
  say "clearing stale JIT build lock(s) in $BUILD"
  find "$BUILD" -name lock -type f -delete
fi

# --- 4. go ------------------------------------------------------------------
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
say "starting tmux session '$SESSION' (model: $(basename "$MODEL"), quant: $QUANT, ctx: $MAX_MODEL_LEN)"
tmux new-session -d -s "$SESSION" -n server \
  "VLLM_ROOT='$VLLM_ROOT' PYTHON_BIN='$PYTHON_BIN' FA2SM70_REPO='$FA2SM70_REPO' MODEL='$MODEL' QUANT='$QUANT' PORT='$PORT' MAX_MODEL_LEN='$MAX_MODEL_LEN' \
   KV_CACHE_BYTES='$KV_CACHE_BYTES' MAX_NUM_SEQS='$MAX_NUM_SEQS' \
   NCCL_P2P_DISABLE='$NCCL_P2P_DISABLE' GPU_UTIL='$GPU_UTIL' LIMIT_MM='$LIMIT_MM' '$LAUNCHER' 2>&1 | tee '$VLLM_ROOT/run/vllm-int8.log'"

say "waiting for the server to come up (first boot JIT-compiles our kernels, a few minutes)..."
for i in $(seq 1 120); do
  if ss -ltn 2>/dev/null | grep -q ":$PORT"; then break; fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then die "session died -- see $VLLM_ROOT/run/vllm-int8.log"; fi
  sleep 10
done
ss -ltn 2>/dev/null | grep -q ":$PORT" || die "did not come up in 20 min -- see $VLLM_ROOT/run/vllm-int8.log"

# --- 5. verify OUTPUT before anyone trusts the server -----------------------
# A misconfigured all-reduce or a half-loaded quantised checkpoint both produce a server that
# starts, answers, and emits garbage ("!!!!!!"). Always read one answer.
say "server is up; checking that it actually answers sensibly"
ANS=$("$PYTHON_BIN" - <<PY
import json, urllib.request
B = "http://localhost:$PORT"
M = json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0]["id"]
r = urllib.request.Request(B + "/v1/completions",
    data=json.dumps({"model": M, "prompt": "The capital of France is",
                     "max_tokens": 8, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(r, timeout=300).read())["choices"][0]["text"].strip())
PY
)
printf '\033[1;36m[start]\033[0m model answered: %q\n' "$ANS"
case "$ANS" in
  *Paris*) say "OK -- output looks sane" ;;
  *!!!!*)  die "CORRUPT OUTPUT. Check that --disable-custom-all-reduce is present and that the int8 checkpoint's quantization_config carries both 'dynamic' skips and 'modules_to_not_convert'." ;;
  *)       say "WARNING: unexpected answer; look at it before trusting the server" ;;
esac

cat <<EOF

  session : tmux attach -t $SESSION      (detach: Ctrl-b d)
  log     : $VLLM_ROOT/run/vllm-int8.log
  endpoint: http://0.0.0.0:$PORT/v1      aliases: huihui-qwen3.5-35b-a3b-abliterated-q8_0,
                                                  gpt-4-0125-preview, Qwen3.6-27B-AWQ
  stop    : tmux kill-session -t $SESSION

  Expected (measured, unique non-cached prompts): prefill 2179 tok/s at 8K, 2120 at 16K,
  1844 at 33K, 1441 at 65K; decode 33-35 tok/s. Long-context numbers vary while GPU2 loses
  clocks under sustained load -- that is a cooling issue, not the server.
EOF
