#!/bin/bash
# Sustained attention on each GPU in turn. Expect the cards to match; they did not on this box.
D=$(cd "$(dirname "$0")" && pwd)
cd "$D/../../.."
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"; fi
export CC=${CC:-/usr/bin/gcc} CXX=${CXX:-/usr/bin/g++} CUDAHOSTCXX=${CUDAHOSTCXX:-/usr/bin/g++}
export TORCH_CUDA_ARCH_LIST=7.0
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
find fa2_sm70/_build -name lock -type f -delete 2>/dev/null
for g in ${GPUS:-0 1 2 3}; do CUDA_VISIBLE_DEVICES=$g python -u "$D/percard_attn.py" 2>&1 | grep -E "^GPU"; done
echo PERCARD_DONE
