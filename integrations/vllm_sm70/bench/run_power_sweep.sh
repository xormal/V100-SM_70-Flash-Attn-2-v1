#!/bin/bash
# Find the power cap at which the weak card stops losing clocks, WITHOUT giving up throughput.
# Needs sudo for nvidia-smi -pl. Usage: GPU=2 LEVELS="220 240 260 280 300" ./run_power_sweep.sh
D=$(cd "$(dirname "$0")" && pwd); cd "$D/../../.."
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"; fi
export CC=${CC:-/usr/bin/gcc} CXX=${CXX:-/usr/bin/g++} CUDAHOSTCXX=${CUDAHOSTCXX:-/usr/bin/g++}
export TORCH_CUDA_ARCH_LIST=7.0
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
G=${GPU:-2}
for pl in ${LEVELS:-220 250 280 300}; do
  printf '%s\n' "$SUDO_PW" | sudo -S -p '' nvidia-smi -pl $pl -i $G >/dev/null 2>&1
  sleep 20                                   # let the card cool between levels
  PL=$pl CUDA_VISIBLE_DEVICES=$G SECONDS_RUN=${SECONDS_RUN:-150} python -u "$D/power_sweep.py" 2>&1 | grep "^pl="
done
echo SWEEP_DONE
