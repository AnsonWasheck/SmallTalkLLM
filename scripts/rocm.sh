#!/usr/bin/env bash
# Run any smalltalk-ai command on the AMD GPU.
#
#   bash scripts/rocm.sh scripts/check_device.py
#   bash scripts/rocm.sh scripts/train.py --config configs/train/stage1_4m.yaml
#
# Why this wrapper exists (verified on Radeon 8060S / gfx1151, ROCm 7.1 system
# packages + torch 2.10.0+rocm7.0 wheel):
#   The wheel bundles its own libhsa-runtime64.so (ROCm 7.0). Against a 7.1
#   kernel/driver stack that bundled runtime SEGFAULTS on the first kernel launch
#   -- while still reporting cuda.is_available() == True and the correct device
#   name, so it looks healthy right up until it dies. LD_PRELOAD-ing the system
#   HSA runtime resolves it. Drop this wrapper once the wheel's ROCm matches the
#   distro's.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv-rocm}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no interpreter at $PY (create the venv first)" >&2; exit 1; }

SYS_HSA="${SYS_HSA:-$(ls /usr/lib/x86_64-linux-gnu/libhsa-runtime64.so* 2>/dev/null | head -1)}"
if [ -n "$SYS_HSA" ]; then
    export LD_PRELOAD="${SYS_HSA}${LD_PRELOAD:+:$LD_PRELOAD}"
else
    echo "warning: no system libhsa-runtime64.so found; expect a segfault on gfx1151" >&2
fi

# gfx1151 is new; uncomment if you hit 'invalid device function'.
# export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Silence the harmless missing-ids lookup and keep SDPA on the stable path.
export AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-0}"

exec "$PY" "$@"
