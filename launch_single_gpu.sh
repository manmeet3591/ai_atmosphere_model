#!/bin/bash
# Launch single-GPU training (no DDP, no torchrun needed)
# Usage: ./launch_single_gpu.sh [single|multi] [--gpu 0|1] [extra args...]
#   ./launch_single_gpu.sh single --gpu 1 --resume
#   ./launch_single_gpu.sh multi --gpu 1 --resume --batch-days 48

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIF="/media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/ai_atmosphere.sif"
BIND="/media/airlab/ROCSTOR:/media/airlab/ROCSTOR"

MODE="${1:-multi}"
shift 2>/dev/null || true

# Parse --gpu flag
GPU_ID=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ID="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "$MODE" in
    single) TRAIN_SCRIPT="train.py" ;;
    multi)  TRAIN_SCRIPT="train_multi_days.py" ;;
    *)      echo "Usage: $0 [single|multi] [--gpu 0|1] [extra args...]"; exit 1 ;;
esac

echo "=== Launching $TRAIN_SCRIPT on GPU $GPU_ID ==="
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

CUDA_VISIBLE_DEVICES="$GPU_ID" apptainer exec --nv --env PYTHONNOUSERSITE=1 \
    --bind "$BIND" \
    "$SIF" \
    python3 "$SCRIPT_DIR/$TRAIN_SCRIPT" "${EXTRA_ARGS[@]}"
