#!/bin/bash
# Launch multi-GPU training with power management
# Usage: ./launch_multi_gpu.sh [single|multi] [extra args...]
#   ./launch_multi_gpu.sh single --resume        # single-day trainer on 2 GPUs
#   ./launch_multi_gpu.sh multi --resume          # multi-day trainer on 2 GPUs
#   ./launch_multi_gpu.sh multi --resume --batch-days 48

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIF="/media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/ai_atmosphere.sif"
BIND="/media/airlab/ROCSTOR:/media/airlab/ROCSTOR"
NUM_GPUS=2

# --- GPU Power Management ---
# Cap each GPU to 200W (67% of 300W max) to prevent system shutdown.
# Two GPUs at 200W = 400W total, well within typical workstation PSU headroom.
# Adjust up (max 300) if thermals/PSU allow, or down if still unstable.
GPU_POWER_LIMIT_W=200

echo "=== GPU Power Management ==="
echo "Setting power limit to ${GPU_POWER_LIMIT_W}W per GPU..."
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    sudo nvidia-smi -i "$gpu_id" -pl "$GPU_POWER_LIMIT_W" 2>/dev/null || \
        echo "WARNING: Could not set power limit on GPU $gpu_id (need sudo)"
done
nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader
echo ""

# --- Select training mode ---
MODE="${1:-multi}"
shift 2>/dev/null || true

case "$MODE" in
    single)
        TRAIN_SCRIPT="train.py"
        ;;
    multi)
        TRAIN_SCRIPT="train_multi_days.py"
        ;;
    *)
        echo "Usage: $0 [single|multi] [extra args...]"
        exit 1
        ;;
esac

echo "=== Launching $TRAIN_SCRIPT on $NUM_GPUS GPUs ==="
echo "Extra args: $*"
echo ""

# --- Launch with torchrun inside Apptainer ---
apptainer exec --nv --env PYTHONNOUSERSITE=1 \
    --bind "$BIND" \
    "$SIF" \
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        "$SCRIPT_DIR/$TRAIN_SCRIPT" "$@"

# --- Restore default power limits ---
echo ""
echo "=== Restoring default GPU power limits ==="
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    sudo nvidia-smi -i "$gpu_id" -pl 300 2>/dev/null || true
done
