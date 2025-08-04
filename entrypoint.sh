#!/bin/bash
set -e

# Default values
NUM_GPUS=${NUM_GPUS:-1}
CONFIG_FILE=${CONFIG_FILE:-/config/config.toml}

# Set NCCL environment variables for RTX 4000 series compatibility
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# If no arguments provided, run the training command
if [ $# -eq 0 ]; then
    echo "Starting diffusion-pipe training with config: $CONFIG_FILE"
    echo "Using $NUM_GPUS GPU(s)"
    exec deepspeed --num_gpus=$NUM_GPUS train.py --deepspeed --config "$CONFIG_FILE" "$@"
else
    # If arguments provided, execute them directly
    exec "$@"
fi 