#!/bin/bash
# Launch 4 independent vLLM servers, one per GPU.
#
# Each server:
#   - Binds to a separate port (8000-8003)
#   - Uses CUDA_VISIBLE_DEVICES to pin to one GPU
#   - Enables prefix caching (reuses KV cache for shared prompt prefixes)
#   - Enables Flash Attention via the default attention backend
#   - Uses 90% of GPU memory for KV cache
#
# Prerequisites:
#   pip install vllm
#   Model will auto-download from HuggingFace on first run
#
# Usage:
#   bash experiments/launch_vllm_servers.sh           # 4 GPUs, Qwen3-4B
#   bash experiments/launch_vllm_servers.sh 2          # 2 GPUs only
#   bash experiments/launch_vllm_servers.sh 4 Qwen/Qwen3-8B  # Different model
#
# Stop all:
#   bash experiments/launch_vllm_servers.sh stop

set -e

NUM_GPUS=${1:-4}
MODEL=${2:-"Qwen/Qwen3-4B"}
BASE_PORT=8000

if [ "$1" = "stop" ]; then
    echo "Stopping all vLLM servers..."
    pkill -f "vllm.entrypoints.openai.api_server" || echo "No servers running."
    exit 0
fi

echo "========================================="
echo " OTllm vLLM Cluster Launcher"
echo " GPUs: $NUM_GPUS"
echo " Model: $MODEL"
echo " Ports: $BASE_PORT - $((BASE_PORT + NUM_GPUS - 1))"
echo "========================================="
echo ""

# Check vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm not installed. Run:"
    echo "  pip install vllm"
    exit 1
fi

# Launch one server per GPU
PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "Starting vLLM on GPU $i, port $PORT..."

    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --max-model-len 4096 \
        --port "$PORT" \
        > "vllm_gpu${i}.log" 2>&1 &

    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}, log: vllm_gpu${i}.log"
done

echo ""
echo "Waiting for servers to be ready..."

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    READY=false
    for attempt in $(seq 1 60); do
        if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "  GPU $i (port $PORT): READY"
            READY=true
            break
        fi
        sleep 2
    done
    if [ "$READY" = false ]; then
        echo "  GPU $i (port $PORT): TIMEOUT - check vllm_gpu${i}.log"
    fi
done

echo ""
echo "========================================="
echo " All servers launched."
echo " Run experiments with:"
echo "   python experiments/gpu_cluster.py --gpus $NUM_GPUS"
echo ""
echo " Stop servers with:"
echo "   bash experiments/launch_vllm_servers.sh stop"
echo "========================================="

# Wait for all background processes
wait
