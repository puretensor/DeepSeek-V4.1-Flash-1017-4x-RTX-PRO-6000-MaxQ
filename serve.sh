#!/usr/bin/env bash
# Start DeepSeek-V4.1-Flash-1017 on four GPUs with the BRRR image from ./build.sh.
#   ./serve.sh bench    single-slot benchmark arm: 32K context, 1 slot, CUDA graphs up to 8 tokens (the record config)
#   ./serve.sh serve    the config we serve from: 1M context, 16 slots (the image's defaults)
# env: MODEL (checkpoint dir, deepseek-ai/DeepSeek-V4.1-Flash-1017 rev 1bae0e10, on local NVMe), PORT, NAME
# The Engram tables are read from NVMe with io_uring, hence seccomp=unconfined and a local ext4/xfs model directory.
set -euo pipefail
MODE="${1:-bench}"
MODEL="${MODEL:?set MODEL to the checkpoint directory}"
PORT="${PORT:-8000}"; NAME="${NAME:-dsv41-brrr}"
IMAGE="${IMAGE:-deepseek-4.1-flash-1017-brrr:0bc4dbd0}"
EXTRA=()
[ "$MODE" = bench ] && EXTRA=(--max-model-len 32768 --max-num-seqs 1 --max-cudagraph-capture-size 8)
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --init --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --security-opt seccomp=unconfined \
  -e PORT="$PORT" -e ADAPTIVE=false \
  -v "$MODEL:/model:ro" -v deepseek-4.1-flash-1017-brrr-jit:/cache/jit \
  "$IMAGE" --served-model-name dsv41 --default-chat-template-kwargs '{"thinking":false}' "${EXTRA[@]}"
echo "waiting for http://127.0.0.1:$PORT (first start compiles kernels and can take 20 minutes; later starts about 2 to 7)"
until curl -sf "localhost:$PORT/v1/models" >/dev/null; do
  docker ps -q --filter "name=$NAME" | grep -q . || { docker logs --tail 20 "$NAME"; exit 1; }
  sleep 10
done
echo "ready. Give it a few minutes of traffic before timing it: the first runs after a start are slow."
