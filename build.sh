#!/usr/bin/env bash
# Build the serving image: Fraser Price's BRRR recipe (Apache-2.0) at a pinned Hugging Face revision, unmodified.
# A cold build pulls the Local Inference Lab base image and compiles the sm_120a kernels. Needs docker buildx and `hf`.
# usage: ./build.sh        (env: REV, DEST)
set -euo pipefail
REV="${REV:-0bc4dbd0a15458e0957b899d72c7a5089e3335a5}"
DEST="${DEST:-$HOME/.cache/dsv41-brrr}"
TAG="deepseek-4.1-flash-1017-brrr:${REV:0:8}"
hf download fraserprice/DeepSeek-4.1-Flash-1017-BRRR --revision "$REV" --local-dir "$DEST/$REV" >/dev/null
docker build -t "$TAG" "$DEST/$REV"
echo "built $TAG"
