#!/usr/bin/env bash
# Download and extract the LFW dataset + official pairs.txt
# Usage: bash data/download_lfw.sh

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
LFW_DIR="$DATA_DIR/lfw"

echo "==> Creating $LFW_DIR"
mkdir -p "$LFW_DIR"

echo "==> Downloading LFW images (~173 MB)..."
curl -L -o "$DATA_DIR/lfw.tgz" \
  "http://vis-www.cs.umass.edu/lfw/lfw.tgz"

echo "==> Extracting..."
tar -xzf "$DATA_DIR/lfw.tgz" -C "$DATA_DIR"
rm "$DATA_DIR/lfw.tgz"

echo "==> Downloading pairs.txt (standard 6000-pair split)..."
curl -L -o "$LFW_DIR/pairs.txt" \
  "http://vis-www.cs.umass.edu/lfw/pairs.txt"

echo ""
echo "Done. LFW images: $LFW_DIR"
echo "      Pairs file: $LFW_DIR/pairs.txt"
echo ""
echo "Update configs/experiment.yaml:"
echo "  data.lfw_root:  \"$LFW_DIR\""
echo "  data.lfw_pairs: \"$LFW_DIR/pairs.txt\""
