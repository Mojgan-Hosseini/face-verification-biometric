#!/usr/bin/env bash
# Download and extract the CFP dataset (Celebrities in Frontal-Profile)
# Usage: bash data/download_cfp.sh
#
# Note: CFP requires registration at http://www.cfpw.io/
# After registering you will receive a download link.
# Paste it below as CFP_URL or pass it as an argument:
#   bash data/download_cfp.sh "https://your-download-link/cfp-dataset.zip"

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
CFP_DIR="$DATA_DIR/cfp-dataset"
CFP_URL="${1:-}"

if [ -z "$CFP_URL" ]; then
  echo "Usage: bash data/download_cfp.sh <download_url>"
  echo ""
  echo "To get the URL:"
  echo "  1. Visit http://www.cfpw.io/"
  echo "  2. Register with your institutional email"
  echo "  3. You will receive a download link by email"
  echo ""
  echo "Alternatively, if you have the zip already:"
  echo "  unzip cfp-dataset.zip -d data/"
  exit 1
fi

echo "==> Downloading CFP dataset..."
curl -L -o "$DATA_DIR/cfp-dataset.zip" "$CFP_URL"

echo "==> Extracting..."
unzip -q "$DATA_DIR/cfp-dataset.zip" -d "$DATA_DIR"
rm "$DATA_DIR/cfp-dataset.zip"

echo ""
echo "Done. CFP dataset: $CFP_DIR"
echo ""
echo "Expected structure:"
echo "  $CFP_DIR/Data/Images/<subject_id>/frontal/*.jpg"
echo "  $CFP_DIR/Data/Images/<subject_id>/profile/*.jpg"
echo "  $CFP_DIR/Protocol/Split/FP/*.mat"
