#!/usr/bin/env bash
# Push scripts/kaggle/train_natscore_t4.ipynb to Kaggle with the repo-side
# kernel-metadata.train.json (which has dataset_sources for resume).
#
# Important: --accelerator nvidiaTeslaT4 is included because CLI defaults to
# P100 and Kaggle's PyTorch wheel no longer supports sm_60 (P100). If Kaggle
# silently re-assigns P100, you must also flip the accelerator in the editor UI
# (the override happens to stick across re-pushes once set there).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="/tmp/natscore-train-push"

rm -rf "$STAGE" && mkdir -p "$STAGE"
cp "$REPO_ROOT/scripts/kaggle/kernel-metadata.train.json" "$STAGE/kernel-metadata.json"
cp "$REPO_ROOT/scripts/kaggle/train_natscore_t4.ipynb"    "$STAGE/natscore-train-on-t4.ipynb"

echo "==> Staging: $STAGE"
ls -la "$STAGE"
echo
echo "==> Pushing ..."
kaggle kernels push -p "$STAGE" --accelerator nvidiaTeslaT4
