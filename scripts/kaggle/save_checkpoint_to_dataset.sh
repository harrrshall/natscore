#!/usr/bin/env bash
# Pull `latest.pt` from a finished Kaggle training kernel and push it to the
# `natscore-checkpoint` Kaggle Dataset so the *next* run can resume from it.
#
# Usage:
#   bash scripts/kaggle/save_checkpoint_to_dataset.sh
#
# Idempotent. Run this every time the Kaggle training kernel finishes (or hits
# the 9-h wall) — before pushing the next version.

set -euo pipefail

KERNEL_REF="harshalsinghcn/natscore-train-on-t4"
DATASET_REF="harshalsinghcn/natscore-checkpoint"
WORK_DIR="/tmp/natscore-ckpt-sync"

# 1. Pull whatever the kernel saved into /kaggle/working/.
rm -rf "$WORK_DIR" && mkdir -p "$WORK_DIR/pull"
echo "==> Pulling $KERNEL_REF output ..."
kaggle kernels output "$KERNEL_REF" -p "$WORK_DIR/pull"

# 2. Locate latest.pt.
LATEST="$WORK_DIR/pull/outputs/natscore-small-v0-kaggle/latest.pt"
if [[ ! -f "$LATEST" ]]; then
    echo "ERROR: $LATEST not found in kernel output." >&2
    echo "       Kernel may have died before the first 500-step checkpoint." >&2
    ls -R "$WORK_DIR/pull" >&2 || true
    exit 1
fi
echo "==> Found checkpoint: $(du -h "$LATEST" | cut -f1)"

# 3. Stage the dataset directory.
mkdir -p "$WORK_DIR/dataset"
cp "$LATEST" "$WORK_DIR/dataset/latest.pt"

# 4. First-time bootstrap vs subsequent version.
if kaggle datasets status "$DATASET_REF" >/dev/null 2>&1; then
    echo "==> Dataset $DATASET_REF exists; pushing new version ..."
    # Need a dataset-metadata.json with the existing ref.
    cat > "$WORK_DIR/dataset/dataset-metadata.json" <<EOF
{
  "title": "NatScore training checkpoint",
  "id": "$DATASET_REF",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    kaggle datasets version -p "$WORK_DIR/dataset" -m "checkpoint from $(date -u +%Y-%m-%dT%H:%MZ)"
else
    echo "==> Dataset $DATASET_REF does not exist; creating ..."
    cat > "$WORK_DIR/dataset/dataset-metadata.json" <<EOF
{
  "title": "NatScore training checkpoint",
  "id": "$DATASET_REF",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    kaggle datasets create -p "$WORK_DIR/dataset" --dir-mode zip
fi

echo
echo "Done. Next push of the training kernel will mount the new checkpoint at"
echo "  /kaggle/input/natscore-checkpoint/latest.pt"
echo "and the resume scaffold cell will copy it into the run dir for OnlineTrainer."
