#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
    PYTHON="$VENV_DIR/Scripts/python.exe"
elif [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
else
    echo "Virtual environment not found at $VENV_DIR." >&2
    echo "Create it with: python -m venv $VENV_DIR" >&2
    exit 1
fi

run_training() {
    mode=$1
    output=$2
    "$PYTHON" train.py \
      --mode "$mode" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --image-size "$IMAGE_SIZE" \
      --output "$output"
}

run_training joint concept_bottleneck_joint.pt
run_training sequential concept_bottleneck_sequential.pt
run_training independent concept_bottleneck_independent.pt
