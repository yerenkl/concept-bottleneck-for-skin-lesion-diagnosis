#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

python train.py \
  --mode joint \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --output concept_bottleneck_joint.pt

python train.py \
  --mode sequential \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --output concept_bottleneck_sequential.pt

python train.py \
  --mode independent \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --output concept_bottleneck_independent.pt
