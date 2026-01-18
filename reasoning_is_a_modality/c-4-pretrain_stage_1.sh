#!/usr/bin/env bash
set -e

torchrun --nproc_per_node=8 offline_train_ARC.py \
  --epochs 30 \
  --depth 10 \
  --batch-size 32 \
  --image-size 64 \
  --patch-size 2 \
  --recurrent-depth 4 \
  --max-ctx-tokens 4 \
  --patch-local-attn "none" \
  --learning-rate 1e-4 \
  --weight-decay 0 \
  --embed-dim 512 \
  --num-heads 8 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --resume-skip-corr-head \
  --save-path "ckpts/c-4-stage_1_100_epoch_final.pt" \
  --best-save-path "ckpts/c-4-stage_1_100_epoch_best.pt" \
  --lr-scheduler "cosine" \
  --warmup 0 \
  --cosine-bump 0 \
  --architecture "vit" \
  --distributed \