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
  --learning-rate 3e-5 \
  --weight-decay 0 \
  --embed-dim 512 \
  --num-heads 8 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --resume-reset-optimizer \
  --resume-skip-corr-head \
  --resume-reset-epoch \
  --resume-checkpoint "ckpts/c-4-stage_1_100_epoch_final.pt" \
  --save-path "ckpts/c-d-4-stage_2_30_epoch_final.pt_final.pt" \
  --best-save-path "ckpts/ckpts/c-d-4-stage_2_30_epoch_final.pt_final.pt" \
  --lr-scheduler "cosine" \
  --warmup 0 \
  --cosine-bump 0 \
  --architecture "vit" \
  --distributed \