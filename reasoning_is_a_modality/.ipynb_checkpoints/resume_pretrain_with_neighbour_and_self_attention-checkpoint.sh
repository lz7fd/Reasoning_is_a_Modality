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
  --resume-checkpoint "ckpts/v8_checkpoint_recurrent_4_self_neighbor_attn_resume_lr_3e-5_pre_warmup.pt" \
  --save-path "ckpts/v8_checkpoint_recurrent_4_neighbor_attn_30_epoch_final.pt" \
  --best-save-path "ckpts/v8_checkpoint_recurrent_4_neighbor_attn_30_epoch_best.pt" \
  --lr-scheduler "cosine" \
  --warmup 0 \
  --cosine-bump 0 \
  --architecture "vit" \
  --distributed \