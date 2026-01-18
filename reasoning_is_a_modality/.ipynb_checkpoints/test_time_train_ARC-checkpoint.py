import argparse
from copy import deepcopy
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from utils.args import parse_args
from utils.distribution import init_distributed_mode
from utils.load_model import load_model_only, load_optimizer

from src.ARC_loader import build_dataloaders, IGNORE_INDEX
from utils.eval_utils_ttt import generate_predictions, get_eval_rot_transform_resolver


def _format_eta(seconds: float) -> str:
    total_seconds = int(max(seconds, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ttt_once(model, device, distributed, rank, train_loader, train_sampler, eval_loader, cur_attempt_idx):
    autocast_device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cuda"
    is_main_process = (not distributed) or rank == 0

    global_start = time.time()
    previous_total_steps = 0
    optimizer, scaler, scheduler = load_optimizer(
        model=model, args=args, device=device, distributed=distributed, rank=rank
    )
    # early stop streak
    success_streak = 0
    try:
        for epoch in range(0, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            sample_count = 0
            total_batches = len(train_loader)
            epoch_start = time.time()
            train_exact = 0
            train_examples = 0
            running_ce_loss = 0.0
            running_mid_loss = 0.0
            running_corr_loss = 0.0
            # ---- pixel-level diagnostics (epoch accumulators) ----
            running_valid_pix = 0.0
            running_wrong_pix = 0.0
            running_correct_pix = 0.0
            running_ce_wrong_sum = 0.0
            running_ce_correct_sum = 0.0
            running_margin_wrong_sum = 0.0  # sum of (logit_true - logit_pred) over wrong pixels

            for step, batch in enumerate(train_loader, 1):
                inputs = batch["inputs"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                task_ids = batch["task_ids"].to(device)
                contexts = batch["contexts"].to(device)

                optimizer.zero_grad(set_to_none=True)
                
                # Use automatic mixed precision
                with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                    # forward: final, middle logits, and encoded for correctness head
                    logits, logits_middle, encoded, num_prefix = model(
                        inputs,
                        contexts,
                        task_ids,
                        attention_mask=attention_mask,
                        middle_loss=True,
                        return_encoded=True,
                    )
                
                    num_colors = logits.size(1)
                    batch_size = logits.size(0)
                
                    # --- 1) Compute exactness mask from FINAL output ---
                    preds = logits.argmax(dim=1)  # [B, H, W]
                    is_exact_tensor = torch.zeros(batch_size, dtype=torch.bool, device=device)
                
                    final_ce_losses = []
                    mid_ce_losses = []
                
                    # reward weight for correct grids
                    w_correct_ce = getattr(args, "ce_reward_weight_correct", 0.1)
                
                    for b in range(batch_size):
                        target_b = targets[b]   # [H, W]
                        pred_b   = preds[b]     # [H, W]
                
                        valid = target_b != IGNORE_INDEX
                        if valid.any():
                            exact = torch.equal(pred_b[valid], target_b[valid])
                        else:
                            exact = False
                
                        is_exact_tensor[b] = exact
                
                        # final-layer CE per sample
                        logits_b = logits[b]  # [C, H, W]
                        logits_flat_b = logits_b.permute(1, 2, 0).reshape(-1, num_colors)
                        target_flat_b = target_b.view(-1)
                        
                        # per-pixel CE (ignored pixels get 0)
                        ce_pix = F.cross_entropy(
                            logits_flat_b,
                            target_flat_b,
                            ignore_index=IGNORE_INDEX,
                            reduction="none",
                        )  # [H*W]
                        
                        valid_flat = (target_flat_b != IGNORE_INDEX)
                        
                        # CE_all (big picture)
                        if valid_flat.any():
                            ce_all = ce_pix[valid_flat].mean()
                        else:
                            ce_all = ce_pix.new_tensor(0.0)
                        
                        # pixel masks based on current prediction
                        pred_flat = pred_b.view(-1)
                        wrong_flat = valid_flat & (pred_flat != target_flat_b)
                        correct_flat = valid_flat & (~wrong_flat)
                        
                        # sums over groups
                        sum_pos = ce_pix[correct_flat].sum()
                        sum_neg = ce_pix[wrong_flat].sum()
                        
                        # counts (as float for denom)
                        m = correct_flat.sum().item()  # #correct pixels
                        n = wrong_flat.sum().item()    # #wrong pixels
                        v = valid_flat.sum().item()    # #valid pixels
                        
                        running_valid_pix += v
                        running_wrong_pix += n
                        running_correct_pix += m

                        # ---- margin diagnostics on wrong pixels ----
                        if n > 0:
                            with torch.no_grad():
                                lf = logits_flat_b.detach()  # [H*W, C]
                                tf = target_flat_b           # [H*W]
                                pf = pred_flat               # [H*W]
                        
                                true_logits = lf[wrong_flat, tf[wrong_flat]]
                                pred_logits = lf[wrong_flat, pf[wrong_flat]]
                        
                                # margin = logit_true - logit_pred (negative if wrong by argmax)
                                running_margin_wrong_sum += (true_logits - pred_logits).sum().item()

                        if n > 0:
                            running_ce_wrong_sum += ce_pix[wrong_flat].sum().item()
                        if m > 0:
                            running_ce_correct_sum += ce_pix[correct_flat].sum().item()
                        
                        if exact:
                            # exact grid: reward only, and you already set mid loss to 0 later
                            loss_final_b = w_correct_ce * ce_all
                        else:
                            # wrong grid: big picture secondary + weighted (pos/neg) term
                            w_pos = 2.0 * w_correct_ce          # e.g. if exact reward is 0.1 -> 0.2 here
                            w_neg = 10.0 * w_pos                # 10x louder per wrong pixel
                        
                            denom = max((w_pos * m + w_neg * n), 1e-8)
                            weighted = (w_pos * sum_pos + w_neg * sum_neg) / denom
                        
                            loss_final_b = 0.5 * ce_all + weighted
                        
                        final_ce_losses.append(loss_final_b)
                        
                        # ----- middle-layer CE per sample (keep as-is) -----
                        logits_middle_b = logits_middle[b]  # [C, H, W]
                        logits_middle_flat_b = logits_middle_b.permute(1, 2, 0).reshape(-1, num_colors)
                        
                        ce_mid_b = F.cross_entropy(
                            logits_middle_flat_b,
                            target_flat_b,
                            ignore_index=IGNORE_INDEX,
                            reduction="mean",
                        )
                        
                        # mid loss gate: exact => 0, else 1
                        w_mid = 0.0 if exact else 1.0
                        mid_ce_losses.append(w_mid * ce_mid_b)

                    # aggregate CE & mid losses
                    if final_ce_losses:
                        L_ce = torch.stack(final_ce_losses).mean()
                        L_mid = torch.stack(mid_ce_losses).mean()
                    else:
                        L_ce = torch.tensor(0.0, device=device)
                        L_mid = torch.tensor(0.0, device=device)
                
                    # base supervised loss (output + mid)
                    loss = L_ce + 0.01 * L_mid

                    # --- 2) Correctness head BCE (global success/failure signal) ---
                    # labels: 1 if grid is exact, else 0
                    labels_corr = is_exact_tensor.float()  # [B]
                
                    # Compute features for correctness head
                    feats_corr = model._compute_correct_features(encoded, num_prefix)  # [B, 2D]
                
                    # Warmup schedule:
                    #  - epoch < 20: skip correctness loss entirely
                    #  - 20 <= epoch < 30: train head only (detach features)
                    #  - epoch >= 30: train head + backbone
                    if epoch < 20:
                        L_corr = torch.tensor(0.0, device=device)
                    else:
                        if 20 <= epoch < 30:
                            feats_for_head = feats_corr.detach()
                        else:
                            feats_for_head = feats_corr
                
                        logits_corr = model.correct_head(feats_for_head).squeeze(-1)  # [B]
                
                        # per-sample BCE (no reduction yet)
                        bce_corr = F.binary_cross_entropy_with_logits(
                            logits_corr,
                            labels_corr,
                            reduction="none",
                        )  # [B]
                
                        # apply CE reward weight based ONLY on label (grid correctness)
                        # same pattern as output CE
                        weight_corr = torch.where(
                            labels_corr == 1.0,
                            torch.full_like(labels_corr, w_correct_ce),  # reward for correct grids
                            torch.ones_like(labels_corr),                # full weight for wrong grids
                        )
                
                        corr_losses = weight_corr * bce_corr
                        L_corr = corr_losses.mean()
                
                    # weight for correctness head
                    lambda_corr =args.correct_lambda
                
                    # simplest version: no dynamic scaling yet
                    loss = loss + lambda_corr * L_corr

                for idx in range(batch_size):
                    train_exact += int(is_exact_tensor[idx].item())
                    train_examples += 1
                
                # accumulate per-sample losses, weighted by batch size
                running_ce_loss += L_ce.item() * batch_size
                running_mid_loss += L_mid.item() * batch_size
                running_corr_loss += lambda_corr * L_corr.item() * batch_size
                
                # ---- Gradient / AMP guard ----
                # 0) Decide globally if we should skip due to bad loss
                found_bad_loss = not torch.isfinite(loss)
                
                if distributed and dist.is_initialized():
                    flag = torch.tensor([int(found_bad_loss)], device=device, dtype=torch.int32)
                    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
                    found_bad_loss = flag.item() > 0
                
                if found_bad_loss:
                    if is_main_process:
                        print("Non-finite loss detected on at least one rank, skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    # IMPORTANT: DO NOT call scaler.step or scaler.update here
                    continue
                
                # 1) Backward + unscale
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # 2) Clip and check gradient norm
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                found_bad_grad = not torch.isfinite(total_norm)
                
                if distributed and dist.is_initialized():
                    flag = torch.tensor([int(found_bad_grad)], device=device, dtype=torch.int32)
                    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
                    found_bad_grad = flag.item() > 0
                
                if found_bad_grad:
                    if is_main_process:
                        print(f"Non-finite grad norm ({total_norm}) on at least one rank, skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    # Here it's OK to call scaler.update(), because unscale_() has already
                    # done an inf-check and recorded found_inf for this step.
                    scaler.update()
                    continue
                
                # 3) Safe optimizer step
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * batch_size
                sample_count += batch_size

                if total_batches > 0 and is_main_process and step % 10 == 0:  # Update every 10 steps
                    elapsed = time.time() - epoch_start
                    avg_step_time = elapsed / step
                    steps_completed = previous_total_steps + step
                    total_steps = len(train_loader) * args.epochs
                    remaining_steps = total_steps - steps_completed
                    elapsed_global = time.time() - global_start
                    avg_time_per_step_global = elapsed_global / max(steps_completed, 1)
                    eta = remaining_steps * avg_time_per_step_global
                    bar_length = 30
                    progress_ratio = steps_completed / total_steps if total_steps else 0
                    filled = int(bar_length * progress_ratio)
                    bar = "#" * filled + "-" * (bar_length - filled)
                    progress = 100.0 * progress_ratio
                    sys.stdout.write(
                        f"\rEpoch {epoch} [{bar}] {progress:5.1f}% ETA {_format_eta(eta)}"
                    )
                    sys.stdout.flush()

            if total_batches > 0 and is_main_process:
                sys.stdout.write("\n")
            previous_total_steps += total_batches

            epoch_duration = time.time() - epoch_start if total_batches > 0 else 0.0

            train_totals = torch.tensor(
                [
                    running_loss,
                    sample_count,
                    train_exact,
                    train_examples,
                    running_ce_loss,
                    running_mid_loss,
                    running_corr_loss,
                    running_valid_pix,
                    running_wrong_pix,
                    running_correct_pix,
                    running_ce_wrong_sum,
                    running_ce_correct_sum,
                    running_margin_wrong_sum,
                ],
                dtype=torch.float64,
                device=device,
            )
            
            if distributed and dist.is_initialized():
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            
            (
                running_loss_total,
                sample_count_total,
                train_exact_total,
                train_examples_total,
                ce_loss_total,
                mid_loss_total,
                corr_loss_total,
                valid_pix_total,
                wrong_pix_total,
                correct_pix_total,
                ce_wrong_sum_total,
                ce_correct_sum_total,
                margin_wrong_sum_total,
            ) = train_totals.tolist()

            avg_train_loss = running_loss_total / max(sample_count_total, 1)
            avg_ce_loss = ce_loss_total / max(sample_count_total, 1)
            avg_mid_loss = mid_loss_total / max(sample_count_total, 1)
            avg_corr_loss = corr_loss_total / max(sample_count_total, 1)
            train_acc = train_exact_total / max(train_examples_total, 1)
            
            wrong_pixel_frac = wrong_pix_total / max(valid_pix_total, 1.0)
            mean_ce_wrong = ce_wrong_sum_total / max(wrong_pix_total, 1.0)
            mean_ce_correct = ce_correct_sum_total / max(correct_pix_total, 1.0)
            
            avg_m = correct_pix_total / max(sample_count_total, 1.0)
            avg_n = wrong_pix_total / max(sample_count_total, 1.0)

            mean_margin_wrong = margin_wrong_sum_total / max(wrong_pix_total, 1.0)


            total_elapsed = time.time() - global_start
            total_steps = len(train_loader) * args.epochs
            steps_completed = min(previous_total_steps, total_steps)
            remaining_steps = total_steps - steps_completed
            avg_time_per_step_global = total_elapsed / max(steps_completed, 1)
            total_eta = remaining_steps * avg_time_per_step_global

            log_parts = [
                f"epoch={epoch}",
                f"train_loss={avg_train_loss:.4f}",
                f"L_ce={avg_ce_loss:.4f}",
                f"L_mid={avg_mid_loss:.4f}",
                f"L_corr={avg_corr_loss:.4f}",
                f"train_acc={train_acc:.4f}",
                f"wrong_pix_frac={wrong_pixel_frac:.4f}",
                f"mean_CE_wrong={mean_ce_wrong:.4f}",
                f"mean_CE_correct={mean_ce_correct:.4f}",
                f"mean_margin_wrong={mean_margin_wrong:.4f}",
                f"avg_m={avg_m:.1f}",
                f"avg_n={avg_n:.1f}",
                f"epoch_time={epoch_duration:.1f}s",
            ]

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.learning_rate
            log_parts.append(f"lr={current_lr:.6f}")           
            if is_main_process:
                print(" | ".join(log_parts))

            if scheduler is not None:
                scheduler.step()
                
            # early stop based on num_success
            if args.num_success > 0:
                # train_acc is already global (all-reduced), so same on all ranks
                if train_acc == 1.0:
                    success_streak += 1
                else:
                    success_streak = 0

                if success_streak >= args.num_success:
                    if is_main_process:
                        print(
                            f"Early stopping TTT: train_acc == 1.0 for "
                            f"{success_streak} consecutive epochs (>= num_success={args.num_success})."
                        )
                    # Break out of the epoch loop
                    break

    finally:
        if distributed and dist.is_initialized():
            dist.barrier()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()

    generate_predictions(
        model,
        eval_loader,
        device,
        img_size=args.image_size,
        attempt_nums=args.num_attempts,
        task_transform_resolver=get_eval_rot_transform_resolver(),
        fix_scale_factor=args.fix_scale_factor,
        disable_translation=args.disable_translation,
        if_fix_scale=args.disable_resolution_augmentation,
        save_name=args.eval_save_name + "_attempt_" + str(cur_attempt_idx),
        eval_split=args.eval_split,
        task_type=args.data_root.split("/")[-1],  # e.g., "ARC-AGI"
    )

def train(args: argparse.Namespace) -> None:
    distributed, rank, world_size, local_rank, device = init_distributed_mode(args)
    set_seed(args.seed + (rank if distributed else 0))

    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler = build_dataloaders(
        args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    total_train_examples = len(train_dataset)

    if (not distributed) or rank == 0:
        print(f"Total training examples: {total_train_examples}")


    model_original = load_model_only(
        args=args, train_dataset=train_dataset, device=device, distributed=distributed, rank=rank, local_rank=local_rank
    )
    
    for attempt_idx in range(args.ttt_num_each):
        model = deepcopy(model_original)
        print(f"Starting test-time training attempt {attempt_idx + 1}/{args.ttt_num_each}...")
        ttt_once(model=model, device=device, distributed=distributed, rank=rank,
                train_loader=train_loader, train_sampler=train_sampler,
                eval_loader=eval_loader, cur_attempt_idx=attempt_idx)



if __name__ == "__main__":
    args = parse_args()
    train(args)
