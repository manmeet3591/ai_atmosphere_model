"""
AI Atmosphere Model — Multi-day batched training
================================================
Identical physics to train.py but loads B consecutive days into a single
GPU batch to fill available VRAM.

Prefetch strategy:
  - ThreadPoolExecutor(max_workers=B) keeps B ERA5 GCS fetches running
    concurrently.
  - While the GPU trains on batch i, the pool is already fetching batch i+1.
  - build_sample (GODAS + channels + HEALPix) is sequential but fast (<2 s/day).

Usage:
  python train_multi_days.py [--batch-days 4] [--epochs-per-batch 5] ...
"""

import os
import gc
import json
import datetime
import argparse
import logging
import time
import concurrent.futures
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import LCMScheduler

# ---------------------------------------------------------------------------
# Reuse every helper from train.py — no duplication
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (
    open_era5,
    build_godas_index,
    load_static_fields,
    build_regridder,
    prepare_static_fields,
    build_model,
    fetch_era5_for_day,
    build_sample,
    load_progress,
    save_progress,
    push_checkpoint_to_hub,
    COND_CHANNELS,
    TARGET_CHANNELS,
    CROSS_ATTN_DIM,
    DEFAULT_GODAS_DIR,
    DEFAULT_STATIC_DIR,
    DEFAULT_CHECKPOINT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}  batch_days={args.batch_days}  "
             f"epochs_per_batch={args.epochs_per_batch}")

    ds_atmos, ds_land = open_era5()

    log.info(f"Indexing GODAS from {args.godas_dir} ...")
    godas_index = build_godas_index(args.godas_dir)

    log.info(f"Loading static fields from {args.static_dir} ...")
    ds_mask_raw, ds_topo_raw = load_static_fields(args.static_dir)

    regridder, ref_lat, ref_lon = build_regridder(ds_atmos)
    ds_stat = prepare_static_fields(ds_mask_raw, ds_topo_raw, ref_lat, ref_lon)

    model     = build_model(device)
    scheduler = LCMScheduler(num_train_timesteps=1000)
    loss_fn   = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # -- Resume --
    completed_dates, best_loss = load_progress(args.progress_file)
    if args.checkpoint and os.path.exists(args.checkpoint):
        log.info(f"Loading weights from checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # -- Date list --
    start = datetime.date.fromisoformat(args.start_date)
    end   = datetime.date.fromisoformat(args.end_date)
    all_dates = []
    d = start
    while d < end:
        all_dates.append(d)
        d += datetime.timedelta(days=1)

    dates_todo = [d for d in all_dates if d.isoformat() not in completed_dates]
    skipped = len(all_dates) - len(dates_todo)
    log.info(f"Training: {start} -> {end}  ({len(all_dates)} days total, "
             f"{skipped} already done, {len(dates_todo)} remaining, "
             f"batch={args.batch_days}, epochs/batch={args.epochs_per_batch})")

    if not dates_todo:
        log.info("All dates already completed. Nothing to do.")
        return

    model.train()
    step = 0
    B = args.batch_days

    def _submit(pool, date_t):
        date_t1 = date_t + datetime.timedelta(days=1)
        return pool.submit(
            fetch_era5_for_day,
            ds_atmos, ds_land,
            date_t.isoformat(), date_t1.isoformat(),
        )

    # prefetch_workers controls RAM (2 concurrent ERA5 fetches = ~4 GB overhead).
    # B controls GPU batch size. We accumulate B samples using a sliding window
    # of prefetch_workers futures — separating RAM safety from batch size.
    prefetch_workers = min(B, args.prefetch_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
        # Sliding-window prefetch: always keep prefetch_workers futures in flight
        future_queue = deque()
        prefill = min(prefetch_workers, len(dates_todo))
        for i in range(prefill):
            future_queue.append((dates_todo[i], _submit(pool, dates_todo[i])))
            log.info(f"  Prefetch started for {dates_todo[i].isoformat()}")
        next_idx = prefill

        batch_X, batch_Y, dates_in_batch = [], [], []

        while future_queue:
            # ---- Pull one sample at a time, replenish the sliding window ----
            date_t, future = future_queue.popleft()

            # Immediately submit the next future to keep the pool busy
            if next_idx < len(dates_todo):
                nxt = dates_todo[next_idx]
                future_queue.append((nxt, _submit(pool, nxt)))
                log.info(f"  Prefetch started for {nxt.isoformat()}")
                next_idx += 1

            date_t1     = date_t + datetime.timedelta(days=1)
            date_t_str  = date_t.isoformat()
            date_t1_str = date_t1.isoformat()

            t_wait = time.perf_counter()
            try:
                ds_atm_t, ds_atm_t1, ds_lnd_t = future.result()
            except Exception as exc:
                log.warning(f"  SKIP {date_t_str}: ERA5 failed: {exc}")
                continue
            wait_s = time.perf_counter() - t_wait
            log.info(f"  [{date_t_str}] ERA5 ready  wait={wait_s:.1f}s")

            try:
                xb, yb = build_sample(
                    date_t_str, date_t1_str,
                    ds_atm_t, ds_atm_t1, ds_lnd_t,
                    godas_index, ds_stat, ref_lat, ref_lon, regridder,
                )
            except Exception as exc:
                log.warning(f"  SKIP {date_t_str}: sample build failed: {exc}")
                continue

            del ds_atm_t, ds_atm_t1, ds_lnd_t
            batch_X.append(xb)
            batch_Y.append(yb)
            dates_in_batch.append(date_t_str)

            # ---- Train when we have B samples or the queue is exhausted ----
            if len(batch_X) < B and future_queue:
                continue  # keep accumulating

            if not batch_X:
                continue

            # ---- Stack into GPU batch ----
            N = len(batch_X)
            X = torch.cat(batch_X, dim=0).to(device)   # [N, 20, 12, 64, 64]
            Y = torch.cat(batch_Y, dim=0).to(device)   # [N, 14, 12, 64, 64]
            del batch_X, batch_Y

            enc_hid = torch.zeros(N, 1, CROSS_ATTN_DIM, device=device)

            log.info(f"  === GPU batch: {N} days  {dates_in_batch[0]} .. "
                     f"{dates_in_batch[-1]}  "
                     f"X{list(X.shape)}  Y{list(Y.shape)} ===")

            # ---- Training epochs ----
            batch_losses = []
            for epoch in range(1, args.epochs_per_batch + 1):
                noise      = torch.randn_like(Y)
                timesteps  = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (N,), device=device).long()
                noisy_Y    = scheduler.add_noise(Y, noise, timesteps)
                net_input  = torch.cat([noisy_Y, X], dim=1)

                noise_pred = model(
                    sample=net_input,
                    timestep=timesteps,
                    encoder_hidden_states=enc_hid,
                ).sample

                loss = loss_fn(noise_pred, noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                batch_losses.append(loss_val)
                step += 1
                log.info(f"  {dates_in_batch[0]}+{N-1}d  "
                         f"epoch {epoch:02d}/{args.epochs_per_batch}  "
                         f"loss={loss_val:.6f}  step={step}")

                del noise, noisy_Y, net_input, noise_pred, loss

            # ---- Checkpoint + HF push ----
            mean_loss = np.mean(batch_losses)
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), args.checkpoint)
                log.info(f"  Checkpoint saved  (best_loss={best_loss:.6f})")
                if args.hf_repo:
                    push_checkpoint_to_hub(
                        args.checkpoint, args.hf_repo,
                        best_loss, dates_in_batch[0])

            # ---- Mark all days in batch as done ----
            for d_str in dates_in_batch:
                completed_dates.add(d_str)
            save_progress(args.progress_file, completed_dates, best_loss)
            log.info(f"  Progress: {len(completed_dates)}/{len(all_dates)} days done")

            del X, Y, enc_hid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Reset accumulators for next batch
            batch_X, batch_Y, dates_in_batch = [], [], []

    log.info(f"Training complete. Best loss: {best_loss:.6f}")
    log.info(f"Model saved to: {args.checkpoint}")
    log.info(f"Progress saved to: {args.progress_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AI Atmosphere S2S — Multi-day batched diffusion training")
    p.add_argument("--start-date",        default="2018-01-06")
    p.add_argument("--end-date",          default="2018-08-13")
    p.add_argument("--godas-dir",         default=DEFAULT_GODAS_DIR)
    p.add_argument("--static-dir",        default=DEFAULT_STATIC_DIR)
    p.add_argument("--checkpoint",        default=DEFAULT_CHECKPOINT)
    p.add_argument("--progress-file",     default="training_progress.json",
                   help="Shared with train.py — resume works across both scripts")
    p.add_argument("--hf-repo",           default=None,
                   help="HuggingFace repo ID (e.g. sluitel/ai-atmosphere-s2s)")
    p.add_argument("--batch-days",        type=int, default=8,
                   help="Days per GPU batch (default: 8). "
                        "Increase until ~80%% VRAM used (watch nvidia-smi).")
    p.add_argument("--prefetch-workers",  type=int, default=2,
                   help="Parallel ERA5 GCS fetches (default: 2). "
                        "Each fetch holds ~2 GB of system RAM; keep low to avoid OOM.")
    p.add_argument("--epochs-per-batch",  type=int, default=5,
                   help="Training epochs per batch (default: 5)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
