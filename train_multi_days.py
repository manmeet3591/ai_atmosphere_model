"""
AI Atmosphere Model — Multi-day batched training
================================================
Loads B consecutive days into a single GPU batch to fill VRAM.

Key optimisations vs train.py:
  1. Batch ERA5 fetch  — fetch all B days in ONE GCS .compute() call
                         instead of B × 300 s individual calls.
  2. Batch prefetch    — while GPU trains on batch i, a background thread
                         fetches batch i+1 ERA5 data (hides ~400 s wait).
  3. GODAS file cache  — pentad files shared across many days; cache in
                         RAM to avoid re-reading the same file repeatedly.

Usage:
  python train_multi_days.py [--batch-days 48] [--epochs-per-batch 8] ...
"""

import os
import gc
import json
import datetime
import argparse
import logging
import time
import concurrent.futures

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
    fetch_era5_date_range,
    build_sample,
    load_progress,
    save_progress,
    push_checkpoint_to_hub,
    _read_godas_grib_eccodes,
    ensure_lat_ascending,
    standardize_latlon,
    COND_CHANNELS,
    TARGET_CHANNELS,
    CROSS_ATTN_DIM,
    DEFAULT_GODAS_DIR,
    DEFAULT_STATIC_DIR,
    DEFAULT_CHECKPOINT,
)
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GODAS file cache (pentad files are shared across ~5 consecutive days)
# ---------------------------------------------------------------------------

_godas_file_cache: dict = {}


def load_godas_cached(godas_index, date_str):
    """
    Load GODAS for date_str with per-file RAM caching.
    GODAS pentad files cover ~5 days, so the same file is reused for
    consecutive days in a batch — reading it once saves ~13 s/reuse.
    """
    target = np.datetime64(date_str, "D")
    prior  = [(t, p) for t, p in godas_index if t <= target]
    if not prior:
        raise ValueError(f"No GODAS data on or before {date_str}")
    _, fpath = prior[-1]

    if fpath not in _godas_file_cache:
        t_das, s_das = _read_godas_grib_eccodes(fpath)
        if not t_das:
            raise RuntimeError(f"No temperature data in {fpath}")
        ds = xr.Dataset({
            "potential_temperature": xr.concat(t_das, dim="level"),
            "salinity":              xr.concat(s_das, dim="level") if s_das else
                                     xr.concat(t_das, dim="level") * float("nan"),
        })
        _godas_file_cache[fpath] = standardize_latlon(ensure_lat_ascending(ds))
        log.debug(f"  GODAS cached: {os.path.basename(fpath)}")

    return _godas_file_cache[fpath]


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
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        except RuntimeError as e:
            log.warning(f"  Checkpoint incompatible — starting from scratch. {e}")

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

    # -- Group into batches of B --
    B = args.batch_days
    batches = [dates_todo[i:i+B] for i in range(0, len(dates_todo), B)]
    log.info(f"  {len(batches)} batches of up to {B} days each")

    model.train()
    step = 0

    def _submit_batch(pool, date_batch):
        return pool.submit(fetch_era5_date_range, ds_atmos, ds_land, date_batch)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        # Prefetch first batch immediately
        era5_future = _submit_batch(pool, batches[0])
        log.info(f"  ERA5 prefetch started: batch 1/{len(batches)}  "
                 f"({batches[0][0]} .. {batches[0][-1]})")

        for bi, date_batch in enumerate(batches):
            # ---- Wait for this batch's ERA5 ----
            t_wait = time.perf_counter()
            try:
                era5_map = era5_future.result()  # dict: date_str -> (atm_t, atm_t1, lnd_t)
            except Exception as exc:
                log.warning(f"  SKIP batch {bi+1}: ERA5 fetch failed: {exc}")
                if bi + 1 < len(batches):
                    era5_future = _submit_batch(pool, batches[bi + 1])
                continue
            wait_s = time.perf_counter() - t_wait
            log.info(f"  Batch {bi+1}/{len(batches)} ERA5 ready  wait={wait_s:.1f}s  "
                     f"({len(era5_map)} days fetched)")

            # ---- Prefetch next batch in background ----
            if bi + 1 < len(batches):
                era5_future = _submit_batch(pool, batches[bi + 1])
                log.info(f"  ERA5 prefetch started: batch {bi+2}/{len(batches)}  "
                         f"({batches[bi+1][0]} .. {batches[bi+1][-1]})")

            # ---- Build samples (GODAS from cache + channels + HEALPix) ----
            batch_X, batch_Y, dates_in_batch = [], [], []

            for date_t in date_batch:
                date_t_str  = date_t.isoformat()
                date_t1_str = (date_t + datetime.timedelta(days=1)).isoformat()

                if date_t_str not in era5_map:
                    log.warning(f"  SKIP {date_t_str}: not in ERA5 batch result")
                    continue
                ds_atm_t, ds_atm_t1, ds_lnd_t = era5_map[date_t_str]

                try:
                    ds_ocn_t = load_godas_cached(godas_index, date_t_str)
                    xb, yb = build_sample(
                        date_t_str, date_t1_str,
                        ds_atm_t, ds_atm_t1, ds_lnd_t,
                        godas_index, ds_stat, ref_lat, ref_lon, regridder,
                        ds_ocn=ds_ocn_t,
                    )
                except Exception as exc:
                    log.warning(f"  SKIP {date_t_str}: sample build failed: {exc}")
                    continue

                batch_X.append(xb)
                batch_Y.append(yb)
                dates_in_batch.append(date_t_str)

            del era5_map
            gc.collect()

            if not batch_X:
                continue

            # ---- Stack into GPU batch ----
            N = len(batch_X)
            X = torch.cat(batch_X, dim=0).to(device)   # [N, 20, 12, 64, 64]
            Y = torch.cat(batch_Y, dim=0).to(device)   # [N, 14, 12, 64, 64]
            del batch_X, batch_Y

            enc_hid = torch.zeros(N, 1, CROSS_ATTN_DIM, device=device)
            log.info(f"  === GPU batch {bi+1}: {N} days  "
                     f"{dates_in_batch[0]} .. {dates_in_batch[-1]}  "
                     f"X{list(X.shape)} ===")

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
                log.info(f"  batch {bi+1}  epoch {epoch:02d}/{args.epochs_per_batch}  "
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

    log.info(f"Training complete. Best loss: {best_loss:.6f}")
    log.info(f"Model saved to: {args.checkpoint}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AI Atmosphere S2S — Multi-day batched diffusion training")
    p.add_argument("--start-date",        default="2018-01-06")
    p.add_argument("--end-date",          default="2022-01-01")
    p.add_argument("--godas-dir",         default=DEFAULT_GODAS_DIR)
    p.add_argument("--static-dir",        default=DEFAULT_STATIC_DIR)
    p.add_argument("--checkpoint",        default=DEFAULT_CHECKPOINT)
    p.add_argument("--progress-file",     default="training_progress.json",
                   help="Shared with train.py — resume works across both scripts")
    p.add_argument("--hf-repo",           default=None,
                   help="HuggingFace repo ID (e.g. dsocairlab/Earthmind-S2S)")
    p.add_argument("--batch-days",        type=int, default=48,
                   help="Days per GPU batch (default: 48)")
    p.add_argument("--epochs-per-batch",  type=int, default=8,
                   help="Training epochs per batch (default: 8)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
