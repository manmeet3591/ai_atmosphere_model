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
from diffusers import FlowMatchEulerDiscreteScheduler

# ---------------------------------------------------------------------------
# Reuse every helper from train.py — no duplication
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
    setup_ddp,
    cleanup_ddp,
    is_main,
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
    rank, world_size, device = setup_ddp()
    if is_main():
        log.info(f"Device: {device}  world_size={world_size}  "
                 f"batch_days={args.batch_days}  "
                 f"epochs_per_batch={args.epochs_per_batch}")

    # -- Data pipeline: rank 0 only (saves ~500s/batch of redundant GCS fetches) --
    if is_main():
        ds_atmos, ds_land = open_era5()
        log.info(f"Indexing GODAS from {args.godas_dir} ...")
        godas_index = build_godas_index(args.godas_dir)
        log.info(f"Loading static fields from {args.static_dir} ...")
        ds_mask_raw, ds_topo_raw = load_static_fields(args.static_dir)
        regridder, ref_lat, ref_lon = build_regridder(ds_atmos)
        ds_stat = prepare_static_fields(ds_mask_raw, ds_topo_raw, ref_lat, ref_lon)
    else:
        ds_atmos = ds_land = godas_index = None
        regridder = ref_lat = ref_lon = ds_stat = None

    # Same seed on all ranks → identical random init (no broadcast needed)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model     = build_model(device)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    loss_fn   = nn.MSELoss()

    # -- Resume (all ranks load from shared filesystem) --
    resume = not args.no_resume
    completed_dates, best_loss = load_progress(args.progress_file)
    if resume and args.checkpoint and os.path.exists(args.checkpoint):
        if is_main():
            log.info(f"Resuming from checkpoint: {args.checkpoint}")
        try:
            ckpt = torch.load(args.checkpoint, map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
                if "best_loss" in ckpt:
                    best_loss = ckpt["best_loss"]
            else:
                model.load_state_dict(ckpt)
        except RuntimeError as e:
            if is_main():
                log.warning(f"  Checkpoint incompatible — starting from scratch. {e}")
    elif not resume:
        best_loss = float("inf")
        completed_dates = set()
        if is_main():
            log.info("Starting fresh (--no-resume)")

    if world_size > 1:
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))])
        if is_main():
            log.info(f"  Model wrapped in DDP ({world_size} GPUs)")

    raw_model = model.module if world_size > 1 else model
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda") if "cuda" in device else None

    if resume and args.checkpoint and os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location=device)
            if isinstance(ckpt, dict) and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # -- Date list (all ranks need this for loop counting) --
    start = datetime.date.fromisoformat(args.start_date)
    end   = datetime.date.fromisoformat(args.end_date)
    all_dates = []
    d = start
    while d < end:
        all_dates.append(d)
        d += datetime.timedelta(days=1)

    dates_todo = [d for d in all_dates if d.isoformat() not in completed_dates]
    if is_main():
        log.info(f"Training: {start} -> {end}  ({len(all_dates)} days total, "
                 f"{len(all_dates) - len(dates_todo)} already done, "
                 f"{len(dates_todo)} remaining, "
                 f"batch={args.batch_days}, epochs/batch={args.epochs_per_batch})")

    if not dates_todo:
        if is_main():
            log.info("All dates already completed. Nothing to do.")
        cleanup_ddp()
        return

    B = args.batch_days
    n_batches = (len(dates_todo) + B - 1) // B
    if is_main():
        batches = [dates_todo[i:i+B] for i in range(0, len(dates_todo), B)]
        log.info(f"  {n_batches} batches of up to {B} days each")

    model.train()
    step = 0

    try:
        # Rank 0 runs the data pipeline + prefetch; other ranks wait at broadcast
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if is_main() else None
        era5_future = None
        if is_main():
            batches_list = batches
            era5_future = pool.submit(fetch_era5_date_range,
                                      ds_atmos, ds_land, batches_list[0])
            log.info(f"  ERA5 prefetch started: batch 1/{n_batches}")

        for bi in range(n_batches):
            # ---- Rank 0: fetch ERA5, build samples, prepare tensors ----
            if is_main():
                date_batch = batches_list[bi]
                t_wait = time.perf_counter()
                try:
                    era5_map = era5_future.result()
                except Exception as exc:
                    log.warning(f"  SKIP batch {bi+1}: ERA5 fetch failed: {exc}")
                    era5_map = {}
                wait_s = time.perf_counter() - t_wait
                log.info(f"  Batch {bi+1}/{n_batches} ERA5 ready  wait={wait_s:.1f}s  "
                         f"({len(era5_map)} days fetched)")

                if bi + 1 < n_batches:
                    era5_future = pool.submit(fetch_era5_date_range,
                                              ds_atmos, ds_land, batches_list[bi + 1])

                batch_X, batch_Y, dates_in_batch = [], [], []
                for date_t in date_batch:
                    date_t_str  = date_t.isoformat()
                    date_t1_str = (date_t + datetime.timedelta(days=1)).isoformat()
                    if date_t_str not in era5_map:
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

                if batch_X:
                    X_full = torch.cat(batch_X, dim=0)
                    Y_full = torch.cat(batch_Y, dim=0)
                    N = X_full.shape[0]
                    del batch_X, batch_Y
                else:
                    N = 0

            # ---- Broadcast batch size to all ranks ----
            if world_size > 1:
                n_tensor = torch.tensor([N if is_main() else 0],
                                        dtype=torch.long, device=device)
                dist.broadcast(n_tensor, src=0)
                N = n_tensor.item()

            if N == 0:
                continue

            # ---- Broadcast tensors from rank 0 to all ranks (NCCL needs CUDA) ----
            if world_size > 1:
                if is_main():
                    X_full = X_full.to(device)
                    Y_full = Y_full.to(device)
                else:
                    X_full = torch.empty(N, 20, 12, 64, 64, device=device)
                    Y_full = torch.empty(N, 14, 12, 64, 64, device=device)
                dist.broadcast(X_full, src=0)
                dist.broadcast(Y_full, src=0)

                chunk = N // world_size
                s = rank * chunk
                e = s + chunk if rank < world_size - 1 else N
                X = X_full[s:e]
                Y = Y_full[s:e]
                n_local = X.shape[0]
                del X_full, Y_full
            else:
                X = X_full.to(device)
                Y = Y_full.to(device)
                n_local = N
                del X_full, Y_full

            enc_hid = torch.zeros(n_local, 1, CROSS_ATTN_DIM, device=device)
            if is_main():
                log.info(f"  === GPU batch {bi+1}: {N} days total, "
                         f"{n_local}/rank  "
                         f"{dates_in_batch[0]} .. {dates_in_batch[-1]}  "
                         f"X{list(X.shape)} ===")

            if torch.isnan(X).any() or torch.isnan(Y).any():
                if is_main():
                    log.warning(f"  SKIP batch {bi+1}: NaN in input/target")
                continue

            # ---- Training epochs (all ranks participate) ----
            batch_losses = []
            for epoch in range(1, args.epochs_per_batch + 1):
                noise = torch.randn_like(Y)
                scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)
                idx = torch.randint(0, len(scheduler.timesteps), (1,), device=device).item()
                t = scheduler.timesteps[idx : idx+1]

                sigmas = scheduler.sigmas.to(device)
                sigma_idx = scheduler.index_for_timestep(t[0].item())
                sigma = sigmas[sigma_idx]
                while len(sigma.shape) < len(Y.shape):
                    sigma = sigma.unsqueeze(-1)
                sigma = sigma.expand(n_local, -1, -1, -1, -1)

                noisy_Y = (1.0 - sigma) * Y + sigma * noise
                target_v = Y - noise
                net_input = torch.cat([noisy_Y, X], dim=1)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    pred_v = model(
                        sample=net_input,
                        timestep=t.expand(n_local),
                        encoder_hidden_states=enc_hid,
                    ).sample
                    loss = loss_fn(pred_v, target_v)

                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main():
                        log.warning(f"  NaN/Inf loss in batch {bi+1} epoch {epoch}")
                    continue

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                loss_val = loss.item()
                batch_losses.append(loss_val)
                step += 1
                if is_main():
                    log.info(f"  batch {bi+1}  epoch {epoch:02d}/{args.epochs_per_batch}  "
                             f"loss={loss_val:.6f}  step={step}")

                del noise, noisy_Y, net_input, pred_v, loss, target_v

            # ---- Checkpoint + progress (rank 0 only) ----
            if batch_losses and is_main():
                mean_loss = np.mean(batch_losses)
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    init_cond = f"{dates_in_batch[0]}_to_{dates_in_batch[-1]}"
                    ckpt_data = {"model": raw_model.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "best_loss": mean_loss,
                                 "date": init_cond}
                    base = os.path.splitext(args.checkpoint)[0]
                    dated_path = f"{base}_{init_cond}_{mean_loss:.6f}.pth"
                    torch.save(ckpt_data, dated_path)
                    torch.save(ckpt_data, args.checkpoint)
                    log.info(f"  Checkpoint saved: {os.path.basename(dated_path)}  "
                             f"(best_loss={best_loss:.6f})")
                if args.hf_repo:
                    push_checkpoint_to_hub(
                        args.checkpoint, args.hf_repo,
                        best_loss, dates_in_batch[0])

            if is_main():
                for d_str in dates_in_batch:
                    completed_dates.add(d_str)
                save_progress(args.progress_file, completed_dates, best_loss)
                log.info(f"  Progress: {len(completed_dates)}/{len(all_dates)} days done")

            del X, Y, enc_hid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
        cleanup_ddp()

    if is_main():
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
    p.add_argument("--epochs-per-batch",  type=int, default=50,
                   help="Training epochs per batch (default: 50). "
                        "Higher = better GPU utilization.")
    p.add_argument("--no-resume",         action="store_true", default=False,
                   help="Start training from scratch (default: resume)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
