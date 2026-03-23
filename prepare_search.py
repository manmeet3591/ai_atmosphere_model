"""
prepare_search.py — ONE-TIME data preparation for autoresearch NAS experiments.

Downloads a fixed set of (X, Y) tensor pairs from the real data pipeline
(ERA5 + GODAS + static fields on HEALPix) and caches them as .pt files.
Subsequent train_search.py runs load from cache — no network needed.

DO NOT MODIFY: this file is fixed infrastructure, like prepare.py in autoresearch.

Usage:
    python prepare_search.py [--godas-dir PATH] [--static-dir PATH]
                             [--cache-dir PATH]

Output:
    <cache-dir>/train/YYYY-MM-DD.pt    — 20 training days
    <cache-dir>/val/YYYY-MM-DD.pt      — 10 validation days

Each .pt file contains:
    {"X": Tensor[1, 20, 12, 64, 64], "Y": Tensor[1, 14, 12, 64, 64]}
"""

import os
import argparse
import datetime
import logging

import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (
    open_era5,
    build_godas_index,
    load_static_fields,
    build_regridder,
    prepare_static_fields,
    fetch_era5_date_range,
    build_sample,
    load_godas_for_date,
    _read_godas_grib_eccodes,
    ensure_lat_ascending,
    standardize_latlon,
    DEFAULT_GODAS_DIR,
    DEFAULT_STATIC_DIR,
)
import xarray as xr
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "search_cache"

# ---------------------------------------------------------------------------
# Fixed date splits — do NOT change these; they define the NAS benchmark.
# Training: 20 days from early 2018  (Jan 6 – Jan 25)
# Validation: 10 days from mid 2018  (Jul 1 – Jul 10)
# The train/val split is far apart in time to test generalization.
# ---------------------------------------------------------------------------
TRAIN_DATES = [
    datetime.date(2018, 1, 6) + datetime.timedelta(days=i)
    for i in range(20)
]

VAL_DATES = [
    datetime.date(2018, 7, 1) + datetime.timedelta(days=i)
    for i in range(10)
]


def _godas_load_cached(cache, godas_index, date_str):
    """Load GODAS with in-memory caching (same pentad file covers ~5 days)."""
    target = np.datetime64(date_str, "D")
    prior = [(t, p) for t, p in godas_index if t <= target]
    if not prior:
        raise ValueError(f"No GODAS data on or before {date_str}")
    _, fpath = prior[-1]
    if fpath not in cache:
        t_das, s_das = _read_godas_grib_eccodes(fpath)
        if not t_das:
            raise RuntimeError(f"No temperature data in {fpath}")
        ds = xr.Dataset({
            "potential_temperature": xr.concat(t_das, dim="level"),
            "salinity": xr.concat(s_das, dim="level") if s_das else
                        xr.concat(t_das, dim="level") * float("nan"),
        })
        cache[fpath] = standardize_latlon(ensure_lat_ascending(ds))
        log.info(f"  GODAS cached: {os.path.basename(fpath)}")
    return cache[fpath]


def cache_date_list(dates, split_name, cache_dir,
                    ds_atmos, ds_land, godas_index,
                    ds_stat, ref_lat, ref_lon, regridder):
    out_dir = os.path.join(cache_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    # Filter already-cached days
    todo = [d for d in dates
            if not os.path.exists(os.path.join(out_dir, f"{d.isoformat()}.pt"))]
    if not todo:
        log.info(f"[{split_name}] All {len(dates)} days already cached.")
        return

    log.info(f"[{split_name}] Fetching ERA5 for {len(todo)} days in one batch ...")
    era5_map = fetch_era5_date_range(ds_atmos, ds_land, todo)
    log.info(f"[{split_name}] ERA5 fetch done. Building samples ...")

    godas_cache: dict = {}
    for d in todo:
        d_str  = d.isoformat()
        d1_str = (d + datetime.timedelta(days=1)).isoformat()
        out_path = os.path.join(out_dir, f"{d_str}.pt")

        if d_str not in era5_map:
            log.warning(f"  SKIP {d_str}: not in ERA5 batch result")
            continue

        ds_atm_t, ds_atm_t1, ds_lnd_t = era5_map[d_str]
        try:
            ds_ocn = _godas_load_cached(godas_cache, godas_index, d_str)
            X, Y = build_sample(
                d_str, d1_str,
                ds_atm_t, ds_atm_t1, ds_lnd_t,
                godas_index, ds_stat, ref_lat, ref_lon, regridder,
                ds_ocn=ds_ocn,
            )
        except Exception as exc:
            log.warning(f"  SKIP {d_str}: {exc}")
            continue

        torch.save({"X": X.cpu(), "Y": Y.cpu()}, out_path)
        log.info(f"  Saved {out_path}  X{list(X.shape)} Y{list(Y.shape)}")

    log.info(f"[{split_name}] Done. {len(os.listdir(out_dir))} files in {out_dir}")


def main(args):
    log.info("Opening ERA5 Zarr (anonymous GCS) ...")
    ds_atmos, ds_land = open_era5()

    log.info(f"Building GODAS index from {args.godas_dir} ...")
    godas_index = build_godas_index(args.godas_dir)

    log.info(f"Loading static fields from {args.static_dir} ...")
    ds_mask_raw, ds_topo_raw = load_static_fields(args.static_dir)

    regridder, ref_lat, ref_lon = build_regridder(ds_atmos)
    ds_stat = prepare_static_fields(ds_mask_raw, ds_topo_raw, ref_lat, ref_lon)

    cache_date_list(
        TRAIN_DATES, "train", args.cache_dir,
        ds_atmos, ds_land, godas_index,
        ds_stat, ref_lat, ref_lon, regridder,
    )
    cache_date_list(
        VAL_DATES, "val", args.cache_dir,
        ds_atmos, ds_land, godas_index,
        ds_stat, ref_lat, ref_lon, regridder,
    )

    n_train = len(os.listdir(os.path.join(args.cache_dir, "train")))
    n_val   = len(os.listdir(os.path.join(args.cache_dir, "val")))
    log.info(f"Cache ready: {n_train} train days, {n_val} val days in {args.cache_dir}/")
    log.info("You can now run: python train_search.py")


def parse_args():
    p = argparse.ArgumentParser(
        description="One-time data prep for autoresearch NAS experiments")
    p.add_argument("--godas-dir",  default=DEFAULT_GODAS_DIR)
    p.add_argument("--static-dir", default=DEFAULT_STATIC_DIR)
    p.add_argument("--cache-dir",  default=DEFAULT_CACHE_DIR)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
