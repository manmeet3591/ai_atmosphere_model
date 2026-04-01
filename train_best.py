"""
AI Atmosphere Model — S2S Diffusion Training Script

Trains a diffusion-based 3D UNet to predict next-day atmospheric state
from multi-source inputs on a HEALPix grid.

Data sources:
  - ERA5 atmosphere + land: GCS Zarr (public, anonymous)
  - GODAS ocean:            local GRIB files in GODAS_DIR
  - Static fields:          local NetCDF cache (downloaded once on first run)

Usage:
  python train.py [--start-date YYYYMMDD] [--end-date YYYYMMDD]
                  [--godas-dir PATH] [--static-dir PATH]
                  [--checkpoint PATH] [--epochs-per-day N]
"""

import os
import gc
import glob
import json
import datetime
import argparse
import logging
import time
import threading
import concurrent.futures

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import cKDTree
from diffusers import UNet3DConditionModel, FlowMatchEulerDiscreteScheduler
import earth2grid

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
ERA5_ZARR  = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_GODAS_DIR  = "/media/airlab/ROCSTOR/earthmind_s2s/godas_pentad"
DEFAULT_STATIC_DIR = "/media/airlab/ROCSTOR/earthmind_s2s/static_fields"
DEFAULT_CHECKPOINT = "best_diffusion_atmos_model.pth"

# ---------------------------------------------------------------------------
# Grid configuration
# ---------------------------------------------------------------------------
HPX_LEVEL = 6
NSIDE     = 2 ** HPX_LEVEL   # 64
G0        = 9.80665
CONVERT_GPH_TO_HEIGHT = True

# ---------------------------------------------------------------------------
# Min-max normalisation stats
# ---------------------------------------------------------------------------
NORM = {
    "2m_temperature":                {"min": 188.023,       "max": 326.906},
    "10m_u_component_of_wind":       {"min": -281.124,      "max": 132.541},
    "10m_v_component_of_wind":       {"min": -159.218,      "max": 166.600},
    "mean_sea_level_pressure":       {"min": 89991.063,     "max": 109923.438},
    "tcwv":                          {"min": 0.050,         "max": 85.455},
    "top_net_thermal_radiation":     {"min": -1313311.5,    "max": -281460.031},
    "toa_incident_solar_radiation":  {"min": -0.25,         "max": 2014404.625},
    "z_1000": {"min": -6174.143,  "max": 4626.292},
    "z_850":  {"min": 5940.180,   "max": 17029.006},
    "z_700":  {"min": 20248.557,  "max": 32538.490},
    "z_500":  {"min": 43559.191,  "max": 58915.500},
    "z_300":  {"min": 75701.602,  "max": 96554.875},
    "z_250":  {"min": 86646.961,  "max": 109212.938},
    "t_850":  {"min": 216.382,    "max": 310.778},
    "volumetric_soil_water_layer_1": {"min": -0.031,  "max": 0.790},
    "soil_temperature_level_1":      {"min": 196.464, "max": 340.485},
    "potential_temperature":         {"min": 267.408, "max": 306.139},
    "salinity":                      {"min": 0.007,   "max": 0.046},
    "landseamask": {"min": 0.0,    "max": 100.0},
    "elevation":   {"min": -81.25, "max": 5764.0},
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def norm_minmax(arr, vname, eps=1e-12):
    vmin = NORM[vname]["min"]
    vmax = NORM[vname]["max"]
    return (arr - vmin) / (vmax - vmin + eps)


def ensure_lat_ascending(ds, lat_name="latitude"):
    if lat_name in ds.coords and ds[lat_name].values[0] > ds[lat_name].values[-1]:
        ds = ds.sortby(lat_name)
    return ds


def standardize_latlon(ds):
    rename = {}
    for old, new in [("lat", "latitude"), ("lon", "longitude")]:
        if old in ds.dims and new not in ds.dims:
            rename[old] = new
        if old in ds.coords and new not in ds.coords:
            rename[old] = new
    return ds.rename(rename) if rename else ds


def fill_latlon_nans_2d(da2d):
    da2d = da2d.interpolate_na(dim="latitude",  method="linear", fill_value="extrapolate")
    da2d = da2d.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
    return da2d


def fill_nan_nearest_2d(da2d):
    """KDTree nearest-neighbour NaN fill for coastal gaps in static fields."""
    da  = da2d.squeeze(drop=True).copy()
    arr = da.values
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return da
    rows, cols = np.indices(arr.shape)
    valid = np.isfinite(arr)
    tree = cKDTree(np.column_stack([rows[valid], cols[valid]]))
    _, idx = tree.query(np.column_stack([rows[nan_mask], cols[nan_mask]]))
    arr[nan_mask] = arr[valid][idx]
    da.values = arr
    return da


def to_2d(da):
    if "time" in da.dims:
        da = da.isel(time=0) if da.sizes["time"] > 1 else da.squeeze("time", drop=True)
    return da


def resolve_var(ds, candidates):
    for c in candidates:
        if c in ds.data_vars:
            return c
    raise KeyError(f"None of {candidates} found in dataset.")


def select_pressure_level(da, hpa):
    for dim in ["isobaricInhPa", "level", "pressure_level"]:
        if dim in da.dims:
            sel = da.sel({dim: hpa}, method="nearest").squeeze(drop=True)
            return to_2d(sel)
    raise ValueError(f"No pressure dim found; dims={da.dims}")


def daily_mean(ds):
    return ds.resample(time="1D").mean()


def pick_single_day(ds_daily, day_str):
    """Select one day from a daily-mean dataset (robust to time encoding)."""
    target_date = np.datetime64(day_str, "D")
    # Try exact match on date portion only
    times = ds_daily.time.values.astype("datetime64[D]")
    matches = np.where(times == target_date)[0]
    if len(matches) == 0:
        raise KeyError(f"Date {day_str} not found in dataset (available: "
                       f"{times[0]} to {times[-1]})")
    return ds_daily.isel(time=int(matches[0]))


# ---------------------------------------------------------------------------
# GODAS ocean loading
# ---------------------------------------------------------------------------

def _read_godas_grib_eccodes(fpath):
    """
    Read a GODAS GRIB file using the eccodes Python library.
    Bypasses pygrib's date-parsing which fails on GRIB1 timeRangeIndicator=7.
    Returns (temp_das, sal_das) lists of xr.DataArray keyed by level.
    """
    import eccodes
    from gribapi.errors import PrematureEndOfFileError as _GribEOF

    t_das, s_das = [], []
    with open(fpath, "rb") as f:
        while True:
            gid = None
            try:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break  # clean end of file
                name  = eccodes.codes_get(gid, "name")
                level = eccodes.codes_get(gid, "level")
                ni    = eccodes.codes_get(gid, "Ni")
                nj    = eccodes.codes_get(gid, "Nj")
                lat_first = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                lat_last  = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
                lon_first = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
                lon_last  = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")
                values = eccodes.codes_get_values(gid).reshape(nj, ni)

                lats = np.linspace(lat_first, lat_last, nj)
                lons = np.linspace(lon_first, lon_last, ni)

                da = xr.DataArray(
                    values,
                    dims=["lat", "lon"],
                    coords={"lat": lats, "lon": lons, "level": level},
                )
                if "Potential temperature" in name:
                    t_das.append(da)
                elif "Salinity" in name:
                    s_das.append(da)
            except _GribEOF:
                log.debug(f"  Truncated GRIB (still downloading?): {os.path.basename(fpath)}")
                break  # stop reading this file, treat as partial
            except Exception as e:
                log.debug(f"  Skipping message in {os.path.basename(fpath)}: {e}")
            finally:
                if gid is not None:
                    eccodes.codes_release(gid)

    return t_das, s_das


def build_godas_index(godas_dir):
    """
    Scan GODAS directory and return a sorted list of (date, filepath) tuples.
    No data is loaded — just filenames. O(1) memory.
    """
    pattern = os.path.join(godas_dir, "godas.P.????????.grb")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No GODAS .grb files found in {godas_dir}")

    index = []
    for fpath in files:
        datestr = os.path.basename(fpath).split(".")[2]   # YYYYMMDD
        t = np.datetime64(f"{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}", "D")
        index.append((t, fpath))

    log.info(f"GODAS index: {len(index)} files  "
             f"({index[0][0]} → {index[-1][0]})")
    return index  # [(date, path), ...]


def load_godas_for_date(godas_index, target_date_str):
    """
    Load a single GODAS file — the one whose date is nearest prior to
    target_date_str. Returns an xr.Dataset with dims (level, latitude, longitude).
    """
    target = np.datetime64(target_date_str, "D")
    prior  = [(t, p) for t, p in godas_index if t <= target]
    if not prior:
        dates = [t for t, _ in godas_index]
        raise ValueError(f"No GODAS data on or before {target_date_str}. "
                         f"Earliest available: {dates[0]}")
    best_date, fpath = prior[-1]
    log.debug(f"  GODAS: using {os.path.basename(fpath)} (date={best_date}) "
              f"for target={target_date_str}")

    t_das, s_das = _read_godas_grib_eccodes(fpath)
    if not t_das:
        raise RuntimeError(f"No temperature data in {fpath}")

    ds = xr.Dataset({
        "potential_temperature": xr.concat(t_das, dim="level"),
        "salinity":              xr.concat(s_das,  dim="level"),
    })
    ds = standardize_latlon(ds)
    ds = ensure_lat_ascending(ds)
    return ds


# ---------------------------------------------------------------------------
# Progress tracking (resume support)
# ---------------------------------------------------------------------------

def load_progress(progress_path):
    """
    Load completed dates and best loss from a JSON progress file.
    Returns (completed_dates: set[str], best_loss: float).
    """
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            data = json.load(f)
        completed = set(data.get("completed_dates", []))
        best_loss = data.get("best_loss", float("inf"))
        log.info(f"Progress file loaded: {len(completed)} dates done, "
                 f"best_loss={best_loss:.6f}")
        return completed, best_loss
    return set(), float("inf")


def save_progress(progress_path, completed_dates, best_loss):
    """Atomically write progress so a crash mid-write doesn't corrupt it."""
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "completed_dates": sorted(completed_dates),
            "best_loss": best_loss,
        }, f, indent=2)
    os.replace(tmp, progress_path)


def _hf_upload(checkpoint_path, repo_id, best_loss, date_str):
    """Upload checkpoint to HuggingFace Hub (runs in background thread)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=os.path.basename(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Best model — loss={best_loss:.6f}  date={date_str}",
        )
        log.info(f"  HuggingFace upload complete: {repo_id}")
    except Exception as exc:
        log.warning(f"  HuggingFace upload failed: {exc}")


def push_checkpoint_to_hub(checkpoint_path, repo_id, best_loss, date_str):
    """Fire-and-forget background upload; does not block training."""
    t = threading.Thread(
        target=_hf_upload,
        args=(checkpoint_path, repo_id, best_loss, date_str),
        daemon=True,
    )
    t.start()
    log.info(f"  HuggingFace upload started in background -> {repo_id}")
    return t


# ---------------------------------------------------------------------------
# Static fields (download once, cache locally)
# ---------------------------------------------------------------------------

IMERG_URL = "https://pmm.nasa.gov/sites/default/files/downloads/IMERG_land_sea_mask.nc.gz"
TOPO_URL  = "https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n060_0250deg.nc"


def load_static_fields(static_dir):
    """Load (or download-and-cache) land-sea mask and topography."""
    os.makedirs(static_dir, exist_ok=True)

    mask_path = os.path.join(static_dir, "IMERG_land_sea_mask.nc")
    topo_path = os.path.join(static_dir, "GMTED2010_15n060_0250deg.nc")

    if not os.path.exists(mask_path):
        log.info("Downloading IMERG land-sea mask...")
        gz_path = mask_path + ".gz"
        import urllib.request, gzip, shutil
        urllib.request.urlretrieve(IMERG_URL, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(mask_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)

    if not os.path.exists(topo_path):
        log.info("Downloading GMTED2010 topography...")
        import urllib.request
        urllib.request.urlretrieve(TOPO_URL, topo_path)

    ds_mask = standardize_latlon(xr.open_dataset(mask_path))
    ds_topo = xr.open_dataset(topo_path)
    # Shift topo longitude from [-180, 180] to [0, 360]
    ds_topo["longitude"] = ds_topo["longitude"] + 180
    ds_topo = standardize_latlon(ds_topo)

    return ds_mask, ds_topo


def prepare_static_fields(ds_mask_raw, ds_topo_raw, ref_lat, ref_lon):
    """Regrid and NaN-fill static fields to the ERA5 reference grid."""
    mask_rg = ds_mask_raw.interp(latitude=ref_lat, longitude=ref_lon, method="linear")
    topo_rg = ds_topo_raw.interp(latitude=ref_lat, longitude=ref_lon, method="linear")

    ds_stat = ensure_lat_ascending(xr.Dataset({
        "mask": fill_nan_nearest_2d(mask_rg["landseamask"].squeeze(drop=True)),
        "topo": fill_nan_nearest_2d(topo_rg["elevation"].squeeze(drop=True)),
    }))
    return ds_stat


# ---------------------------------------------------------------------------
# ERA5 dataset
# ---------------------------------------------------------------------------

ATMOS_CANDIDATES = {
    "2m_temperature":             ["2m_temperature", "t2m"],
    "10m_u_component_of_wind":    ["10m_u_component_of_wind", "u10"],
    "10m_v_component_of_wind":    ["10m_v_component_of_wind", "v10"],
    "total_column_water_vapour":  ["total_column_water_vapour", "tcwv",
                                   "total_column_integrated_water_vapour"],
    "mean_sea_level_pressure":    ["mean_sea_level_pressure", "msl", "mslp"],
    "top_net_thermal_radiation":  ["top_net_thermal_radiation", "ttr"],
    "toa_incident_solar_radiation": ["toa_incident_solar_radiation", "tisr",
                                     "toa_insolation"],
    "geopotential":  ["geopotential", "z"],
    "temperature":   ["temperature", "t"],
}

LAND_CANDIDATES = {
    "volumetric_soil_water_layer_1": ["swvl1", "volumetric_soil_water_layer_1"],
    "soil_temperature_level_1":      ["stl1",  "soil_temperature_level_1"],
}


def open_era5():
    log.info("Opening ERA5 Zarr ...")
    ds = xr.open_zarr(ERA5_ZARR, consolidated=True, storage_options={"token": "anon"})

    def pick(ds, cands_dict):
        out = {}
        for key, cands in cands_dict.items():
            for c in cands:
                if c in ds.data_vars:
                    out[key] = c
                    break
            else:
                log.warning(f"ERA5 variable not found: {key}")
        return out

    atmos_map = pick(ds, ATMOS_CANDIDATES)
    land_map  = pick(ds, LAND_CANDIDATES)

    ds_atmos = ds[list(atmos_map.values())]
    ds_land  = ds[list(land_map.values())]
    log.info(f"ERA5 atmos vars: {list(ds_atmos.data_vars)}")
    log.info(f"ERA5 land  vars: {list(ds_land.data_vars)}")
    return ds_atmos, ds_land


# ---------------------------------------------------------------------------
# HEALPix regridder
# ---------------------------------------------------------------------------

def build_regridder(ds_atmos):
    sample = ensure_lat_ascending(ds_atmos.isel(time=0))
    nlat   = len(sample.latitude)
    nlon   = len(sample.longitude)
    src_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    hpx_grid = earth2grid.healpix.Grid(
        level=HPX_LEVEL, pixel_order=earth2grid.healpix.XY()
    )
    regridder = earth2grid.get_regridder(src_grid, hpx_grid)
    log.info(f"HEALPix regridder: {nlat}x{nlon} -> 12x{NSIDE}x{NSIDE}")
    return regridder, sample.latitude, sample.longitude


def to_healpix(regridder, channel_stack):
    """[C, lat, lon] -> [C, 12, nside, nside]"""
    return torch.stack(
        [regridder(ch.double()).reshape(12, NSIDE, NSIDE).float()
         for ch in channel_stack],
        dim=0,
    )


# ---------------------------------------------------------------------------
# ERA5 fetcher (runs in background thread for prefetching)
# ---------------------------------------------------------------------------

def fetch_era5_for_day(ds_atmos, ds_land, date_t_str, date_t1_str):
    """
    Fetch and .compute() ERA5 data for one (t, t+1) pair into RAM.
    Designed to run in a background thread while the GPU trains.
    Returns (ds_atm_t, ds_atm_t1, ds_lnd_t) as in-memory xr.Datasets.
    """
    date_t2_str = (datetime.date.fromisoformat(date_t1_str)
                   + datetime.timedelta(days=1)).isoformat()

    atm_win    = ensure_lat_ascending(
                     ds_atmos.sel(time=slice(date_t_str, date_t2_str)))
    ds_atm_day = daily_mean(atm_win)
    ds_atm_t   = pick_single_day(ds_atm_day, date_t_str)
    ds_atm_t1  = pick_single_day(ds_atm_day, date_t1_str)

    lnd_win    = ensure_lat_ascending(
                     ds_land.sel(time=slice(date_t_str, date_t1_str)))
    ds_lnd_day = daily_mean(lnd_win)
    ds_lnd_t   = pick_single_day(ds_lnd_day, date_t_str)

    # Single GCS batch compute — this is the ~150 s network call
    ds_atm_t  = ds_atm_t.compute()
    ds_atm_t1 = ds_atm_t1.compute()
    ds_lnd_t  = ds_lnd_t.compute()

    return ds_atm_t, ds_atm_t1, ds_lnd_t


def fetch_era5_date_range(ds_atmos, ds_land, dates_list):
    """
    Fetch ERA5 for a list of dates in ONE GCS .compute() call.
    Far faster than one call per day for large batches.

    Args:
        dates_list: list of datetime.date objects (the input dates t).
                    Targets are dates_list[i] + 1 day.
    Returns:
        dict mapping date_str -> (ds_atm_t, ds_atm_t1, ds_lnd_t) in-memory.
    """
    if not dates_list:
        return {}

    date_start  = dates_list[0].isoformat()
    # Need t+1 for the last date's target
    date_end_t1 = (dates_list[-1] + datetime.timedelta(days=1)).isoformat()
    # For land we only need t, not t+1
    date_end    = dates_list[-1].isoformat()

    atm_win    = ensure_lat_ascending(
                     ds_atmos.sel(time=slice(date_start, date_end_t1)))
    ds_atm_daily = daily_mean(atm_win).compute()   # single GCS batch

    lnd_win    = ensure_lat_ascending(
                     ds_land.sel(time=slice(date_start, date_end)))
    ds_lnd_daily = daily_mean(lnd_win).compute()   # single GCS batch

    result = {}
    for d in dates_list:
        d_str  = d.isoformat()
        d1_str = (d + datetime.timedelta(days=1)).isoformat()
        try:
            result[d_str] = (
                pick_single_day(ds_atm_daily, d_str),
                pick_single_day(ds_atm_daily, d1_str),
                pick_single_day(ds_lnd_daily, d_str),
            )
        except KeyError as e:
            log.warning(f"  ERA5 date missing in batch: {e}")
    return result


# ---------------------------------------------------------------------------
# Build one training sample (ERA5 already in RAM — only GODAS + channels)
# ---------------------------------------------------------------------------

def build_sample(date_t_str, date_t1_str,
                 ds_atm_t, ds_atm_t1, ds_lnd_t,
                 godas_index, ds_stat,
                 ref_lat, ref_lon, regridder,
                 ds_ocn=None):
    """
    ds_atm_t / ds_atm_t1 / ds_lnd_t must be pre-fetched in-memory datasets
    (from fetch_era5_for_day or fetch_era5_date_range).

    ds_ocn: optional pre-loaded GODAS xr.Dataset (e.g. from a RAM cache).
            If None, loads from godas_index using load_godas_for_date().

    Returns:
        xb: [1, 20, 12, nside, nside]
        yb: [1, 14, 12, nside, nside]
    """
    tag = f"input={date_t_str} -> target={date_t1_str}"
    t0 = time.perf_counter()

    t2 = time.perf_counter()
    if ds_ocn is None:
        log.info(f"  [{tag}] Loading GODAS ocean snapshot ...")
        ds_ocn_t = load_godas_for_date(godas_index, date_t_str)
        log.info(f"  [{tag}] GODAS load done  ({time.perf_counter()-t2:.1f}s)")
    else:
        ds_ocn_t = ds_ocn
        log.debug(f"  [{tag}] GODAS from cache")

    t3 = time.perf_counter()
    log.info(f"  [{tag}] Regriding ocean to ERA5 grid ...")
    ds_ocn_t_rg = ds_ocn_t.interp(
        latitude=ref_lat, longitude=ref_lon, method="linear")
    log.info(f"  [{tag}] Ocean regrid done  ({time.perf_counter()-t3:.1f}s)")

    # align() is only needed for non-ERA5 fields (ocean, static).
    # ERA5 is already on the 0.25-deg grid — skip interp for it.
    def align(da2d):
        return da2d.interp(latitude=ref_lat, longitude=ref_lon, method="linear")

    t4b = time.perf_counter()
    log.info(f"  [{tag}] Building channel tensors (NaN fill + normalize) ...")
    GEOPO = resolve_var(ds_atm_t, ["geopotential", "z"])
    TEMP  = resolve_var(ds_atm_t, ["temperature", "t"])

    ATM_2D = [
        (resolve_var(ds_atm_t, ["2m_temperature", "t2m"]),           "2m_temperature"),
        (resolve_var(ds_atm_t, ["10m_u_component_of_wind", "u10"]),  "10m_u_component_of_wind"),
        (resolve_var(ds_atm_t, ["10m_v_component_of_wind", "v10"]),  "10m_v_component_of_wind"),
        (resolve_var(ds_atm_t, ["total_column_water_vapour", "tcwv"]), "tcwv"),
        (resolve_var(ds_atm_t, ["mean_sea_level_pressure", "msl", "mslp"]), "mean_sea_level_pressure"),
        (resolve_var(ds_atm_t, ["top_net_thermal_radiation", "ttr"]), "top_net_thermal_radiation"),
        (resolve_var(ds_atm_t, ["toa_incident_solar_radiation", "tisr", "toa_insolation"]),
         "toa_incident_solar_radiation"),
    ]

    ATM_PLEV = [
        (GEOPO, 1000, "z_1000"),
        (GEOPO,  850, "z_850"),
        (GEOPO,  700, "z_700"),
        (GEOPO,  500, "z_500"),
        (GEOPO,  300, "z_300"),
        (GEOPO,  250, "z_250"),
        (TEMP,   850, "t_850"),
    ]

    LND = [
        (resolve_var(ds_lnd_t, ["volumetric_soil_water_layer_1", "swvl1"]),
         "volumetric_soil_water_layer_1"),
        (resolve_var(ds_lnd_t, ["soil_temperature_level_1", "stl1"]),
         "soil_temperature_level_1"),
    ]

    OCN = [
        ("potential_temperature", "potential_temperature"),
        ("salinity",              "salinity"),
    ]

    def process_2d(da, norm_key, convert_gph=False, needs_align=False):
        da = to_2d(da)
        if convert_gph:
            da = da / G0
        if needs_align:
            # Non-ERA5 fields (ocean, static) need regridding + NaN fill
            da = fill_latlon_nans_2d(align(da))
        else:
            # ERA5 is already on the 0.25-deg grid — just fill any residual NaNs
            vals = da.values
            if np.isnan(vals).any():
                da = fill_latlon_nans_2d(da)
        return torch.tensor(norm_minmax(da.values, norm_key), dtype=torch.float32)

    # X(t): 20 channels
    X_ch = []
    for v, nk in ATM_2D:
        X_ch.append(process_2d(ds_atm_t[v], nk))

    for base_v, hpa, nk in ATM_PLEV:
        da = select_pressure_level(ds_atm_t[base_v], hpa)
        X_ch.append(process_2d(da, nk,
                               convert_gph=(base_v == GEOPO and CONVERT_GPH_TO_HEIGHT)))

    for v, nk in LND:
        X_ch.append(process_2d(ds_lnd_t[v], nk))

    for v, nk in OCN:
        da = ds_ocn_t_rg[v]
        if "level" in da.dims:
            da = da.isel(level=0)
        X_ch.append(process_2d(da, nk, needs_align=True))

    for sv, nk in [("mask", "landseamask"), ("topo", "elevation")]:
        X_ch.append(process_2d(ds_stat[sv], nk, needs_align=True))

    X_ll = torch.stack(X_ch, dim=0)   # [20, lat, lon]

    # Y(t+1): 14 channels
    Y_ch = []
    for v, nk in ATM_2D:
        Y_ch.append(process_2d(ds_atm_t1[v], nk))

    for base_v, hpa, nk in ATM_PLEV:
        da = select_pressure_level(ds_atm_t1[base_v], hpa)
        Y_ch.append(process_2d(da, nk,
                               convert_gph=(base_v == GEOPO and CONVERT_GPH_TO_HEIGHT)))

    Y_ll = torch.stack(Y_ch, dim=0)   # [14, lat, lon]

    log.info(f"  [{tag}] Channel tensors done  ({time.perf_counter()-t4b:.1f}s)")

    t5 = time.perf_counter()
    log.info(f"  [{tag}] Projecting to HEALPix ...")
    X_hpx = to_healpix(regridder, X_ll)   # [20, 12, 64, 64]
    Y_hpx = to_healpix(regridder, Y_ll)   # [14, 12, 64, 64]
    log.info(f"  [{tag}] HEALPix done  ({time.perf_counter()-t5:.1f}s)")
    log.info(f"  [{tag}] Total sample build: {time.perf_counter()-t0:.1f}s  "
             f"X{list(X_hpx.shape)}  Y{list(Y_hpx.shape)}")

    return X_hpx.unsqueeze(0), Y_hpx.unsqueeze(0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

COND_CHANNELS   = 20
TARGET_CHANNELS = 14
CROSS_ATTN_DIM  = 1   # cross-attention context dim (encoder_hidden_states)


def build_model(device):
    """
    Scaled-up 3D UNet for 98 GB GPU.

    Architecture:
      Level 0  (64×64): DownBlock3D / UpBlock3D  — pure conv, large spatial
      Level 1  (32×32): CrossAttnDownBlock3D      — self-attn: 1024 tokens/face
      Level 2  (16×16): CrossAttnDownBlock3D      — self-attn:  256 tokens/face
      Level 3  ( 8×8 ): CrossAttnDownBlock3D      — self-attn:   64 tokens/face

    Self-attention at levels 1-3 provides global receptive field needed for
    S2S teleconnections (ENSO → global). Cross-attention conditions on
    encoder_hidden_states (currently zero-padded; extend later for e.g. ENSO index).

    block_out_channels=(256,512,1024,1024): ~16× capacity vs. prior (64,128,256,512).
    attention_head_dim=64: 4/8/16 heads at each level (richer per-head repr.).
    norm_num_groups=32: standard for models of this scale.

    NOTE: incompatible with old (64,128,256,512) checkpoints — start fresh.
    """
    model = UNet3DConditionModel(
        sample_size=None,
        in_channels=TARGET_CHANNELS + COND_CHANNELS,
        out_channels=TARGET_CHANNELS,
        layers_per_block=1,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "DownBlock3D",           # level 0: 64×64, pure conv
            "DownBlock3D",           # level 1: 32×32, pure conv
            "DownBlock3D",           # level 2: 16×16, pure conv
            "CrossAttnDownBlock3D",  # level 3:  8×8,    64 tokens/face
        ),
        up_block_types=(
            "CrossAttnUpBlock3D",    # level 3 mirror
            "UpBlock3D",             # level 2 mirror
            "UpBlock3D",             # level 1 mirror
            "UpBlock3D",             # level 0 mirror
        ),
        norm_num_groups=32,
        cross_attention_dim=CROSS_ATTN_DIM,
        attention_head_dim=32,
    ).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # -- Load datasets --
    ds_atmos, ds_land = open_era5()

    log.info(f"Indexing GODAS from {args.godas_dir} ...")
    godas_index = build_godas_index(args.godas_dir)

    log.info(f"Loading static fields from {args.static_dir} ...")
    ds_mask_raw, ds_topo_raw = load_static_fields(args.static_dir)

    # -- Reference grid from ERA5 --
    regridder, ref_lat, ref_lon = build_regridder(ds_atmos)

    # -- Regrid static fields once --
    ds_stat = prepare_static_fields(ds_mask_raw, ds_topo_raw, ref_lat, ref_lon)

    # -- Model + training objects --
    model     = build_model(device)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    loss_fn   = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # -- Load progress (completed dates + best loss) --
    completed_dates, best_loss = load_progress(args.progress_file)

    # -- Load best weights if checkpoint exists --
    if args.checkpoint and os.path.exists(args.checkpoint):
        log.info(f"Loading weights from checkpoint: {args.checkpoint}")
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        except RuntimeError as e:
            log.warning(f"  Checkpoint incompatible with current architecture "
                        f"(likely old model size) — starting from scratch. Error: {e}")

    encoder_hidden = torch.zeros(1, 1, CROSS_ATTN_DIM, device=device)

    # -- Build ordered list of dates still to process --
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
             f"{args.epochs_per_day} epochs/day)")

    if not dates_todo:
        log.info("All dates already completed. Nothing to do.")
        return

    model.train()
    step = len(completed_dates) * args.epochs_per_day  # keep step count consistent

    def _submit_era5(pool, date_t, date_t1):
        """Submit a background ERA5 fetch for (date_t, date_t+1)."""
        return pool.submit(
            fetch_era5_for_day,
            ds_atmos, ds_land,
            date_t.isoformat(), date_t1.isoformat(),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        # Kick off the first pending day's ERA5 fetch immediately
        era5_future = _submit_era5(
            pool, dates_todo[0], dates_todo[0] + datetime.timedelta(days=1))
        log.info(f"  Prefetch started for {dates_todo[0].isoformat()}")

        for i, date_t in enumerate(dates_todo):
            date_t1     = date_t + datetime.timedelta(days=1)
            date_t_str  = date_t.isoformat()
            date_t1_str = date_t1.isoformat()

            # ---- Block until ERA5 for today is ready ----
            t_wait = time.perf_counter()
            try:
                ds_atm_t, ds_atm_t1, ds_lnd_t = era5_future.result()
            except Exception as exc:
                log.warning(f"  SKIP {date_t_str}: ERA5 fetch failed: {exc}")
                if i + 1 < len(dates_todo):
                    nxt = dates_todo[i + 1]
                    era5_future = _submit_era5(pool, nxt, nxt + datetime.timedelta(days=1))
                    log.info(f"  Prefetch started for {nxt.isoformat()}")
                continue
            wait_s = time.perf_counter() - t_wait
            if wait_s < 2.0:
                log.info(f"  [{date_t_str}] ERA5 ready instantly (prefetched)  "
                         f"wait={wait_s:.1f}s")
            else:
                log.info(f"  [{date_t_str}] ERA5 waited {wait_s:.1f}s  "
                         f"(GPU was faster than GCS fetch)")

            # ---- Submit next day's ERA5 fetch in background ----
            if i + 1 < len(dates_todo):
                nxt = dates_todo[i + 1]
                era5_future = _submit_era5(pool, nxt, nxt + datetime.timedelta(days=1))
                log.info(f"  Prefetch started for {nxt.isoformat()}")

            # ---- Build sample (GODAS + channels + HEALPix) ----
            try:
                xb, yb = build_sample(
                    date_t_str, date_t1_str,
                    ds_atm_t, ds_atm_t1, ds_lnd_t,
                    godas_index, ds_stat,
                    ref_lat, ref_lon, regridder,
                )
            except Exception as exc:
                log.warning(f"  SKIP {date_t_str}: {exc}")
                continue

            del ds_atm_t, ds_atm_t1, ds_lnd_t  # free RAM once tensors are built

            x = xb.to(device)
            y = yb.to(device)

            day_losses = []
            for epoch in range(1, args.epochs_per_day + 1):
                noise      = torch.randn_like(y)
                timesteps  = torch.randint(
                    0, scheduler.config.num_train_timesteps, (1,), device=device).long()
                
                # For Flow Matching, the noisy_y is a linear interpolation:
                # x_t = (1 - sigma_t) * x_1 + sigma_t * x_0
                # where x_1 is data (y) and x_0 is noise
                sigmas = scheduler.sigmas.to(device)
                step_indices = [scheduler.index_for_timestep(t.item()) for t in timesteps]
                sigma = sigmas[step_indices].flatten()
                while len(sigma.shape) < len(y.shape):
                    sigma = sigma.unsqueeze(-1)
                
                noisy_y = (1.0 - sigma) * y + sigma * noise
                # Flow matching objective: predict velocity from x_0 (noise) to x_1 (y)
                target_velocity = y - noise

                net_input  = torch.cat([noisy_y, x], dim=1)

                # UNet predicts the velocity
                pred_velocity = model(
                    sample=net_input,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden,
                ).sample

                loss = loss_fn(pred_velocity, target_velocity)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                day_losses.append(loss_val)
                step += 1
                log.info(f"  {date_t_str}  epoch {epoch:02d}/{args.epochs_per_day}"
                         f"  loss={loss_val:.6f}  step={step}")

                del noise, noisy_y, net_input, pred_velocity, loss

            mean_loss = np.mean(day_losses)
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), args.checkpoint)
                log.info(f"  Checkpoint saved  (best_loss={best_loss:.6f})")
                if args.hf_repo:
                    push_checkpoint_to_hub(
                        args.checkpoint, args.hf_repo, best_loss, date_t_str)

            # ---- Mark day done and persist progress ----
            completed_dates.add(date_t_str)
            save_progress(args.progress_file, completed_dates, best_loss)
            log.info(f"  Progress saved  ({len(completed_dates)}/{len(all_dates)} days done)")

            del x, y, xb, yb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    log.info(f"Training complete. Best loss: {best_loss:.6f}")
    log.info(f"Model saved to: {args.checkpoint}")
    log.info(f"Progress saved to: {args.progress_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AI Atmosphere S2S Diffusion Training")
    p.add_argument("--start-date",      default="2018-01-06",
                   help="First training day (YYYY-MM-DD, default: 2018-01-06)")
    p.add_argument("--end-date",        default="2018-08-13",
                   help="Last training day exclusive (YYYY-MM-DD)")
    p.add_argument("--godas-dir",       default=DEFAULT_GODAS_DIR)
    p.add_argument("--static-dir",      default=DEFAULT_STATIC_DIR)
    p.add_argument("--checkpoint",      default=DEFAULT_CHECKPOINT)
    p.add_argument("--progress-file",   default="training_progress.json",
                   help="JSON file tracking completed dates and best loss "
                        "(auto-created; used to resume interrupted runs)")
    p.add_argument("--hf-repo",         default=None,
                   help="HuggingFace repo ID to push best checkpoint to "
                        "(e.g. manmeet3591/ai-atmosphere-s2s). "
                        "Requires HF_TOKEN env var or prior huggingface-cli login.")
    p.add_argument("--epochs-per-day",  type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
