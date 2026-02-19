#!/usr/bin/env python3
"""
Compute min/max for a single ERA5 variable (optionally at a pressure level)
on DAILY MEAN for years 2016-2017 from ARCO ERA5 zarr, writing one JSON.

Designed to be called from a shell script in a loop (or controlled parallelism),
while avoiding memory blow-ups by:
- selecting only the requested variable/level (one DataArray) per run
- computing min/max via one Dask graph (dask.compute)
- single-threaded scheduler + capped BLAS/OpenMP threads
- no construction of a full multi-var dataset

Behavior:
✅ If output JSON exists and is valid, skips computation
✅ Runtime re-check so races across jobs still skip
✅ Atomic write to prevent partial JSONs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Dict

import dask
import xarray as xr


# ---- environment / dask ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
dask.config.set(scheduler="single-threaded")


# ---- helpers ----
def pick_var(ds: xr.Dataset, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in ds.data_vars:
            return c
    if required:
        raise KeyError(f"None found from: {candidates}")
    return None


def find_level_dim(da: xr.DataArray, candidates: Sequence[str] = ("isobaricInhPa", "level", "pressure_level")) -> Optional[str]:
    for c in candidates:
        if c in da.dims:
            return c
    return None


def is_valid_json(path: Path, expected_var: str) -> bool:
    """
    True if:
      - file exists, non-empty
      - JSON dict includes variable/min/max
      - variable matches expected_var
      - min/max numeric
    """
    if not path.exists():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        with path.open("r") as f:
            d = json.load(f)
        return (
            isinstance(d, dict)
            and d.get("variable") == expected_var
            and isinstance(d.get("min"), (int, float))
            and isinstance(d.get("max"), (int, float))
        )
    except Exception:
        return False


def atomic_write_json(path: Path, payload: Dict):
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


# ---- variable mapping ----
CANDIDATES = {
    # surface (kept here for completeness)
    "u10": ["10m_u_component_of_wind", "u10"],
    "v10": ["10m_v_component_of_wind", "v10"],
    "t2m": ["2m_temperature", "t2m"],
    "sp": ["surface_pressure", "sp"],
    "tcwv": ["total_column_water_vapour", "tcwv", "total_column_integrated_water_vapour"],
    "mslp": ["mean_sea_level_pressure", "msl", "mslp"],
    "toa_insolation": [
        "toa_incident_solar_radiation",
        "top_of_atmosphere_incident_solar_radiation",
        "incoming_shortwave_radiation_at_top_of_atmosphere",
        "toa_incoming_shortwave_radiation",
        "incident_solar_radiation_at_top_of_atmosphere",
        "tisr",
    ],
    "olr": [
        "top_of_atmosphere_outgoing_longwave_radiation",
        "toa_outgoing_longwave_radiation",
        "outgoing_longwave_radiation_at_top_of_atmosphere",
        "ttr",
    ],
    # pressure-level
    "u": ["u_component_of_wind", "u"],
    "v": ["v_component_of_wind", "v"],
    "z": ["geopotential", "z"],
    "t": ["temperature", "t"],
    "q": ["specific_humidity", "q"],
}


def canonical_output_name(variable: str, pressure_level: Optional[int]) -> str:
    # match your previous naming for plev: u_50, v_500, ...
    return f"{variable}_{pressure_level}" if pressure_level is not None else variable


def open_arco_era5() -> xr.Dataset:
    # Use conservative chunks to avoid huge reads. Adjust if you know better chunking.
    return xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        consolidated=True,
        storage_options={"token": "anon"},
        chunks={},  # let xarray/dask decide; override if you have a known-good scheme
    )


def compute_minmax_daily_mean_2016_2017(
    variable: str,
    pressure_level: Optional[int],
    out_dir: Path,
) -> Path:
    """
    Compute min/max for ONE variable (optionally selecting one pressure level),
    daily mean over 2016-2017, and write JSON.

    Returns output JSON path (whether computed or skipped).
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = canonical_output_name(variable, pressure_level)
    out_path = out_dir / f"{out_name}.json"

    # Runtime skip check (safe for multi-job scenarios)
    if is_valid_json(out_path, out_name):
        return out_path

    # Do the work
    ds = None
    try:
        ds = open_arco_era5()

        # resolve the underlying dataset variable name
        vname = pick_var(ds, CANDIDATES.get(variable, [variable]), required=True)
        da = ds[vname]

        # select level if requested
        if pressure_level is not None:
            lev_dim = find_level_dim(da)
            if lev_dim is None:
                raise ValueError(f"No pressure level dim found for {variable}/{vname}. dims={da.dims}")
            da = da.sel({lev_dim: pressure_level}, drop=True)

        # daily mean over 2016-2017
        da_daily = da.resample(time="1D").mean().sel(time=slice("2016", "2017"))

        # Re-check before compute in case another job finished while we were opening/setting up
        if is_valid_json(out_path, out_name):
            return out_path

        # Compute min & max in one pass
        min_da = da_daily.min()
        max_da = da_daily.max()
        min_val, max_val = dask.compute(min_da, max_da)

        payload = {
            "variable": out_name,
            "min": float(min_val),
            "max": float(max_val),
        }
        atomic_write_json(out_path, payload)
        return out_path

    except Exception as e:
        err_path = out_dir / f"{out_name}.error.txt"
        with err_path.open("w") as f:
            f.write(repr(e))
        raise
    finally:
        # Ensure dataset closes to release any resources
        try:
            if ds is not None:
                ds.close()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Compute ERA5 daily-mean min/max for one variable/level.")
    p.add_argument("--variable", required=True, help="Variable key, e.g. u, v, z, t, q (or surface vars like t2m).")
    p.add_argument("--pressure-level", type=int, default=None, help="Pressure level in hPa, e.g. 500. Omit for surface.")
    p.add_argument("--out-dir", default="norm_daily_mean_per_var_json", help="Output directory for JSON files.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    out_name = canonical_output_name(args.variable, args.pressure_level)
    out_path = out_dir.resolve() / f"{out_name}.json"

    print("CWD:", Path.cwd().resolve())
    print("OUT_DIR:", out_dir.resolve())
    print("TARGET:", out_name)

    if is_valid_json(out_path, out_name):
        print(f"⏭️  Skipping (already valid): {out_path}")
        return 0

    compute_minmax_daily_mean_2016_2017(args.variable, args.pressure_level, out_dir)
    print(f"✅ Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
