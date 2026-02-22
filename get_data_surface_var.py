#!/usr/bin/env python3
"""
Compute min/max for ONE variable (exact dataset variable name passed as argument)
on daily-mean ERA5 (2016-2017) from ARCO ERA5 zarr. Writes ONE JSON: <var>.json

Behavior:
- Only computes the requested variable.
- If <var>.json exists and is valid, it does nothing.
- Atomic write to avoid partial JSONs.
"""

import argparse
import json
import os
from pathlib import Path

import dask
import xarray as xr


# ---- environment / dask ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
dask.config.set(scheduler="single-threaded")


def is_valid_json(path: Path, var: str) -> bool:
    """Return True if path exists and contains a valid JSON for var."""
    if not path.exists():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        with path.open("r") as f:
            d = json.load(f)
        return (
            isinstance(d, dict)
            and d.get("variable") == var
            and isinstance(d.get("min"), (int, float))
            and isinstance(d.get("max"), (int, float))
        )
    except Exception:
        return False


def atomic_write_json(path: Path, payload: dict):
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--var",
        required=True,
        help="Exact ERA5 variable name in the zarr store (must match ds.data_vars key).",
    )
    parser.add_argument(
        "--out-dir",
        default="norm_daily_mean_per_var_json",
        help="Directory to write <var>.json",
    )
    args = parser.parse_args()

    var = args.var.strip()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{var}.json"

    print("CWD:", Path.cwd().resolve())
    print("OUT_DIR:", out_dir)
    print("VAR:", var)
    print("OUT_PATH:", out_path)

    # ✅ skip if already done
    if is_valid_json(out_path, var):
        print(f"⏭️  Skipping '{var}' (valid JSON already exists).")
        return

    # ---- open dataset ----
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        consolidated=True,
        storage_options={"token": "anon"},
    )

    # Will raise KeyError naturally if var doesn't exist (as requested: no mapping / no pre-check)
    da = ds[var]

    # ---- daily mean (2016-2017) ----
    da_daily = da.resample(time="1D").mean().sel(time=slice("2016", "2017"))

    # ---- compute stats ----
    stats = {
        "variable": var,
        "min": float(da_daily.min().compute()),
        "max": float(da_daily.max().compute()),
    }

    atomic_write_json(out_path, stats)
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()
