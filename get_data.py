#!/usr/bin/env python3
"""
Compute per-variable min/max on daily-mean ERA5 (2016-2017) from ARCO ERA5 zarr,
writing one JSON per variable, and SKIPPING variables whose JSON already exists
and is valid.

Key behavior you asked for:
✅ If a valid JSON for a variable is present, it will NOT compute that variable at all.
✅ It also checks existence *per variable at runtime* (not just pre-scan), so even if
   another process/job wrote the JSON after the script started, it still skips.

Robust features:
- Prints CWD and absolute OUT_DIR (helps catch path mismatch)
- Pre-scans existing valid JSONs into a set
- Per-var “stop immediately if json exists” check (covers multi-run / multi-job scenarios)
- Treats empty/partial/corrupt JSONs as invalid (recomputes)
- Atomic write (tmp -> replace) to avoid partial JSONs on interruption
- Single-threaded dask scheduler + thread env vars to avoid OpenMP explosions
"""

import json
import os
from pathlib import Path

import dask
import xarray as xr
from tqdm import tqdm


# ---- environment / dask ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
dask.config.set(scheduler="single-threaded")


# ---- helpers ----
def pick_var(ds, candidates, required=True):
    for c in candidates:
        if c in ds.data_vars:
            return c
    if required:
        raise KeyError(f"None found from: {candidates}")
    return None


def find_level_dim(da, candidates=("isobaricInhPa", "level", "pressure_level")):
    for c in candidates:
        if c in da.dims:
            return c
    return None


def is_valid_json(path: Path, var: str) -> bool:
    """
    Return True if 'path' exists and contains a valid JSON for 'var'.
    Valid criteria:
      - non-empty
      - JSON dict includes: variable, min, max
      - variable == var
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
            and d.get("variable") == var
            and isinstance(d.get("min"), (int, float))
            and isinstance(d.get("max"), (int, float))
        )
    except Exception:
        return False


def scan_valid_jsons(out_dir: Path) -> set[str]:
    """Pre-scan all valid jsons in out_dir and return set of var names."""
    valid = set()
    for p in out_dir.glob("*.json"):
        var = p.stem
        if is_valid_json(p, var):
            valid.add(var)
    return valid


def atomic_write_json(path: Path, payload: dict):
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main():
    # ---- open dataset ----
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        consolidated=True,
        storage_options={"token": "anon"},
    )
    print(ds)

    # ---- candidate mapping ----
    cand = {
        # surface
        "u10": ["10m_u_component_of_wind", "u10"],
        "v10": ["10m_v_component_of_wind", "v10"],
        "t2m": ["2m_temperature", "t2m"],
        "sp": ["surface_pressure", "sp"],
        "tcwv": [
            "total_column_water_vapour",
            "tcwv",
            "total_column_integrated_water_vapour",
        ],
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

    surface_keys = ["u10", "v10", "t2m", "sp", "tcwv", "mslp", "toa_insolation", "olr"]
#    plev_keys = ["u", "v", "z", "t", "q"]
    plev_keys = ["t", "q"]
    PLEV = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # 13

    resolved = {k: pick_var(ds, v, required=False) for k, v in cand.items()}

    # ---- build split dataset (var-per-level) ----
    out = {}

#    for k in surface_keys:
#        vname = resolved.get(k)
#        if vname is not None:
#            out[k] = ds[vname]

    for k in plev_keys:
        vname = resolved.get(k)
        if vname is None:
            continue
        da = ds[vname]
        lev_dim = find_level_dim(da)
        if lev_dim is None:
            raise ValueError(f"No pressure level dim found for {k}/{vname}. dims={da.dims}")
        for lev in PLEV:
            out[f"{k}_{lev}"] = da.sel({lev_dim: lev}, drop=True)

    ds_split = xr.Dataset(out)

    # ---- daily mean (2016-2017) ----
    ds_daily = ds_split.resample(time="1D").mean().sel(time=slice("2016", "2017"))

    # ---- output dir ----
    # IMPORTANT: use an absolute scratch path if your CWD changes between runs
    out_dir = Path("norm_daily_mean_per_var_json").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("CWD:", Path.cwd().resolve())
    print("OUT_DIR:", out_dir)

    # Pre-scan once (fast)
    valid_existing = scan_valid_jsons(out_dir)
    print(f"JSON count in out_dir: {len(list(out_dir.glob('*.json')))}")
    print(f"Found {len(valid_existing)} valid existing jsons (pre-scan).")

    skipped = 0
    computed = 0
    failed = 0

    for var in tqdm(list(ds_daily.data_vars), desc="Computing min/max (daily mean)"):
        out_path = out_dir / f"{var}.json"
        print(out_path)
        print(var)
        print(valid_existing)
        print(is_valid_json(out_path, var))

        # ✅ STOP IMMEDIATELY if JSON already exists and is valid (runtime check)
        # This guarantees we never compute a var that already has a valid json.
        if var in valid_existing or is_valid_json(out_path, var):
            skipped += 1
            valid_existing.add(var)  # keep set up-to-date
            continue

        try:
            da = ds_daily[var]
            stats = {
                "variable": var,
                "min": float(da.min().compute()),
                "max": float(da.max().compute()),
            }
            atomic_write_json(out_path, stats)
            computed += 1

            # update in-memory set so even later in the same run it won't recompute
            valid_existing.add(var)

        except Exception as e:
            failed += 1
            err_path = out_dir / f"{var}.error.txt"
            with err_path.open("w") as f:
                f.write(repr(e))

    print(f"✅ Output dir: {out_dir}")
    print(f"✅ Computed: {computed}")
    print(f"⏭️  Skipped (already valid): {skipped}")
    print(f"⚠️  Failed: {failed}")


if __name__ == "__main__":
    main()
