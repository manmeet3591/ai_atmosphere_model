# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project trains a **diffusion-based 3D UNet** to predict next-day atmospheric state from multi-source inputs on a HEALPix grid. It is implemented as a single Jupyter notebook: `ai_atmosphere_model_clean.ipynb`.

The model is designed for **sub-seasonal to seasonal (S2S) weather prediction**.

## Running the Notebook

This is a Google Colab-oriented notebook. To run locally:

```bash
# Install dependencies (as done in the notebook)
pip install uv
uv pip install --system "xarray[complete]" zarr gcsfs dask pygrib diffusers accelerate
pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/refs/tags/v2025.7.1.tar.gz
pip install pygrib==2.1.6
```

Run the notebook cell-by-cell in order — each step depends on state from previous steps (datasets remain in memory).

## Architecture

### Data Pipeline (Steps 2–9)

**Data sources:**
- **ERA5** (atmosphere + land): loaded from GCS Zarr (`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`) using `xarray` + `gcsfs` with anonymous auth
- **GODAS** (ocean): GRIB pentad files downloaded from NOAA FTP, read with `pygrib`
- **Static fields**: IMERG land-sea mask (NASA) + GMTED2010 topography (KNMI/TEMIS)

**Processing steps:** resample to daily means → interpolate all sources to 0.25° lat-lon grid → fill NaNs (linear interp + KDTree nearest-neighbor fallback) → min-max normalize using hardcoded `NORM` dict → project to HEALPix using `earth2grid`

**Grid:** HEALPix level 6 (`nside=64`), XY pixel order — shape `[12, 64, 64]` per variable

### Tensor Layout

| Tensor | Shape | Contents |
|--------|-------|----------|
| X(t) input | `[1, 20, 12, 64, 64]` | 7 atmos surface + 7 pressure-level + 2 land + 2 ocean + 2 static |
| Y(t+1) target | `[1, 14, 12, 64, 64]` | 7 atmos surface + 7 pressure-level (next day only) |

**Pressure levels used:** 1000, 850, 700, 500, 300, 250 hPa for geopotential (converted to geopotential height by dividing by g=9.80665); 850 hPa for temperature.

### Model (Step 10)

- **Architecture:** HuggingFace `diffusers.UNet3DConditionModel` (~93.9M parameters)
- **Input:** `[batch, 34, 12, 64, 64]` — noisy Y(t+1) concatenated with X(t) along channel dim
- **Output:** `[batch, 14, 12, 64, 64]` — predicted noise in Y space
- **Scheduler:** `LCMScheduler` (1000 timesteps)
- **Optimizer:** AdamW, lr=1e-4, weight_decay=1e-2
- **Loss:** MSE between predicted and actual noise (standard DDPM objective)
- Cross-attention is present but uses a dummy `[1, 1, 1]` encoder hidden state (not meaningfully used)

### Training Loop (Step 12)

Iterates over consecutive days within a year (`YEAR`, `START_DOY`, `END_DOY`). For each day:
1. Calls `build_sample_for_day(date_t, date_t1, date_t2)` to construct X/Y tensors
2. Runs `EPOCHS_PER_DAY` diffusion training steps with random noise timesteps
3. Saves best checkpoint to `best_diffusion_atmos_model.pth`

**Known issue:** Training skips days when ERA5 data for that date isn't available in the GCS Zarr store. The ERA5 dataset covers 1940–present but the Zarr store may not have all dates indexed.

## Key Configuration

All normalization statistics are hardcoded in the `NORM` dict (Step 9). If adding new variables or using different data ranges, these must be updated manually.

```python
HPX_LEVEL = 6      # nside = 64
CONVERT_GPH_TO_HEIGHT = True  # divide geopotential by g before normalizing
YEAR = 2018
EPOCHS_PER_DAY = 5
```

## Important Functions

- `build_sample_for_day(date_t_str, date_t1_str, date_t2_str)` — end-to-end pipeline for one training sample
- `to_healpix(channel_stack)` — converts `[C, lat, lon]` tensor to `[C, 12, nside, nside]`
- `fill_nan_nearest_2d(da2d)` — KDTree-based NaN filling for static fields (handles coastal gaps after linear interp)
- `norm_minmax(arr, vname)` — applies hardcoded min-max normalization
- `ocean_nearest_prior(ds_ocean, target_time)` — selects most recent ocean snapshot before target date (GODAS is pentad, not daily)
