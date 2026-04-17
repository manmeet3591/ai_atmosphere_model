# AI Atmosphere Model - Project Context

## Project Overview
This project trains a diffusion-based 3D UNet to predict next-day atmospheric states from multi-source inputs on a HEALPix grid. The model is designed for sub-seasonal to seasonal (S2S) weather prediction.

The project incorporates both standard training workflows (via notebooks and scripts) and an autonomous Neural Architecture Search (NAS) loop designed to be run by AI agents to optimize the model's architecture and hyperparameters.

## Key Technologies
- **Python, PyTorch** (Deep Learning framework)
- **HuggingFace Diffusers** (`UNet3DConditionModel`, `LCMScheduler`, `DDPMScheduler`)
- **xarray, zarr, dask** (Data manipulation and ERA5/Zarr loading)
- **earth2grid** (HEALPix grid projection)
- **Apptainer** (Containerization for the NAS execution environment)

## Directory Structure & Key Files
- `ai_atmosphere_model_clean.ipynb`: The original single-file implementation of the model and data pipeline.
- `train.py`: Standard training script implementation.
- `train_best.py`: Training script updated with the best performing Neural Architecture Search (NAS) model configuration (`128, 256, 512, 512` channels, cross-attention only at the 8x8 bottleneck).
- `get_data*.py` / `run_min_max.sh`: Utilities for downloading data and computing normalization statistics.
- **Autoresearch (NAS) Files:**
  - `prepare_search.py`: **DO NOT MODIFY**. Fixed infrastructure. One-time data preparation that caches tensors to `search_cache/`.
  - `train_search.py`: **AGENT EDITABLE**. The main NAS script containing model architecture and hyperparameters. Agents modify this file to test new configurations.
  - `autoresearch_loop_v2.py`: The autonomous loop script that continuously mutates `train_search.py`, runs 15-minute training sessions, and logs results.
  - `program_atmos.md`: Instructions for human/AI collaboration on the NAS process.
  - `results_search.tsv`: Experiment logs containing configurations and resulting validation loss.

## Building and Running

### Dependency Installation (Local)
```bash
pip install uv
uv pip install --system "xarray[complete]" zarr gcsfs dask pygrib diffusers accelerate
pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/refs/tags/v2025.7.1.tar.gz
pip install pygrib==2.1.6
```

### NAS Workflow (via Apptainer)
The NAS workflow runs inside an Apptainer container (`ai_atmosphere.sif`) using an RTX PRO 6000.
1. **Prepare Cache (Run Once):**
   ```bash
   CUDA_VISIBLE_DEVICES=1 apptainer exec --nv --env PYTHONNOUSERSITE=1 \
     --bind /media/airlab/ROCSTOR:/media/airlab/ROCSTOR \
     /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/ai_atmosphere.sif \
     python3 prepare_search.py \
     --godas-dir /media/airlab/ROCSTOR/earthmind_s2s/godas_pentad \
     --static-dir /media/airlab/ROCSTOR/earthmind_s2s/static_fields
   ```
2. **Run Single Experiment:**
   ```bash
   CUDA_VISIBLE_DEVICES=1 apptainer exec --nv --env PYTHONNOUSERSITE=1 \
     --bind /media/airlab/ROCSTOR:/media/airlab/ROCSTOR \
     /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/ai_atmosphere.sif \
     python3 train_search.py > run_search.log 2>&1
   ```

## Development Conventions
- **Fixed Infrastructure:** Never modify `prepare_search.py`.
- **Normalization:** All normalization statistics are hardcoded in the `NORM` dict within the code. Updating variables or data ranges requires manual updates to this dictionary.
- **Grid Layout:** The project uses a HEALPix level 6 grid (`nside=64`). Tensors are formatted as `[batch, channels, 12, 64, 64]`.
- **NAS Metric:** The key metric to optimize during NAS is `val_loss` (MSE on noise prediction for held-out validation days). Lower is better.
