# autoresearch — AI Atmosphere S2S Model

Neural architecture search for a diffusion-based 3D UNet that predicts next-day
atmospheric state from multi-source inputs on a HEALPix grid (S2S weather).

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** for full context:
   - `program_atmos.md` — this file. Agent instructions.
   - `prepare_search.py` — fixed constants, data caching. Do NOT modify.
   - `train_search.py` — the file you modify: model arch, optimizer, hyperparams.
4. **Verify cache exists**: Check that `search_cache/train/` and `search_cache/val/`
   each contain `.pt` files. If not, run: `python prepare_search.py`
   (this takes ~30-60 minutes to fetch ERA5 + build HEALPix tensors — run once).
5. **Initialize results.tsv**: Create `results_search.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good, then begin the experiment loop.

## Context: What is this model?

This is a **diffusion-based 3D UNet** (`UNet3DConditionModel` from HuggingFace
diffusers) trained to predict the **next-day atmospheric state** from today's
multi-source observations. It operates on a **HEALPix level-6 grid** (nside=64),
where each face is 64x64, and there are 12 faces total.

**Tensor layout:**
- X(t) input:   `[B, 20, 12, 64, 64]` — 7 atmos surface + 7 pressure-level +
                                          2 land + 2 ocean + 2 static fields
- Y(t+1) target: `[B, 14, 12, 64, 64]` — 7 atmos surface + 7 pressure-level (next day)
- UNet input:   `[B, 34, 12, 64, 64]` — noisy Y(t+1) concatenated with X(t)
- UNet output:  `[B, 14, 12, 64, 64]` — predicted noise (standard DDPM objective)

**Training objective:** MSE between predicted noise and actual noise (DDPM).
Lower val_loss = better model.

**Hardware:** Single NVIDIA RTX PRO 6000 Blackwell (98 GB VRAM).
Run with `CUDA_VISIBLE_DEVICES=1`.

**Time budget:** 15 minutes (900 seconds) of pure training per experiment.
This gives ~12 experiments per day if run back-to-back.

## What to search

**In `train_search.py`, everything in the "MODEL CONFIGURATION" section is fair game:**

Architecture:
- `BLOCK_OUT_CHANNELS` — channel sizes per resolution level, e.g. `(128,256,512,512)`
  or `(256,512,1024,1024)` or even deeper `(64,128,256,512,512)`
- `LAYERS_PER_BLOCK` — 1, 2, or 3 ResNet layers per block
- `DOWN_BLOCK_TYPES` / `UP_BLOCK_TYPES` — which levels have cross-attention
  (attention is expensive at high resolution; consider `DownBlock3D` at levels 0-1)
- `ATTENTION_HEAD_DIM` — try 32, 64, or 128
- `NORM_NUM_GROUPS` — 8, 16, or 32 (must divide all channel sizes)

Optimizer:
- `LR` — try 3e-5, 1e-4, 3e-4
- `WEIGHT_DECAY` — 0, 1e-3, 1e-2, 1e-1
- `GRAD_CLIP` — try None, 0.5, 1.0, 2.0

Scheduler:
- `SCHEDULER_CLASS` — `LCMScheduler` vs `DDPMScheduler`
- `NUM_TRAIN_TIMESTEPS` — 100, 500, 1000

Batch:
- `BATCH_SIZE` — 1, 2, 4, 8 (higher = more stable gradients, but uses more VRAM)

## What you CANNOT do

- Modify `prepare_search.py` — it is read-only fixed infrastructure.
- Modify `train_multi_days.py` or `train.py` — these are the production scripts.
- Change the training time budget constant `TRAIN_TIME_BUDGET` in `train_search.py`.
- Change the fixed constants (`COND_CHANNELS`, `TARGET_CHANNELS`, `IN_CHANNELS`,
  `CACHE_DIR`, `N_EVAL_STEPS`) — these define the benchmark.
- Add new Python packages.

## The goal

**Minimize `val_loss`** (MSE on noise prediction for held-out validation days).
Since the time budget is fixed, focus on architectures that learn fastest within
15 minutes on one RTX PRO 6000.

VRAM is a soft constraint: some increase is fine for meaningful val_loss gains,
but OOM (crash) means the experiment failed.

Simplicity: a 0.001 val_loss improvement that adds complex code may not be worth it.
A similar improvement from deleting/simplifying code? Always keep.

## Running an experiment

```bash
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv --env PYTHONNOUSERSITE=1 \
  --bind /media/airlab/ROCSTOR:/media/airlab/ROCSTOR \
  /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/ai_atmosphere.sif \
  python3 /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/train_search.py \
  > run_search.log 2>&1
```

Extract the key metric:
```bash
grep "^val_loss:" run_search.log
```

## Logging results

Log to `results_search.tsv` (tab-separated, NOT commas — they break in descriptions).

Header and columns:
```
commit	val_loss	memory_gb	status	description
```

- `commit` — short git hash (7 chars)
- `val_loss` — val MSE achieved (e.g. 0.123456) — use 0.000000 for crashes
- `memory_gb` — peak_vram_mb / 1024, rounded to .1f — use 0.0 for crashes
- `status` — `keep`, `discard`, or `crash`
- `description` — short text of what was tried

Do NOT commit `results_search.tsv` — leave it untracked.

## The experiment loop

LOOP FOREVER (do not pause to ask the human if you should continue):

1. Check current git state and branch.
2. Modify `train_search.py` with one experimental idea.
3. `git commit` the change.
4. Run: `CUDA_VISIBLE_DEVICES=1 python train_search.py > run_search.log 2>&1`
5. Read results: `grep "^val_loss:\|^peak_vram_mb:" run_search.log`
6. If grep is empty, run crashed — check `tail -n 50 run_search.log` for traceback.
   Fix easy bugs and re-run. For fundamental failures, log "crash" and move on.
7. Record in `results_search.tsv`.
8. If val_loss improved (lower): keep the commit, advance the branch.
9. If val_loss equal or worse: `git reset --hard HEAD~1` to discard.

Timeout: if a run exceeds 25 minutes, kill it (`kill <pid>`) and treat as crash.

**The first run establishes the baseline — run `train_search.py` as-is before making any changes.**

**NEVER STOP**: once the experiment loop begins, continue indefinitely until manually interrupted. If stuck for ideas: vary one thing at a time, try combinations of previously successful changes, look at what architectural patterns work well for 3D spatiotemporal data (weather transformers, video diffusion), try different noise schedules, etc.
