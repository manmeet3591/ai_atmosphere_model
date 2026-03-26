"""
train_search.py — Agent-editable NAS training script for the S2S diffusion model.

Autoresearch loop: the AI agent modifies THIS file, runs it, checks val_loss,
keeps or discards the change, and repeats until stopped.

DO NOT modify prepare_search.py — it is fixed infrastructure.

Usage:
    python train_search.py [--cache-dir search_cache]

Requires:
    python prepare_search.py   # run once to build the cache

Output (always printed at the very end, one key per line for easy grep):
    val_loss:         0.123456
    training_seconds: 900.0
    total_seconds:    910.0
    peak_vram_mb:     45000.0
    num_steps:        1200
    num_params_M:     93.9
    block_channels:   (256,512,1024,1024)
    layers_per_block: 2
    lr:               0.0001
"""

import os
import time
import argparse
import glob
import math

import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import UNet3DConditionModel, LCMScheduler, DDPMScheduler

# ---------------------------------------------------------------------------
# Fixed constants — do NOT change
# ---------------------------------------------------------------------------
CACHE_DIR          = "search_cache"
COND_CHANNELS      = 20   # X channels
TARGET_CHANNELS    = 14   # Y channels
IN_CHANNELS        = TARGET_CHANNELS + COND_CHANNELS  # 34
TRAIN_TIME_BUDGET  = 900  # wall-clock seconds of pure training (15 min)

# ---------------------------------------------------------------------------
# MODEL CONFIGURATION — agent modifies everything in this section
# ---------------------------------------------------------------------------

# UNet3D architecture
BLOCK_OUT_CHANNELS   = (160, 320, 640, 640)     # wider channels with confirmed best lpb=1
LAYERS_PER_BLOCK     = 1                         # confirmed best: more steps wins
NORM_NUM_GROUPS      = 32                        # GroupNorm groups (must divide all channels)
CROSS_ATTN_DIM       = 1                         # cross-attention dim (keep at 1)
ATTENTION_HEAD_DIM   = 32                        # attention head dim

# CrossAttn at levels 2-3 (16×16=3072 + 8×8=768 tokens = 3840 total).
# More global receptive field for teleconnections than level 3 only.
DOWN_BLOCK_TYPES = (
    "DownBlock3D",            # level 0: 64×64, pure conv
    "DownBlock3D",            # level 1: 32×32, pure conv
    "CrossAttnDownBlock3D",   # level 2: 16×16, 3072 tokens — intermediate attn
    "CrossAttnDownBlock3D",   # level 3:  8×8,   768 tokens — bottleneck attn
)
UP_BLOCK_TYPES = (
    "CrossAttnUpBlock3D",     # mirror of level 3
    "CrossAttnUpBlock3D",     # mirror of level 2
    "UpBlock3D",              # mirror of level 1
    "UpBlock3D",              # mirror of level 0
)

# Diffusion scheduler — try LCMScheduler or DDPMScheduler
SCHEDULER_CLASS    = DDPMScheduler
NUM_TRAIN_TIMESTEPS = 1000

# Optimizer
LR           = 2e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP    = 1.0   # set to None to disable gradient clipping

# Batch size
BATCH_SIZE = 4

# LR warmup steps (linear ramp from 0 → LR); 0 = no warmup
WARMUP_STEPS = 200

# ---------------------------------------------------------------------------
# Fixed infrastructure — do NOT modify below this line
# ---------------------------------------------------------------------------

def build_model(device):
    model = UNet3DConditionModel(
        sample_size=None,
        in_channels=IN_CHANNELS,
        out_channels=TARGET_CHANNELS,
        layers_per_block=LAYERS_PER_BLOCK,
        block_out_channels=BLOCK_OUT_CHANNELS,
        down_block_types=DOWN_BLOCK_TYPES,
        up_block_types=UP_BLOCK_TYPES,
        norm_num_groups=NORM_NUM_GROUPS,
        cross_attention_dim=CROSS_ATTN_DIM,
        attention_head_dim=ATTENTION_HEAD_DIM,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"# Model: {n_params:.1f}M params  channels={BLOCK_OUT_CHANNELS}")
    return model, n_params


def load_cache(split, cache_dir, device):
    """Load all .pt files for a split into a list of (X, Y) tensors."""
    pattern = os.path.join(cache_dir, split, "*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No cached tensors found in {os.path.join(cache_dir, split)}/\n"
            f"Run: python prepare_search.py"
        )
    samples = []
    for f in files:
        d = torch.load(f, map_location=device)
        samples.append((d["X"], d["Y"]))
    print(f"# Loaded {len(samples)} {split} samples from {cache_dir}/{split}/")
    return samples


def make_batch(samples, batch_size, device):
    """Randomly sample a mini-batch from the cached samples."""
    import random
    chosen = random.choices(samples, k=batch_size)
    X = torch.cat([s[0] for s in chosen], dim=0)  # [B, 20, 12, 64, 64]
    Y = torch.cat([s[1] for s in chosen], dim=0)  # [B, 14, 12, 64, 64]
    return X.to(device), Y.to(device)


def evaluate(model, scheduler, val_samples, device, n_eval_steps=50):
    """Compute average val MSE over n_eval_steps random batches."""
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_eval_steps):
            X, Y = make_batch(val_samples, batch_size=1, device=device)
            N = X.shape[0]
            enc_hid   = torch.zeros(N, 1, CROSS_ATTN_DIM, device=device)
            noise     = torch.randn_like(Y)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (N,), device=device).long()
            noisy_Y   = scheduler.add_noise(Y, noise, timesteps)
            net_input = torch.cat([noisy_Y, X], dim=1)
            pred      = model(
                sample=net_input,
                timestep=timesteps,
                encoder_hidden_states=enc_hid,
            ).sample
            total_loss += loss_fn(pred, noise).item()
    model.train()
    return total_loss / n_eval_steps


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# Device: {device}")

    t_total_start = time.perf_counter()

    train_samples = load_cache("train", args.cache_dir, device)
    val_samples   = load_cache("val",   args.cache_dir, device)

    model, n_params = build_model(device)
    scheduler = SCHEDULER_CLASS(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    loss_fn   = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    peak_vram_start = torch.cuda.memory_allocated(device) if device == "cuda" else 0

    # ---- Training loop with fixed wall-clock budget ----
    model.train()
    step = 0
    t_train_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_train_start
        if elapsed >= TRAIN_TIME_BUDGET:
            break

        X, Y = make_batch(train_samples, BATCH_SIZE, device)
        N    = X.shape[0]
        enc_hid   = torch.zeros(N, 1, CROSS_ATTN_DIM, device=device)
        noise     = torch.randn_like(Y)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (N,), device=device).long()
        noisy_Y   = scheduler.add_noise(Y, noise, timesteps)
        net_input = torch.cat([noisy_Y, X], dim=1)

        noise_pred = model(
            sample=net_input,
            timestep=timesteps,
            encoder_hidden_states=enc_hid,
        ).sample

        loss = loss_fn(noise_pred, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP is not None:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Linear LR warmup
        if WARMUP_STEPS > 0 and step < WARMUP_STEPS:
            lr_scale = (step + 1) / WARMUP_STEPS
            for pg in optimizer.param_groups:
                pg['lr'] = LR * lr_scale

        step += 1
        if step % 50 == 0:
            print(f"  step={step}  loss={loss.item():.6f}  "
                  f"elapsed={time.perf_counter()-t_train_start:.0f}s")

        del X, Y, enc_hid, noise, noisy_Y, net_input, noise_pred, loss

    training_seconds = time.perf_counter() - t_train_start

    # ---- Validation ----
    val_loss = evaluate(model, scheduler, val_samples, device, n_eval_steps=50)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device == "cuda" else 0.0
    )
    total_seconds = time.perf_counter() - t_total_start

    # ---- Standard output (autoresearch format — grep-friendly) ----
    print("---")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params:.1f}")
    print(f"block_channels:   {BLOCK_OUT_CHANNELS}")
    print(f"layers_per_block: {LAYERS_PER_BLOCK}")
    print(f"lr:               {LR}")


def parse_args():
    p = argparse.ArgumentParser(
        description="NAS training script for S2S diffusion model (agent edits this)")
    p.add_argument("--cache-dir", default=CACHE_DIR)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
