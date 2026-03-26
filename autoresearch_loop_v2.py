#!/usr/bin/env python3
"""
autoresearch_loop_v2.py — Fixed autonomous NAS loop: 1000 experiments × 30 min each.

Robust mutation logic: directly modifies Python variable values, not regex.
Runs continuously, tracks best config, logs all results to results_search.tsv.
"""

import os
import subprocess
import time
import re
import random
import ast
import datetime
from pathlib import Path

WORKDIR = "/media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model"
TRAIN_SCRIPT = os.path.join(WORKDIR, "train_search.py")
RESULTS_FILE = os.path.join(WORKDIR, "results_search.tsv")
RUN_LOG = os.path.join(WORKDIR, "run_search.log")
NUM_EXPERIMENTS = 1000
TIMEOUT_SECONDS = 30 * 60 + 300  # 30 min budget + 5 min overhead

os.chdir(WORKDIR)

# Read current best from results
best_loss = float('inf')
best_commit = None
try:
    with open(RESULTS_FILE) as f:
        for line in f:
            if line.startswith('commit') or '\t' not in line:
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    loss = float(parts[1])
                    if 0 < loss < best_loss:
                        best_loss = loss
                        best_commit = parts[0]
                except:
                    pass
except:
    pass

print(f"[loop] Starting from best: {best_commit} (loss={best_loss:.6f})" if best_commit else "[loop] Starting fresh")

# Hyperparameter ranges
PARAM_RANGES = {
    'BLOCK_OUT_CHANNELS': [
        (64, 128, 256, 256),
        (96, 192, 384, 384),
        (112, 224, 448, 448),
        (128, 256, 512, 512),
        (160, 320, 640, 640),
        (80, 160, 320, 320),
    ],
    'LAYERS_PER_BLOCK': [1, 2],
    'LR': [1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3],
    'WEIGHT_DECAY': [0, 1e-3, 1e-2, 1e-1],
    'WARMUP_STEPS': [0, 100, 200, 300, 500],
    'BATCH_SIZE': [1, 2, 4],
    'GRAD_CLIP': [None, 0.5, 1.0, 2.0],
    'ATTENTION_HEAD_DIM': [16, 32, 64],
}

def read_train_script():
    """Read train_search.py as Python AST to safely modify it."""
    with open(TRAIN_SCRIPT) as f:
        return f.read()

def mutate_single_param(content, param, value):
    """
    Mutate a single parameter. Safely replaces the value assignment.
    Handles tuples, numbers, None, etc.
    """
    if param == 'BLOCK_OUT_CHANNELS':
        # Format: BLOCK_OUT_CHANNELS   = (128, 256, 512, 512)
        pattern = r'(BLOCK_OUT_CHANNELS\s*=\s*)\([^)]+\)'
        replacement = f'\\1{value}'
    elif param == 'LAYERS_PER_BLOCK':
        pattern = r'(LAYERS_PER_BLOCK\s*=\s*)\d+'
        replacement = f'\\1{value}'
    elif param == 'LR':
        pattern = r'(LR\s*=\s*)[\d.e\-]+'
        replacement = f'\\1{value:.1e}'
    elif param == 'WEIGHT_DECAY':
        if value == 0:
            pattern = r'(WEIGHT_DECAY\s*=\s*)[\d.e\-]+'
            replacement = f'\\10'
        else:
            pattern = r'(WEIGHT_DECAY\s*=\s*)[\d.e\-]+'
            replacement = f'\\1{value:.1e}'
    elif param == 'WARMUP_STEPS':
        pattern = r'(WARMUP_STEPS\s*=\s*)\d+'
        replacement = f'\\1{value}'
    elif param == 'BATCH_SIZE':
        pattern = r'(BATCH_SIZE\s*=\s*)\d+'
        replacement = f'\\1{value}'
    elif param == 'GRAD_CLIP':
        if value is None:
            pattern = r'(GRAD_CLIP\s*=\s*)(?:None|[\d.]+)'
            replacement = r'\\1None'
        else:
            pattern = r'(GRAD_CLIP\s*=\s*)(?:None|[\d.]+)'
            replacement = f'\\1{value}'
    elif param == 'ATTENTION_HEAD_DIM':
        pattern = r'(ATTENTION_HEAD_DIM\s*=\s*)\d+'
        replacement = f'\\1{value}'
    else:
        return content

    new_content = re.sub(pattern, replacement, content)

    # Verify the substitution actually happened
    if new_content == content:
        print(f"  [warn] Mutation failed for {param}={value}, pattern didn't match")
        return None

    return new_content

def extract_val_loss(logfile):
    """Read val_loss from run_search.log"""
    try:
        with open(logfile) as f:
            for line in f:
                if line.startswith('val_loss:'):
                    return float(line.split()[-1])
    except:
        pass
    return None

def run_experiment(exp_num):
    """Run one experiment: mutate → commit → run → check result"""
    global best_loss, best_commit

    print(f"\n[exp{exp_num}] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Read current script
    content = read_train_script()

    # Mutate one hyperparameter
    param = random.choice(list(PARAM_RANGES.keys()))
    value = random.choice(PARAM_RANGES[param])

    print(f"  {param} = {value}")

    # Apply mutation
    new_content = mutate_single_param(content, param, value)
    if new_content is None:
        print(f"  [skip] Mutation failed, skipping")
        return False

    # Write mutated script
    with open(TRAIN_SCRIPT, 'w') as f:
        f.write(new_content)

    # Commit
    result = subprocess.run(
        ["git", "add", "train_search.py"],
        cwd=WORKDIR,
        capture_output=True
    )

    msg = f"exp{exp_num}: {param}={value}"
    result = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=WORKDIR,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  [warn] Git commit failed: {result.stderr[:100]}")
        subprocess.run(["git", "checkout", "train_search.py"], cwd=WORKDIR, capture_output=True)
        return False

    # Get commit hash
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=WORKDIR,
        capture_output=True,
        text=True
    ).stdout.strip()

    # Run with timeout
    try:
        result = subprocess.run(
            ["bash", "-c",
             "CUDA_VISIBLE_DEVICES=0 apptainer exec --nv --env PYTHONNOUSERSITE=1 "
             "--bind /media/airlab/ROCSTOR:/media/airlab/ROCSTOR "
             f"{WORKDIR}/ai_atmosphere.sif "
             f"python3 {TRAIN_SCRIPT} > {RUN_LOG} 2>&1"],
            cwd=WORKDIR,
            timeout=TIMEOUT_SECONDS,
            capture_output=False
        )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
        print(f"  [timeout] Experiment exceeded {TIMEOUT_SECONDS}s")

    # Read result
    val_loss = extract_val_loss(RUN_LOG)

    if exit_code == 0 and val_loss is not None and val_loss > 0:
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_commit = commit
            print(f"  ✓ val_loss={val_loss:.6f} NEW BEST")
            status = "keep"
        else:
            print(f"  ✗ val_loss={val_loss:.6f} (best={best_loss:.6f})")
            status = "discard"
            # Reset if not better
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=WORKDIR, capture_output=True)
    else:
        status = "crash"
        val_loss = 0.0
        print(f"  ✗ CRASH (exit={exit_code}, loss={val_loss})")
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=WORKDIR, capture_output=True)
        return False

    # Log to TSV
    mem_gb = 0.0
    with open(RESULTS_FILE, 'a') as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{mem_gb:.1f}\t{status}\t{param}={value}\n")

    return status == "keep"

# Main loop
print(f"\n[loop] Running {NUM_EXPERIMENTS} experiments × 30 min each (~20 days GPU time)")
print(f"[loop] Results logged to {RESULTS_FILE}")
print(f"[loop] GPU: CUDA_VISIBLE_DEVICES=0\n")

start_exp = 19  # Continue from exp19 if previous run reached it
improved_count = 0
crash_count = 0

for exp_num in range(start_exp, start_exp + NUM_EXPERIMENTS):
    try:
        improved = run_experiment(exp_num)
        if improved:
            improved_count += 1
    except KeyboardInterrupt:
        print("\n[loop] Interrupted by user")
        break
    except Exception as e:
        print(f"[loop] Error: {e}")
        crash_count += 1

    # Progress update every 50 experiments
    if (exp_num - start_exp + 1) % 50 == 0:
        completed = exp_num - start_exp + 1
        remaining = NUM_EXPERIMENTS - completed
        eta_hours = remaining * 0.5
        print(f"\n[loop] Progress: {completed}/{NUM_EXPERIMENTS}")
        print(f"[loop] Improvements: {improved_count}, Crashes: {crash_count}")
        print(f"[loop] Best: {best_commit} (loss={best_loss:.6f})")
        print(f"[loop] ETA: ~{eta_hours:.0f} hours\n")

print(f"\n[loop] DONE")
print(f"[loop] Total: {NUM_EXPERIMENTS} experiments")
print(f"[loop] Improvements: {improved_count}")
print(f"[loop] Best: {best_commit} (loss={best_loss:.6f})")
