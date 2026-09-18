#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=a100      
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --job-name=tsc
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail

# uv run python -u -m scripts.evaluate_classification \
#     --ckpt /home/d32485/tsc_icl_fm/runs/first_classif_run/ckpt/best.pt \
#     --corpus-dir /home/d32485/synthetic-ts-classif/out/tasks/mlp_ou_n12_c4_seed0_len512/ \
#     --split eval --n-repeats 10

uv run python -u -m scripts.visualize_corpus \
        --corpus-dir /home/d32485/tsc_icl_fm/runs/20260915_093413_perceiver_5roots_job48572/ckpt/best.pt \
        --split eval --indices 10,20