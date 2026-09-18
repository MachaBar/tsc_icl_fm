#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=a100
#SBATCH --time=0-00:30:00            # éval = inférence seule, bien plus léger qu'un entraînement -- 8h était surdimensionné
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                 # corrigé : gpu:0 sur une partition GPU n'a pas de sens (voir commentaire ci-dessus)
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --job-name=tsc_eval
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail

# ---- éditer pour votre run -----------------------------------------------------------------
# doit pointer vers le dossier daté créé par le dernier train_classification.sbatch
# (runs/{timestamp}_{pooling}_{n}roots_job{id}/ckpt/best.pt) -- pas de nom fixe possible
# puisque OUT_DIR est maintenant horodaté.
# CKPT=/home/d32485/tsc_icl_fm/runs/20260915_093413_perceiver_5roots_job48572/ckpt/best.pt
# CKPT=/home/d32485/tsc_icl_fm/runs/20260915_224540_perceiver_1roots_job48663_trend_mlp/ckpt/best.pt

CKPT=/home/d32485/tsc_icl_fm/runs/20260916_132536_perceiver_17roots_job48861_all/ckpt/best.pt

# CORPUS_DIRS=(
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_trend_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_ou_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_motif_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_ecg_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_bump_n12_c4_seed0_len512
# )

# CORPUS_DIRS=(
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_ou_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_motif_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_ecg_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_bump_n12_c4_seed0_len512
# )

CORPUS_DIRS=(
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_ar_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_bump_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_burst_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_count_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_ecg_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_ets_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_gam_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_garch_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_gp_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_motif_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_ou_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_sensor_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_spike_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_step_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_transient_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_trend_n12_c4_seed0_len512
    /home/d32485/synthetic-ts-classif/out/tasks/mlp_tsi_n12_c4_seed0_len512
)

SPLIT=eval
N_REPEATS=10
# ----------------------------------------------------------------------------------------------

mkdir -p logs

echo "[$(date)] eval start -- ckpt=${CKPT}"
echo "[$(date)] corpus combinés (${#CORPUS_DIRS[@]}) :"
printf '    %s\n' "${CORPUS_DIRS[@]}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this job from the repo root: sbatch evaluate_classification.sbatch}"

uv run python -u -m scripts.evaluate_classification \
    --ckpt "$CKPT" \
    --corpus-dir "${CORPUS_DIRS[@]}" \
    --split "$SPLIT" \
    --n-repeats "$N_REPEATS"

echo "[$(date)] eval done"
