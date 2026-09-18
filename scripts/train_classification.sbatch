#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100-bis
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --job-name=tsc
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail

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

# CORPUS_DIRS=(
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_trend_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_ou_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_motif_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_ecg_n12_c4_seed0_len512
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_bump_n12_c4_seed0_len512
# )

# CORPUS_DIRS=(
#     /home/d32485/synthetic-ts-classif/out/tasks/mlp_trend_n12_c4_seed0_len512
# )

POOLING=perceiver                   # mean / max / sum / attention / cls / perceiver
INCLUDE_LABEL_TOKEN=true            # true pour fusion du label comme un token (M+1) dans l'encodeur (incompatible avec POOLING=cls)

BATCH_SIZE=8
TRAIN_FRAC=0.6
MAX_STEPS=50000
EVAL_EVERY=100                      # cadence (steps) de validation/log/plot/checkpoint
PATIENCE=30                         # en nb d'évaluations (donc PATIENCE * EVAL_EVERY steps)
LR=3e-4
WEIGHT_DECAY=0.0
GRAD_CLIP_VALUE=1.0
SCHEDULER=none                      # none / cosine

HIDDEN_DIM=64
LATENT_DIM=16
NUM_LATENTS=16
ENCODER_DEPTH=2

ICL_D_MODEL=64
ICL_BLOCKS=2
ICL_NHEAD=4
ICL_DIM_FEEDFORWARD=128

SEED=0                               # graine PyTorch de l'entraînement (mélange des batches) -- sans rapport avec SEEDS ci-dessus
# ----------------------------------------------------------------------------------------------

mkdir -p logs

# OUT_DIR daté -- horodatage + pooling + job SLURM, pour que deux runs ne s'écrasent jamais
RUN_ID="$(date +%Y%m%d_%H%M%S)_${POOLING}_${#CORPUS_DIRS[@]}roots_job${SLURM_JOB_ID:-manual}_all"
OUT_DIR="runs/${RUN_ID}/"
mkdir -p "$OUT_DIR"

echo "[$(date)] training start -- run=${RUN_ID}, pooling=${POOLING}"
echo "[$(date)] corpus combinés (${#CORPUS_DIRS[@]}) :"
printf '    %s\n' "${CORPUS_DIRS[@]}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this job from the repo root: sbatch train_classification.sbatch}"

EXTRA_ARGS=()
if [ "$INCLUDE_LABEL_TOKEN" = "true" ]; then
    EXTRA_ARGS+=(--include-label-token)
fi

uv run python -u -m scripts.train_classification \
    --corpus-dir "${CORPUS_DIRS[@]}" \
    --out "$OUT_DIR" \
    --pooling "$POOLING" \
    --batch-size "$BATCH_SIZE" \
    --train-frac "$TRAIN_FRAC" \
    --max-steps "$MAX_STEPS" \
    --eval-every "$EVAL_EVERY" \
    --patience "$PATIENCE" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --grad-clip-value "$GRAD_CLIP_VALUE" \
    --scheduler "$SCHEDULER" \
    --hidden-dim "$HIDDEN_DIM" \
    --latent-dim "$LATENT_DIM" \
    --num-latents "$NUM_LATENTS" \
    --encoder-depth "$ENCODER_DEPTH" \
    --icl-d-model "$ICL_D_MODEL" \
    --icl-blocks "$ICL_BLOCKS" \
    --icl-nhead "$ICL_NHEAD" \
    --icl-dim-feedforward "$ICL_DIM_FEEDFORWARD" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}"

echo "[$(date)] training done -- run=${RUN_ID}"
