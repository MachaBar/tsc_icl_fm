#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100-bis
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --job-name=tsc_onefamily_multiseed
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail

BASE=/home/d32485/synthetic-ts-classif/out/tasks

FAMILY=ou                          # <-- la SEULE famille sur laquelle on entraîne/évalue -- changez juste ça
SEEDS=(0 1 2 3 4 5 6 7 8 9 10)        # seed0 = corpus existant, 1-10 = nouveaux seeds générés

# ---- construit CORPUS_DIRS = 1 famille x 11 seeds = 11 dossiers ----
CORPUS_DIRS=()
for seed in "${SEEDS[@]}"; do
    CORPUS_DIRS+=("${BASE}/mlp_${FAMILY}_n12_c4_seed${seed}_len512")
done

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

# vérifie que tous les dossiers existent AVANT de lancer
missing=0
for d in "${CORPUS_DIRS[@]}"; do
    if [ ! -d "$d" ]; then
        echo "[MANQUANT] $d" >&2
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo "$missing dossier(s) manquant(s) sur ${#CORPUS_DIRS[@]} pour la famille '${FAMILY}' -- avez-vous généré les seeds 1-10 pour cette famille ?" >&2
    exit 1
fi

# OUT_DIR daté -- horodatage + famille + pooling + job SLURM
RUN_ID="$(date +%Y%m%d_%H%M%S)_${POOLING}_${FAMILY}_${#CORPUS_DIRS[@]}seeds_job${SLURM_JOB_ID:-manual}"
OUT_DIR="runs/${RUN_ID}/"
mkdir -p "$OUT_DIR"

echo "[$(date)] training start -- run=${RUN_ID}, famille=${FAMILY}, pooling=${POOLING}"
echo "[$(date)] ${#CORPUS_DIRS[@]} seeds combinés pour la famille '${FAMILY}' :"
printf '    %s\n' "${CORPUS_DIRS[@]}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this job from the repo root: sbatch train_classification_onefamily_multiseed.sbatch}"

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
