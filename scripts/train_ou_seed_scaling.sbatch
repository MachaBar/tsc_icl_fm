#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --job-name=tsc_ou_seedscale
#SBATCH -o ./jobs/%A_%a.out
#SBATCH -e ./jobs/%A_%a.err
#SBATCH --array=0-2
# array size = nombre de tailles de pool d'entraînement testées (11/25/50) - 1

set -euo pipefail

BASE=/home/d32485/synthetic-ts-classif/out/tasks
FAMILY=ou

# les 3 tailles de pool à comparer -- toutes des SOUS-ENSEMBLES de 0..49,
# donc les seeds 0-10 sont dans les 3 runs (comparaison cumulative)
N_TRAIN_SEEDS_LIST=(11 25 50)
N_TRAIN_SEEDS=${N_TRAIN_SEEDS_LIST[$SLURM_ARRAY_TASK_ID]}

# fixe, IDENTIQUE pour les 3 runs -- jamais entraîné, seulement monitoré en lecture seule
MONITOR_SEEDS=(80 81 82 83 84)

# construit CORPUS_DIRS = seeds 0..N_TRAIN_SEEDS-1
CORPUS_DIRS=()
for ((seed=0; seed<N_TRAIN_SEEDS; seed++)); do
    CORPUS_DIRS+=("${BASE}/mlp_${FAMILY}_n12_c4_seed${seed}_len512")
done

OOD_CORPUS_DIRS=()
for seed in "${MONITOR_SEEDS[@]}"; do
    OOD_CORPUS_DIRS+=("${BASE}/mlp_${FAMILY}_n12_c4_seed${seed}_len512")
done

POOLING=perceiver
INCLUDE_LABEL_TOKEN=true

BATCH_SIZE=8
TRAIN_FRAC=0.6
MAX_STEPS=50000
EVAL_EVERY=100
PATIENCE=30
LR=3e-4
WEIGHT_DECAY=0.0
GRAD_CLIP_VALUE=1.0
SCHEDULER=none

HIDDEN_DIM=64
LATENT_DIM=16
NUM_LATENTS=16
ENCODER_DEPTH=2

ICL_D_MODEL=64
ICL_BLOCKS=2
ICL_NHEAD=4
ICL_DIM_FEEDFORWARD=128

SEED=0
# ----------------------------------------------------------------------------------------------

mkdir -p logs jobs

missing=0
for d in "${CORPUS_DIRS[@]}" "${OOD_CORPUS_DIRS[@]}"; do
    if [ ! -d "$d" ]; then
        echo "[MANQUANT] $d" >&2
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo "$missing dossier(s) manquant(s) -- avez-vous lancé generate_ou_seed_sweep.py ?" >&2
    exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_${POOLING}_${FAMILY}_${N_TRAIN_SEEDS}trainseeds_job${SLURM_JOB_ID:-manual}"
OUT_DIR="runs/${RUN_ID}/"
mkdir -p "$OUT_DIR"

echo "[$(date)] training start -- run=${RUN_ID}, famille=${FAMILY}, ${N_TRAIN_SEEDS} train seeds, monitor=${MONITOR_SEEDS[*]}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit from repo root}"

EXTRA_ARGS=()
if [ "$INCLUDE_LABEL_TOKEN" = "true" ]; then
    EXTRA_ARGS+=(--include-label-token)
fi

uv run python -u -m scripts.train_classification \
    --corpus-dir "${CORPUS_DIRS[@]}" \
    --ood-corpus-dir "${OOD_CORPUS_DIRS[@]}" \
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
echo "[$(date)] pour l'éval finale (holdout seeds 90-99, jamais touchées ici), utilisez eval_ou_holdout.py sur ${OUT_DIR}ckpt/best.pt"
