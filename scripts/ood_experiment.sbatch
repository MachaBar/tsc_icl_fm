#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=a100
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --job-name=tsc_ood
#SBATCH -o ./jobs/%A_%a.out
#SBATCH -e ./jobs/%A_%a.err
#SBATCH --array=0-24%4
# array size = len(HELD_OUT_FAMILIES) * len(K_VALUES) - 1 -- recalculer si vous changez les listes
# ci-dessous (5 held-out x 5 k = 25 tasks -> 0-24). %4 = au plus 4 jobs GPU en même temps.

set -euo pipefail

BASE=/home/d32485/synthetic-ts-classif/out/tasks
family_path() { echo "${BASE}/mlp_${1}_n12_c4_seed0_len512"; }

# ---- toutes les familles disponibles -- l'ordre sert aussi d'ordre d'ajout progressif ----
ALL_FAMILIES=(ar bump burst count ecg ets gam garch gp motif ou sensor spike step transient trend tsi)

# ---- familles utilisées comme cible OOD, en rotation -- éditer selon le budget compute ----
HELD_OUT_FAMILIES=(trend ou motif ecg bump)

# ---- nombre de familles d'entraînement testées, progressif (max = 16 = toutes sauf la held-out) ----
K_VALUES=(1 2 4 8 16)

POOLING=perceiver                   # mean / max / sum / attention / cls / perceiver
INCLUDE_LABEL_TOKEN=true            # incompatible avec POOLING=cls

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
N_REPEATS_EVAL=10
# ----------------------------------------------------------------------------------------------

# ---- décoder SLURM_ARRAY_TASK_ID -> (held_out, k) ----
N_K=${#K_VALUES[@]}
HELD_OUT_IDX=$(( SLURM_ARRAY_TASK_ID / N_K ))
K_IDX=$(( SLURM_ARRAY_TASK_ID % N_K ))
HELD_OUT=${HELD_OUT_FAMILIES[$HELD_OUT_IDX]}
K=${K_VALUES[$K_IDX]}

# ---- construire les k familles d'entraînement : toutes sauf HELD_OUT, tronquées à K ----
TRAIN_FAMILIES=()
for f in "${ALL_FAMILIES[@]}"; do
    if [ "$f" != "$HELD_OUT" ]; then
        TRAIN_FAMILIES+=("$f")
    fi
done
if [ "$K" -gt "${#TRAIN_FAMILIES[@]}" ]; then
    echo "[$(date)] K=${K} > ${#TRAIN_FAMILIES[@]} familles disponibles (17 - 1 held-out) -- abandon de cette task" >&2
    exit 0
fi
TRAIN_FAMILIES=("${TRAIN_FAMILIES[@]:0:$K}")

CORPUS_DIRS=()
for f in "${TRAIN_FAMILIES[@]}"; do
    CORPUS_DIRS+=("$(family_path "$f")")
done
HELD_OUT_DIR="$(family_path "$HELD_OUT")"

mkdir -p logs

RUN_ID="$(date +%Y%m%d_%H%M%S)_ood_holdout-${HELD_OUT}_k${K}_job${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="runs/${RUN_ID}/"
mkdir -p "$OUT_DIR"

echo "[$(date)] OOD run start -- held_out=${HELD_OUT}, k=${K}"
echo "[$(date)] familles d'entraînement (${#CORPUS_DIRS[@]}) :"
printf '    %s\n' "${TRAIN_FAMILIES[@]}"
echo "[$(date)] famille OOD (jamais entraînée dessus) : ${HELD_OUT}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this job from the repo root: sbatch ood_experiment.sbatch}"

EXTRA_ARGS=()
if [ "$INCLUDE_LABEL_TOKEN" = "true" ]; then
    EXTRA_ARGS+=(--include-label-token)
fi

# ---- 1. entraînement sur les k familles ----
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

# ---- 2. éval OOD : famille jamais vue à l'entraînement (le vrai test) ----
echo "[$(date)] éval OOD sur '${HELD_OUT}'"
uv run python -u -m scripts.evaluate_classification \
    --ckpt "${OUT_DIR}/ckpt/best.pt" \
    --corpus-dir "$HELD_OUT_DIR" \
    --split eval \
    --n-repeats "$N_REPEATS_EVAL" \
    --out "${OUT_DIR}/eval_ood_${HELD_OUT}"

# ---- 3. éval in-distribution : pool eval des familles d'entraînement (référence) ----
echo "[$(date)] éval in-distribution sur les ${K} familles d'entraînement"
uv run python -u -m scripts.evaluate_classification \
    --ckpt "${OUT_DIR}/ckpt/best.pt" \
    --corpus-dir "${CORPUS_DIRS[@]}" \
    --split eval \
    --n-repeats "$N_REPEATS_EVAL" \
    --out "${OUT_DIR}/eval_indist"

echo "[$(date)] OOD experiment done -- held_out=${HELD_OUT}, k=${K}, run=${RUN_ID}"
