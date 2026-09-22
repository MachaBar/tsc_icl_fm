#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=a100
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --job-name=tsc_ood_mech_k
#SBATCH -o ./jobs/%A_%a.out
#SBATCH -e ./jobs/%A_%a.err
#SBATCH --array=0-15%4
# array size = len(HELD_OUT_MECHANISMS) * len(K_VALUES) - 1 -- recalculer si vous changez les listes
# ci-dessous (4 held-out x 4 k = 16 tasks -> 0-15). %4 = au plus 4 jobs GPU en même temps.

set -euo pipefail

BASE=/home/d32485/synthetic-ts-classif/out/tasks
# root_family fixée à "ar" partout -- seul le mécanisme varie (isole l'axe mécanisme, cf. discussion précédente)
mechanism_path() { echo "${BASE}/${1}_ar_n12_c4_seed0_len512"; }

# ---- les 8 mécanismes disponibles pour root_family=ar -- l'ordre sert aussi d'ordre d'ajout progressif ----
ALL_MECHANISMS=(convolutional linear local mlp polynomial radial rnn threshold)

# ---- mécanismes utilisés comme cible OOD, en rotation -- éditer selon le budget compute ----
# mlp : le mécanisme "de référence" utilisé partout ailleurs dans le projet
# linear : le plus simple (cas dégénéré)
# rnn : dépendance séquentielle/récurrente, structurellement différente d'un MLP
# radial : base non-linéaire différente (RBF-like)
HELD_OUT_MECHANISMS=(mlp linear rnn radial)

# ---- nombre de mécanismes d'entraînement testés, progressif (max = 7 = tous sauf la held-out) ----
K_VALUES=(1 2 4 7)

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
N_REPEATS_EVAL=10
# ----------------------------------------------------------------------------------------------

# ---- décoder SLURM_ARRAY_TASK_ID -> (held_out, k) ----
N_K=${#K_VALUES[@]}
HELD_OUT_IDX=$(( SLURM_ARRAY_TASK_ID / N_K ))
K_IDX=$(( SLURM_ARRAY_TASK_ID % N_K ))
HELD_OUT=${HELD_OUT_MECHANISMS[$HELD_OUT_IDX]}
K=${K_VALUES[$K_IDX]}

# ---- construire les k mécanismes d'entraînement : tous sauf HELD_OUT, tronqués à K ----
TRAIN_MECHANISMS=()
for m in "${ALL_MECHANISMS[@]}"; do
    if [ "$m" != "$HELD_OUT" ]; then
        TRAIN_MECHANISMS+=("$m")
    fi
done
if [ "$K" -gt "${#TRAIN_MECHANISMS[@]}" ]; then
    echo "[$(date)] K=${K} > ${#TRAIN_MECHANISMS[@]} mécanismes disponibles (8 - 1 held-out) -- abandon de cette task" >&2
    exit 0
fi
TRAIN_MECHANISMS=("${TRAIN_MECHANISMS[@]:0:$K}")

CORPUS_DIRS=()
for m in "${TRAIN_MECHANISMS[@]}"; do
    CORPUS_DIRS+=("$(mechanism_path "$m")")
done
HELD_OUT_DIR="$(mechanism_path "$HELD_OUT")"

mkdir -p logs jobs

# NB : préfixe "ood_mech_holdout" (pas "ood_holdout") pour ne pas mélanger avec l'expérience
# "famille exclue" dans les agrégations -- voir aggregate_ood_mechanism_results.py
RUN_ID="$(date +%Y%m%d_%H%M%S)_ood_mech_holdout-${HELD_OUT}_k${K}_job${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="runs/${RUN_ID}/"
mkdir -p "$OUT_DIR"

echo "[$(date)] OOD mécanisme progressif -- held_out=${HELD_OUT}, k=${K} (root_family=ar fixée)"
echo "[$(date)] mécanismes d'entraînement (${#CORPUS_DIRS[@]}) :"
printf '    %s\n' "${TRAIN_MECHANISMS[@]}"
echo "[$(date)] mécanisme OOD (jamais entraîné dessus) : ${HELD_OUT}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this job from the repo root: sbatch ood_mechanism_progressive.sbatch}"

EXTRA_ARGS=()
if [ "$INCLUDE_LABEL_TOKEN" = "true" ]; then
    EXTRA_ARGS+=(--include-label-token)
fi

# ---- 1. entraînement sur les k mécanismes (root_family=ar partout) ----
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

# ---- 2. éval OOD : mécanisme jamais vu à l'entraînement, même root_family=ar (le vrai test) ----
echo "[$(date)] éval OOD sur mechanism='${HELD_OUT}'"
uv run python -u -m scripts.evaluate_classification \
    --ckpt "${OUT_DIR}/ckpt/best.pt" \
    --corpus-dir "$HELD_OUT_DIR" \
    --split eval \
    --n-repeats "$N_REPEATS_EVAL" \
    --out "${OUT_DIR}/eval_ood_${HELD_OUT}"

# ---- 3. éval in-distribution : pool eval des mécanismes d'entraînement (référence) ----
echo "[$(date)] éval in-distribution sur les ${K} mécanismes d'entraînement"
uv run python -u -m scripts.evaluate_classification \
    --ckpt "${OUT_DIR}/ckpt/best.pt" \
    --corpus-dir "${CORPUS_DIRS[@]}" \
    --split eval \
    --n-repeats "$N_REPEATS_EVAL" \
    --out "${OUT_DIR}/eval_indist"

echo "[$(date)] OOD mechanism (progressif) experiment done -- held_out=${HELD_OUT}, k=${K}, run=${RUN_ID}"
