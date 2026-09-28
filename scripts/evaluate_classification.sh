#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=a100      
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

# uv run python -u -m scripts.evaluate_classification \
#     --ckpt /home/d32485/tsc_icl_fm/runs/first_classif_run/ckpt/best.pt \
#     --corpus-dir /home/d32485/synthetic-ts-classif/out/tasks/mlp_ou_n12_c4_seed0_len512/ \
#     --split eval --n-repeats 10

# uv run python -u -m scripts.evaluate_classification \
#     --ckpt /home/d32485/tsc_icl_fm/runs/20260923_225613_perceiver_ou_11seeds_job50689/ckpt/best.pt \
#     --corpus-dir /home/d32485/synthetic-ts-classif/out/tasks/mlp_ou_n12_c4_seed12_len512 \
#     --split eval --n-repeats 10


# uv run python -u -m scripts.visualize_corpus \
#         --corpus-dir /home/d32485/tsc_icl_fm/runs/20260915_093413_perceiver_5roots_job48572/ckpt/best.pt \
#         --split eval --indices 10,20

# uv run python -u -m scripts.aggregate_ood_results --runs-dir runs/ --out /home/d32485/tsc_icl_fm/runs/ood_curves.png

# uv run python -u -m scripts.aggregate_ood_mechanism_results --runs-dir runs/ --out /home/d32485/tsc_icl_fm/runs/ood_mechanism_curves.png


# BASE=/home/d32485/synthetic-ts-classif/out/tasks
# HOLDOUT_DIRS=()
# for s in 90 91 92 93 94 95 96 97 98 99; do
#     HOLDOUT_DIRS+=("${BASE}/mlp_ou_n12_c4_seed${s}_len512")
# done

# uv run python -m scripts.eval_ou_holdout \
#     --ckpt /home/d32485/tsc_icl_fm/runs/20260924_134944_perceiver_ou_50trainseeds_job50867/ckpt/best.pt \
#     --holdout-corpus-dir "${HOLDOUT_DIRS[@]}" \
#     --batch-size 8


# BASE=/home/d32485/synthetic-ts-classif/out/tasks
# # FAMILIES=(ar bump burst count ecg ets gam garch gp motif sensor spike step transient trend tsi)  # sans ou = famille d'entraînement
# FAMILIES=(ar bump burst count ecg ets gam garch gp motif sensor spike step transient trend tsi)  # sans ou = famille d'entraînement

# CORPUS_DIRS=()
# for f in "${FAMILIES[@]}"; do
#     CORPUS_DIRS+=("${BASE}/mlp_${f}_n12_c4_seed0_len512")
# done

# uv run python -m scripts.eval_cross_family \
#     --ckpt runs/<RUN_ID>/ckpt/best.pt \
#     --corpus-dir "${CORPUS_DIRS[@]}" \
#     --batch-size 8


#!/bin/bash
# Éval OOD cross-famille : modèle entraîné sur 'ou' (seeds 0-49), évalué
# séparément sur les 16 autres familles, sur des seeds JAMAIS vues à
# l'entraînement (90-98, hors pool 0-49) -- donc un vrai test de
# généralisation structurelle (topologie + poids MLP différents), pas juste
# un changement de signal d'entrée sur un circuit déjà vu (ce qu'aurait donné
# seed0, partagée avec le pool d'entraînement).
set -euo pipefail
# ---- à éditer avant chaque lancement ----
CKPT=/home/d32485/tsc_icl_fm/runs/20260924_134944_perceiver_ou_50trainseeds_job50867/ckpt/best.pt
# ------------------------------------------
BASE=/home/d32485/synthetic-ts-classif/out/tasks
FAMILIES=(ar bump burst count ecg ets gam garch gp )  # sans ou = famille d'entraînement
OOD_SEEDS=(90 91 92 93 94 95 96 97 98)  # hors pool d'entraînement (0-49)
mkdir -p jobs
CORPUS_DIRS=()
for f in "${FAMILIES[@]}"; do
    for s in "${OOD_SEEDS[@]}"; do
        CORPUS_DIRS+=("${BASE}/mlp_${f}_n12_c4_seed${s}_len512")
    done
done
echo "[$(date)] éval cross-famille OOD -- ckpt=${CKPT}"
echo "[$(date)] ${#FAMILIES[@]} familles x ${#OOD_SEEDS[@]} seeds = ${#CORPUS_DIRS[@]} dossiers"
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit from tsc_icl_fm repo root: sbatch eval_cross_family_ood.sbatch}"
uv run python -m scripts.eval_cross_family \
    --ckpt "$CKPT" \
    --corpus-dir "${CORPUS_DIRS[@]}" \
    --batch-size 8
echo "[$(date)] terminé"