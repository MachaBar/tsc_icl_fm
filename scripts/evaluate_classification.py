"""
Évalue un checkpoint entraîné (ckpt/best.pt, voir train_classification.py)
sur un pool d'épisodes -- accuracy (moyenne ± écart-type sur --n-repeats
tirages), matrice de confusion, comparaison au baseline k-NN (voir
corpus_io.knn_baseline_accuracy). Ne fait aucun backward, juste de l'inférence.

Répéter l'évaluation plusieurs fois est nécessaire car `split_context_query`
retire aléatoirement le split contexte/requête à CHAQUE appel -- un seul
passage donne un seul tirage, pas une estimation stable.

Usage :
    python -m scripts.evaluate_classification \
        --ckpt runs/20260825_151530_perceiver_job45812/ckpt/best.pt \
        --corpus-dir out/tasks/mlp_ou_n12_c4_seed0_len512/ \
        --split eval --n-repeats 10

Le checkpoint contient déjà tous les hyperparamètres d'archi utilisés à
l'entraînement (`ckpt["args"]`) -- pas besoin de les repasser en CLI, le
modèle est reconstruit à l'identique automatiquement.

--corpus-dir accepte plusieurs dossiers (combinés en un seul pool), comme
--train_classification.py -- utile pour évaluer sur plusieurs graines/racines
d'un coup plutôt qu'une par une.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # headless (noeud de calcul SLURM sans serveur X)
import matplotlib.pyplot as plt
import torch

from scripts.train_classification import build_model, batches
from scripts.corpus_io import knn_baseline_accuracy, load_corpus_shards_multi, make_batch


# ----------------------------------------------------------------------
# Inférence
# ----------------------------------------------------------------------

def run_inference(model, episodes, train_frac, length, batch_size, device):
    """Une passe complète sur `episodes` (no grad) -- prédictions + labels
    vrais des séries requête, concaténés sur tout le pool."""
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch_episodes in batches(episodes, batch_size, shuffle=False):
            series, coords, y_train, y_query = make_batch(batch_episodes, train_frac, length)
            series, coords   = series.to(device), coords.to(device)
            y_train, y_query = y_train.to(device), y_query.to(device)

            logits, _ = model(series=series, coords=coords, y_train=y_train)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.reshape(-1).cpu())
            all_labels.append(y_query.reshape(-1).cpu())

    return torch.cat(all_preds), torch.cat(all_labels)


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for p, l in zip(preds.tolist(), labels.tolist()):
        cm[l, p] += 1
    return cm


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------

def plot_confusion(cm: torch.Tensor, out_path: Path) -> None:
    n = cm.shape[0]
    cm_norm = cm.float() / cm.sum(dim=1, keepdim=True).clamp(min=1)  # normalisé par ligne (= recall par classe vraie)

    fig, ax = plt.subplots(figsize=(4.5 + 0.4 * n, 4 + 0.4 * n))
    im = ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("classe vraie")
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}\n(n={cm[i, j].item()})",
                ha="center", va="center", fontsize=8,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="recall (normalisé par ligne)")
    ax.set_title("Matrice de confusion (séries requête)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="chemin vers ckpt/best.pt")
    ap.add_argument("--corpus-dir", type=Path, nargs="+", required=True, help="un ou plusieurs dossiers shard_*.parquet + metadata.json (peuvent être différents du corpus d'entraînement -- ex. un pool de test jamais vu ; plusieurs = pool combiné, mêmes n_classes/length requis)")
    ap.add_argument("--split", choices=["train", "eval"], default="eval", help="quel pool d'épisodes du corpus évaluer")
    ap.add_argument("--n-repeats", type=int, default=10, help="répète l'évaluation N fois (split contexte/requête re-tiré à chaque fois) pour une estimation stable")
    ap.add_argument("--batch-size", type=int, default=None, help="défaut : celui utilisé à l'entraînement (dans le ckpt)")
    ap.add_argument("--out", type=Path, default=None, help="défaut : <dossier du run>/eval/")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Eval] using {}".format(device))

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = SimpleNamespace(**ckpt["args"])  # ré-hydrate les hyperparamètres d'archi du training
    n_classes = ckpt["n_classes"]

    model = build_model(train_args, n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    print(
        "[Eval] checkpoint chargé -- step={}, val_acc à l'entraînement={:.3f}".format(
            ckpt.get("step"), ckpt.get("val_acc", float("nan"))
        )
    )

    data = load_corpus_shards_multi(args.corpus_dir)
    episodes = data["train_episodes"] if args.split == "train" else data["eval_episodes"]
    length = data["length"]
    batch_size = args.batch_size or train_args.batch_size
    print("[Eval] {} corpus combiné(s), {} épisodes dans le pool '{}'".format(data["n_corpora"], len(episodes), args.split))

    out_dir = args.out or (args.ckpt.parent.parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    accs = []
    all_preds, all_labels = [], []
    for _ in range(args.n_repeats):
        preds, labels = run_inference(model, episodes, train_args.train_frac, length, batch_size, device)
        accs.append((preds == labels).float().mean().item())
        all_preds.append(preds)
        all_labels.append(labels)

    accs_t = torch.tensor(accs)
    print(
        "[Eval] accuracy sur '{}' ({} répétitions) : {:.4f} ± {:.4f} (hasard = {:.3f})".format(
            args.split, args.n_repeats, accs_t.mean().item(), accs_t.std().item(), 1 / n_classes
        )
    )

    baseline_acc = knn_baseline_accuracy(episodes, train_args.train_frac, k=1)
    print("[Eval] baseline 1-NN (sans modèle) sur les mêmes épisodes : {:.4f}".format(baseline_acc))

    cm = confusion_matrix(torch.cat(all_preds), torch.cat(all_labels), n_classes)
    cm_path = out_dir / "confusion_{}.png".format(args.split)
    plot_confusion(cm, cm_path)
    print("[Eval] matrice de confusion -> {}".format(cm_path))


if __name__ == "__main__":
    main()
