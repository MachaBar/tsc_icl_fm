"""
Lecture du corpus synthétique (format parquet, voir generate_synthetic_corpus.py
côté projet génération). 

`Episode` reproduit juste l'interface utilisée en aval (`.values`,
`.raw_values`, `.labels`, `.n_classes`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt


@dataclass
class Episode:
    values: torch.Tensor       # (n_series, T) -- normalisé par ligne
    raw_values: torch.Tensor   # (n_series, T) -- échelle naturelle
    labels: torch.Tensor       # (n_series,)
    n_classes: int


# ----------------------------------------------------------------------
# Rechargement
# ----------------------------------------------------------------------

def load_corpus_shards_parquet(out_dir: Path) -> dict:
    """Recombine tous les `shard_*.parquet` + `metadata.json` d'un dossier
    (voir `save_shard_parquet`/`save_metadata_json` côté génération) en un
    seul corpus, au même format que `load_corpus_shards` (.pt) : un dict
    avec `train_episodes`/`eval_episodes` (listes d'`Episode`) + métadonnées."""

    shard_paths = sorted(out_dir.glob("shard_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"aucun fichier shard_*.parquet dans {out_dir}")

    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json manquant dans {out_dir}")
    meta = json.loads(meta_path.read_text())

    df = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)

    train_episodes: list[Episode] = []
    eval_episodes: list[Episode] = []

    for (_episode_id, split), group in df.groupby(["episode_id", "split"], sort=True):
        group = group.sort_values("series_idx")
        episode = Episode(
            values     = torch.tensor(np.stack(group["values"].to_numpy()), dtype=torch.float32),
            raw_values = torch.tensor(np.stack(group["raw_values"].to_numpy()), dtype=torch.float32),
            labels     = torch.tensor(group["label"].to_numpy(), dtype=torch.long),
            n_classes  = meta["n_classes"],
        )
        (train_episodes if split == "train" else eval_episodes).append(episode)

    return {
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "n_classes": meta["n_classes"],
        "mechanism": meta["mechanism"],
        "dag_mode": meta["dag_mode"],
        "root_families": meta["root_families"],
        "length": meta["length"],
    }


def load_corpus_shards_multi(corpus_dirs: list[Path]) -> dict:
    """Combine PLUSIEURS corpus (ex: plusieurs graines/initialisations de
    racine de la même famille structurelle -- même mechanism/n_nodes/n_classes,
    seed différente) en un seul pool train/eval, pour entraîner sur plus de
    variété tout en restant in-distribution (même famille, tirages différents).

    `n_classes` et `length` DOIVENT être identiques entre tous les corpus
    combinés -- contrainte architecturale dure : le nombre de classes est figé
    dans `ICLearningClassification`, et `make_batch` empile les séries en
    supposant une longueur commune."""
    if len(corpus_dirs) == 1:
        data = load_corpus_shards_parquet(corpus_dirs[0])
        data["mechanism"] = [data["mechanism"]]
        data["dag_mode"] = [data["dag_mode"]]
        data["root_families"] = [data["root_families"]]
        data["n_corpora"] = 1
        return data

    datasets = [load_corpus_shards_parquet(d) for d in corpus_dirs]

    n_classes_set = {d["n_classes"] for d in datasets}
    if len(n_classes_set) > 1:
        raise ValueError(
            f"n_classes incohérent entre corpus combinés : {n_classes_set} "
            f"(dossiers : {corpus_dirs}) -- tous doivent avoir le même nombre de classes"
        )
    length_set = {d["length"] for d in datasets}
    if len(length_set) > 1:
        raise ValueError(
            f"length incohérent entre corpus combinés : {length_set} "
            f"(dossiers : {corpus_dirs}) -- tous doivent avoir la même longueur de série"
        )

    train_episodes: list[Episode] = []
    eval_episodes: list[Episode] = []
    for d in datasets:
        train_episodes.extend(d["train_episodes"])
        eval_episodes.extend(d["eval_episodes"])

    return {
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "n_classes": n_classes_set.pop(),
        "length": length_set.pop(),
        "mechanism": [d["mechanism"] for d in datasets],
        "dag_mode": [d["dag_mode"] for d in datasets],
        "root_families": [d["root_families"] for d in datasets],
        "n_corpora": len(datasets),
    }


# ----------------------------------------------------------------------
# Batching -- identique à generate_synthetic_corpus.py (ne dépend que de
# l'interface .values/.labels/.n_classes, jamais de synthgen)
# ----------------------------------------------------------------------

def split_context_query(
    episode: Episode,
    train_frac: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Découpe un épisode en contexte (train) / requête (query), STRATIFIÉ par
    classe -- un slice naïf laisserait potentiellement une classe entière
    d'un seul côté."""
    n_classes = episode.n_classes
    n_per_class = episode.values.shape[0] // n_classes
    n_train_pc = max(1, round(train_frac * n_per_class))

    train_idx, query_idx = [], []
    for c in range(n_classes):
        idx = (episode.labels == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx))]
        train_idx.append(idx[:n_train_pc])
        query_idx.append(idx[n_train_pc:])

    train_idx = torch.cat(train_idx)
    train_idx = train_idx[torch.randperm(len(train_idx))]  # mélange les classes entre elles
    query_idx = torch.cat(query_idx)
    query_idx = query_idx[torch.randperm(len(query_idx))]

    order = torch.cat([train_idx, query_idx])
    return episode.values[order], episode.labels[order], len(train_idx)


def make_batch(
    episodes: list[Episode],
    train_frac: float,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble une liste d'épisodes en un batch (bs, N, T, 1) prêt pour
    AROMAEncoderICLClassifier.forward(series, coords, y_train). Retourne
    aussi y_query (labels des séries requête, pour la loss -- jamais passé
    au modèle)."""
    values_list, labels_list, train_sizes = [], [], []
    for episode in episodes:
        v, l, ts = split_context_query(episode, train_frac)
        values_list.append(v)
        labels_list.append(l)
        train_sizes.append(ts)

    assert len(set(train_sizes)) == 1, "train_size doit être identique dans tout le batch"
    train_size = train_sizes[0]

    series = torch.stack(values_list, dim=0).unsqueeze(-1)  # (bs, N, T, 1)
    coords = (
        torch.linspace(0.0, 1.0, length)
        .view(1, 1, length, 1)
        .expand_as(series)
        .contiguous()
    )
    y_all = torch.stack(labels_list, dim=0)  # (bs, N)

    return series, coords, y_all[:, :train_size], y_all[:, train_size:]


def knn_baseline_accuracy(episodes: list[Episode], train_frac: float, k: int = 1) -> float:
    """1-NN (par défaut) sur les séries brutes normalisées, distance
    euclidienne point à point, MÊME split contexte/requête que le modèle
    (`split_context_query`) -- aucun apprentissage, juste "la requête
    ressemble-t-elle à un contexte de la même classe ?". Si ce baseline
    trivial approche l'accuracy de AROMAEncoderICLClassifier, la tâche ne
    teste pas grand-chose au-delà d'une similarité de forme brute -- pas
    besoin de l'encodeur/pooling/tête ICL pour la résoudre à ce niveau."""
    correct, total = 0, 0
    for episode in episodes:
        values, labels, train_size = split_context_query(episode, train_frac)
        ctx_values,   ctx_labels   = values[:train_size], labels[:train_size]
        query_values, query_labels = values[train_size:], labels[train_size:]

        dists = torch.cdist(query_values, ctx_values)          # (n_query, n_context)
        knn_idx = dists.topk(k, largest=False).indices          # (n_query, k)
        knn_labels = ctx_labels[knn_idx]                        # (n_query, k)
        preds = knn_labels.mode(dim=1).values                   # vote majoritaire

        correct += (preds == query_labels).sum().item()
        total += len(query_labels)

    return correct / total


def plot_episode(
    episode: Episode,
    title: str,
    out_path: Path,
    max_per_class: int = 6,
) -> None:
    """Un sous-graphe par classe, quelques séries superposées par classe."""
    n_classes = episode.n_classes
    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 3.2), sharey=True)
    if n_classes == 1:
        axes = [axes]

    for c in range(n_classes):
        ax = axes[c]
        idx = (episode.labels == c).nonzero(as_tuple=True)[0][:max_per_class]
        for i in idx:
            ax.plot(episode.values[i].numpy(), color=cmap(c), alpha=0.7, linewidth=1)
        ax.set_title(f"classe {c} (n={int((episode.labels == c).sum())})")
        ax.set_xlabel("t")
        if c == 0:
            ax.set_ylabel("valeur (normalisée)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_overlay(
    episode: Episode,
    title: str,
    out_path: Path,
    max_per_class: int = 15,
) -> None:
    """Toutes les classes superposées sur un seul graphe -- pour juger la
    séparabilité visuelle entre classes d'un coup d'oeil."""
    n_classes = episode.n_classes
    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in range(n_classes):
        idx = (episode.labels == c).nonzero(as_tuple=True)[0][:max_per_class]
        for j, i in enumerate(idx):
            ax.plot(
                episode.values[i].numpy(),
                color=cmap(c),
                alpha=0.5,
                linewidth=1,
                label=f"classe {c}" if j == 0 else None,
            )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("valeur (normalisée)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_raw_vs_normalized(
    episode: Episode,
    series_idx: int,
    title: str,
    out_path: Path,
) -> None:
    """Compare values (normalisé par ligne, ce que voit le modèle) vs
    raw_values (échelle naturelle) pour UNE série."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(episode.raw_values[series_idx].numpy(), color="tab:orange")
    axes[0].set_title("raw_values (échelle naturelle, avant normalisation)")
    axes[1].plot(episode.values[series_idx].numpy(), color="tab:blue")
    axes[1].set_title("values (normalisé par ligne -- ce que reçoit l'encodeur)")
    axes[1].set_xlabel("t")
    fig.suptitle(f"{title} -- série #{series_idx}, classe {int(episode.labels[series_idx])}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)