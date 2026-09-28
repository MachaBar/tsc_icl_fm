#!/usr/bin/env python3
"""Bucketing  pour batcher des épisodes de T/n_per_class variables
depuis un corpus généré par generate_variable_corpus.py, SANS padding ni
masque d'attention -- chaque épisode retenu dans un batch est rogné
(crop, jamais complété) à la taille exacte du bucket, arrondie au multiple de
16 inférieur le plus proche (alignement mémoire/GPU classique).

Pourquoi arrondir vers le BAS uniquement : ça garantit qu'on peut toujours
rogner (jamais besoin de compléter avec du padding) -- un épisode de T=203
dans le bucket T=192 perd juste ses 11 derniers points, aucune valeur
inventée n'entre jamais dans un batch.

Le rognage par classe (`crop_episode`) prend les `n_per_class_bucket`
PREMIÈRES instances de CHAQUE classe -- jamais un crop global sur les N
lignes, qui casserait l'équilibre inter-classe (les valeurs sont empilées
bloc de classe par bloc de classe dans task.values/labels, cf. dag.py
_sample_once: `class_mode='node'`).

Une fois passé à la génération online (set_batch_params-style, cf. discussion
sur TimeIndexForecastingTrainDataset), ce module devient inutile -- gardé
séparé du reste exprès pour pouvoir le supprimer proprement plus tard.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch

from load_variable_corpus import load_manifest  # même dossier -- cf. load_variable_corpus.py


def round_down_multiple(x: int, multiple: int = 16, min_val: int = 16) -> int:
    return max(min_val, (x // multiple) * multiple)


def min_class_count(row: dict) -> int:
    """Nb de séries de la classe la MOINS représentée -- pas une moyenne
    (n_series // n_classes), qui suppose des classes équilibrées et peut
    dépasser le compte réel d'une classe minoritaire pour un épisode généré
    avec --imbalance-mode dirichlet (voir manifest['class_counts'], toujours
    présent depuis generate_variable_corpus.py -- fallback moyenne seulement
    pour un manifest antérieur à ce champ)."""
    class_counts = row.get("class_counts")
    if class_counts:
        return min(class_counts)
    return row["n_series"] // row["n_classes"]


def bucket_key(row: dict, multiple: int = 16, min_val: int = 16) -> tuple[int, int]:
    T_b = round_down_multiple(row["length"], multiple, min_val)
    npc_b = round_down_multiple(min_class_count(row), multiple, min_val)
    return (T_b, npc_b)


def build_buckets(manifest: list[dict], multiple: int = 16, min_val: int = 16) -> dict[tuple[int, int], list[dict]]:
    """Groupe le manifest par bucket (T_arrondi, n_per_class_arrondi). Un
    épisode dont sa classe la moins représentée ou sa longueur est déjà
    < min_val est écarté (aucun padding possible pour lui sans mentir sur
    les données)."""
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    dropped = 0
    for row in manifest:
        if row["length"] < min_val or min_class_count(row) < min_val:
            dropped += 1
            continue
        buckets[bucket_key(row, multiple, min_val)].append(row)
    if dropped:
        print(f"[bucketing] {dropped} épisode(s) écarté(s) (length ou classe minoritaire < {min_val})")
    return dict(buckets)


def crop_episode(task, T_b: int, npc_b: int):
    """Rogne un épisode (ClassificationTask) à (T_b, npc_b par classe),
    en gardant les npc_b PREMIÈRES instances de CHAQUE classe (préserve
    l'équilibre inter-classe) et les T_b premiers points temporels."""
    n_classes = task.n_classes
    keep_idx = []
    for c in range(n_classes):
        class_rows = (task.labels == c).nonzero(as_tuple=True)[0]
        keep_idx.append(class_rows[:npc_b])
    keep_idx = torch.cat(keep_idx)

    values = task.values[keep_idx, :T_b]
    raw_values = task.raw_values[keep_idx, :T_b]
    labels = task.labels[keep_idx]
    return values, raw_values, labels


class BucketBatchSampler:
    """Itère des batchs d'episode_ids, un bucket à la fois (jamais de mélange
    inter-bucket dans un même batch, donc jamais de padding/masque
    nécessaire). L'ordre des buckets ET des épisodes à l'intérieur d'un
    bucket sont mélangés à chaque epoch -- seule la composition taille par
    taille d'un batch donné est contrainte, pas l'ordre de présentation
    global."""

    def __init__(self, buckets: dict[tuple[int, int], list[dict]], batch_size: int, seed: int = 0,
                 drop_last: bool = True):
        self.buckets = buckets
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        for key, rows in self.buckets.items():
            rows = rows[:]
            self.rng.shuffle(rows)
            n_batches = len(rows) // self.batch_size
            if not self.drop_last and len(rows) % self.batch_size:
                n_batches += 1
            for i in range(n_batches):
                chunk = rows[i * self.batch_size: (i + 1) * self.batch_size]
                if chunk:
                    batches.append((key, chunk))
        self.rng.shuffle(batches)  # ordre des batchs mélangé entre buckets
        yield from batches

    def __len__(self):
        return sum(len(rows) // self.batch_size for rows in self.buckets.values())


def load_batch(corpus_dir: str | Path, bucket_key_: tuple[int, int], rows: list[dict]):
    """Charge et rogne les épisodes d'un batch (déjà choisis par
    BucketBatchSampler, tous du même bucket). Retourne une liste de
    (values, raw_values, labels) rognés -- à empiler ensuite selon la forme
    attendue par votre forward_batch (B épisodes, chacun N=n_classes*npc_b
    séries de T_b points)."""
    corpus_dir = Path(corpus_dir)
    T_b, npc_b = bucket_key_

    by_shard: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_idx"]].append(row)

    episodes = []
    for shard_idx, shard_rows in by_shard.items():
        shard = torch.load(corpus_dir / f"shard_{shard_idx:06d}.pt", weights_only=False)
        for row in shard_rows:
            task = shard[row["idx_in_shard"]]
            episodes.append(crop_episode(task, T_b, npc_b))
    return episodes


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="stats rapides sur les buckets d'un corpus")
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--multiple", type=int, default=16)
    args = ap.parse_args()

    manifest = load_manifest(args.corpus_dir)
    buckets = build_buckets(manifest, multiple=args.multiple)
    sampler = BucketBatchSampler(buckets, batch_size=args.batch_size)

    print(f"{len(manifest)} épisodes -> {len(buckets)} buckets distincts")
    for key, rows in sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  T={key[0]:4d} n_per_class={key[1]:4d} -- {len(rows):4d} épisodes "
              f"({len(rows)//args.batch_size} batchs pleins de taille {args.batch_size})")
    print(f"\nTotal batchs pleins disponibles par epoch: {len(sampler)}")
