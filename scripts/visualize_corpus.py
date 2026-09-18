"""
Recharge un corpus déjà généré (parquet, voir generate_synthetic_corpus.py
--format parquet) et trace les épisodes de son choix.
Zéro dépendance à `synthgen` (voir corpus_io.py).

Usage :
    # épisode 0 et le dernier du pool eval, per_class + overlay + raw_vs_norm
    python -m scripts.visualize_corpus \
        --corpus-dir out/tasks/mlp_ou_n12_c4_seed0_len512/ \
        --split eval --indices 0,-1

    # un épisode précis du pool train, juste l'overlay
    python -m scripts.visualize_corpus \
        --corpus-dir out/tasks/mlp_ou_n12_c4_seed0_len512/ \
        --split train --indices 42 --plots overlay
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.corpus_io import (
    load_corpus_shards_parquet,
    plot_episode,
    plot_overlay,
    plot_raw_vs_normalized,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, required=True, help="dossier shard_*.parquet + metadata.json")
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--indices", type=str, default="0,-1", help="indices (dans la liste chargée) séparés par des virgules, ex: '0,-1,42' -- indices négatifs autorisés (Python)")
    ap.add_argument("--plots", type=str, default="per_class,overlay", help="sous-ensemble parmi per_class,overlay,raw_vs_norm, séparés par des virgules")
    ap.add_argument("--out", type=Path, default=None, help="défaut : <corpus-dir>/plots_extra/")
    args = ap.parse_args()

    data = load_corpus_shards_parquet(args.corpus_dir)
    episodes = data["train_episodes"] if args.split == "train" else data["eval_episodes"]
    print(f"[Viz] {len(episodes)} épisodes dans le pool '{args.split}'")

    out_dir = args.out or (args.corpus_dir / "plots_extra")
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(i) for i in args.indices.split(",")]
    plots = set(args.plots.split(","))

    for idx in indices:
        episode = episodes[idx]  # indexation Python normale -- -1 fonctionne
        tag = f"{args.split}_{idx}"

        if "per_class" in plots:
            out_path = out_dir / f"{tag}_per_class.png"
            plot_episode(episode, title=f"{args.split} #{idx}", out_path=out_path)
            print(f"[Viz] -> {out_path}")

        if "overlay" in plots:
            out_path = out_dir / f"{tag}_overlay.png"
            plot_overlay(episode, title=f"{args.split} #{idx} -- toutes classes superposées", out_path=out_path)
            print(f"[Viz] -> {out_path}")

        if "raw_vs_norm" in plots:
            out_path = out_dir / f"{tag}_raw_vs_norm.png"
            plot_raw_vs_normalized(episode, series_idx=0, title=f"{args.split} #{idx}", out_path=out_path)
            print(f"[Viz] -> {out_path}")


if __name__ == "__main__":
    main()
