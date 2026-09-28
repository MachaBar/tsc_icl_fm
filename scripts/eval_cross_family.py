#!/usr/bin/env python3
"""Cross-family OOD eval: same edge mechanism (mlp), but a DIFFERENT root
family per corpus than the one the model was trained on (ou), evaluated on
seeds NEVER seen during training (not seed0, which -- as established earlier
-- shares its exact DAG topology/edge weights with the ou seed0 the model was
trained on, so it isn't really testing structural generalization). Each
--corpus-dir can be passed MULTIPLE seeds per family; results are grouped and
averaged PER FAMILY (mean +/- std across its seeds) so one lucky/unlucky seed
doesn't skew the per-family number -- the point is to see WHICH families the
model generalizes to, not just a global average.

Family/seed are parsed from each --corpus-dir's basename, expected shape
"<mechanism>_<family>_n<N>_c<C>_seed<S>_len<L>" (e.g. mlp_gp_n12_c4_seed90_len512
-> family="gp", seed=90). Reuses build_model / forward_batch / batches from
train_classification.py so the eval matches training exactly.

Usage (from tsc_icl_fm repo root, module mode so relative imports resolve):
    uv run python -m scripts.eval_cross_family \
        --ckpt runs/<run_id>/ckpt/best.pt \
        --corpus-dir \
            /home/d32485/synthetic-ts-classif/out/tasks/mlp_gp_n12_c4_seed90_len512 \
            /home/d32485/synthetic-ts-classif/out/tasks/mlp_gp_n12_c4_seed91_len512 \
            /home/d32485/synthetic-ts-classif/out/tasks/mlp_ets_n12_c4_seed90_len512 \
            ... \
        --batch-size 8
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import torch

from scripts.corpus_io import load_corpus_shards_multi
from scripts.train_classification import build_model, batches, forward_batch

_NAME_RE = re.compile(r"^[a-zA-Z]+_([a-zA-Z]+)_n\d+_c\d+_seed(\d+)_len\d+$")


def family_and_seed_from_dirname(d: Path) -> tuple[str, int | None]:
    m = _NAME_RE.match(d.name)
    return (m.group(1), int(m.group(2))) if m else (d.name, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, nargs="+", required=True,
                     help="un ou plusieurs dossiers par famille (mêmes mécanisme, root_family différente, "
                          "seeds au choix -- moyenné par famille sur toutes les seeds passées pour cette famille)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=None, help="défaut: reprend la valeur utilisée à l'entraînement")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--warn-train-seeds", type=int, nargs="*", default=list(range(50)),
                     help="seeds à signaler si présentes dans --corpus-dir (par défaut 0-49, le pool d'entraînement "
                          "'ou' -- même seed = même topologie/poids DAG que ce que le modèle a vu, donc pas un vrai test OOD)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = argparse.Namespace(**ckpt["args"])
    train_frac = args.train_frac if args.train_frac is not None else train_args.train_frac

    # n_classes/model construits une seule fois -- identiques pour toutes les familles (même c4)
    model = None
    n_classes_ref = None
    per_seed_rows = []
    leaked_seeds = []

    for d in args.corpus_dir:
        family, seed = family_and_seed_from_dirname(d)
        if seed is not None and seed in args.warn_train_seeds:
            leaked_seeds.append(d.name)

        data = load_corpus_shards_multi([d])
        episodes = data["train_episodes"] + data["eval_episodes"]
        n_classes = data["n_classes"]

        if model is None:
            model = build_model(train_args, n_classes).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            n_classes_ref = n_classes
        elif n_classes != n_classes_ref:
            raise ValueError(f"{d.name}: n_classes={n_classes} != {n_classes_ref} (premier corpus) -- pas comparable dans le même tableau")

        length = data["length"]
        loss_sum, acc_sum, n_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch_episodes in batches(episodes, args.batch_size, shuffle=False):
                ce_loss, acc = forward_batch(model, batch_episodes, train_frac, length, device)
                loss_sum += ce_loss.item()
                acc_sum  += acc.item()
                n_batches += 1

        per_seed_rows.append({
            "family": family,
            "seed": seed,
            "n_episodes": len(episodes),
            "loss": loss_sum / max(1, n_batches),
            "acc": acc_sum / max(1, n_batches),
        })

    if leaked_seeds:
        print(f"[ATTENTION] {len(leaked_seeds)} dossier(s) utilisent une seed du pool d'entraînement "
              f"(mêmes topologie/poids DAG que le modèle a déjà vus pour 'ou') -- ce n'est pas un vrai test OOD "
              f"pour ces dossiers, seulement un changement de signal d'entrée sur un circuit déjà vu :")
        for name in leaked_seeds:
            print(f"    {name}")
        print()

    # agrégation par famille -- moyenne/écart-type sur toutes les seeds passées pour cette famille
    families = sorted(set(r["family"] for r in per_seed_rows))
    rows = []
    for family in families:
        fam_rows = [r for r in per_seed_rows if r["family"] == family]
        accs = [r["acc"] for r in fam_rows]
        rows.append({
            "family": family,
            "n_seeds": len(fam_rows),
            "n_episodes": sum(r["n_episodes"] for r in fam_rows),
            "loss": statistics.mean(r["loss"] for r in fam_rows),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        })
    rows.sort(key=lambda r: -r["acc_mean"])

    chance = 1 / n_classes_ref
    print(f"ckpt={args.ckpt}")
    print(f"train_acc(in-dist, au meilleur step)={ckpt.get('val_acc', float('nan')):.3f}  |  hasard={chance:.3f}\n")
    header = f"{'family':<12} {'n_seeds':>7} {'n_episodes':>10} {'loss':>8} {'acc_mean':>9} {'acc_std':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['family']:<12} {r['n_seeds']:>7} {r['n_episodes']:>10} {r['loss']:>8.4f} {r['acc_mean']:>9.3f} {r['acc_std']:>8.3f}")
    print("-" * len(header))
    mean_acc = statistics.mean(r["acc_mean"] for r in rows)
    print(f"{'MEAN':<12} {'':>7} {'':>10} {'':>8} {mean_acc:>9.3f}")


if __name__ == "__main__":
    main()
