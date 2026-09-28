#!/usr/bin/env python3
"""Final, one-shot OOD evaluation on the HOLDOUT seeds (90-99) -- never used
for training (--corpus-dir) nor for live monitoring (--ood-corpus-dir) during
any of the 3 progressive runs. Run this once per checkpoint you want to
compare, AFTER training is fully done.

Reuses build_model / forward_batch / batches from train_classification.py
so the eval matches training exactly (same model construction, same
context/query split logic via --train-frac).

Usage (from tsc_icl_fm repo root, same env as training):
    uv run python scripts/eval_ou_holdout.py \
        --ckpt runs/<run_id>/ckpt/best.pt \
        --holdout-corpus-dir \
            /home/d32485/synthetic-ts-classif/out/tasks/mlp_ou_n12_c4_seed90_len512 \
            ... seed91 ... seed99 \
        --batch-size 8 --train-frac 0.6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.corpus_io import load_corpus_shards_multi
from scripts.train_classification import build_model, batches, forward_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="ckpt/best.pt d'un des 3 runs (11/25/50 train seeds)")
    ap.add_argument("--holdout-corpus-dir", type=Path, nargs="+", required=True, help="seeds 90-99, jamais vues (ni training ni monitoring)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=None, help="défaut: reprend la valeur utilisée à l'entraînement (dans le ckpt)")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = argparse.Namespace(**ckpt["args"])
    train_frac = args.train_frac if args.train_frac is not None else train_args.train_frac

    data = load_corpus_shards_multi(args.holdout_corpus_dir)
    holdout_episodes = data["train_episodes"] + data["eval_episodes"]
    n_classes = data["n_classes"]
    length = data["length"]

    model = build_model(train_args, n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loss_sum, acc_sum, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for batch_episodes in batches(holdout_episodes, args.batch_size, shuffle=False):
            ce_loss, acc = forward_batch(model, batch_episodes, train_frac, length, device)
            loss_sum += ce_loss.item()
            acc_sum  += acc.item()
            n_batches += 1

    holdout_loss = loss_sum / max(1, n_batches)
    holdout_acc  = acc_sum / max(1, n_batches)

    print(f"[Holdout eval] ckpt={args.ckpt}")
    print(f"[Holdout eval] entraîné avec val_acc(in-dist)={ckpt.get('val_acc', float('nan')):.3f} au step {ckpt.get('step', '?')}")
    print(f"[Holdout eval] {data['n_corpora']} corpus holdout, {len(holdout_episodes)} épisodes, n_classes={n_classes}")
    print(f"[Holdout eval] loss={holdout_loss:.4f}  acc={holdout_acc:.3f}  (hasard={1/n_classes:.3f})")


if __name__ == "__main__":
    main()
