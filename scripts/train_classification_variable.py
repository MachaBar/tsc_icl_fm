"""
Entraînement sur le corpus À TAILLES VARIABLES (generate_variable_corpus.py) :
n_classes (2-15), longueur T (100-1000) et taille d'épisode variables par
épisode, potentiellement déséquilibré (--imbalance-mode dirichlet).

Contrairement à train_classification.py (corpus fixe, un n_classes/length
uniques pour tout le run), ce script combine 3 mécanismes pour gérer cette
variabilité SANS toucher à UnivariatePerceiverEncoder/SeriesPooler :

  1. BUCKETING (bucketed_batching.py) -- un batch ne mélange que des épisodes
     de (T, n_per_class réel minimal) proches, arrondis au multiple de 16
     inférieur -- jamais de padding sur T, seulement du rognage.

  2. PADDING À MAX_CLASSES + MASQUE D'ATTENTION -- le modèle est construit
     une seule fois avec num_classes=MAX_CLASSES (tête de taille fixe). Un
     épisode à n_classes < MAX_CLASSES est complété par des blocs de classes
     fictives (valeurs nulles), exclus de l'attention via key_padding_mask
     (voir le patch de ICLearningClassification.forward -- déjà appliqué).

  3. TRAIN_FRAC PAR BATCH -- un seul ratio contexte/requête tiré par batch
     (pas par épisode), inspiré du pattern set_batch_params() du repo
     TS-ICL (redraw partagé au niveau batch, pas par exemple).

Le split contexte/requête est appliqué PAR BLOC DE CLASSE (réel ou fictif),
à la même position n_train_pc pour tous les blocs et tous les épisodes du
batch -- ce qui rend train_size = MAX_CLASSES * n_train_pc constant pour
tout le batch, contrainte requise par `Encoder.forward(attn_mask=train_size)`
(un entier unique, pas un masque par-épisode).

Usage (depuis la racine de tsc_icl_fm, corpus généré par
generate_variable_corpus.py) :
    uv run python -m scripts.train_classification_variable \
        --corpus-dir /path/to/out/tasks_variable \
        --out runs/variable_run/ \
        --max-classes 15 --pooling perceiver --batch-size 8 --max-steps 20000
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.modules.aroma import EncoderICLClassifier
from src.modules.aroma.aroma.encoder import UnivariatePerceiverEncoder
from src.modules.icl_learning import ICLearningClassification, POOLING_TYPES, SeriesPooler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# bucketed_batching.py / load_variable_corpus.py doivent être sur le PYTHONPATH
# (copiez-les dans scripts/ à côté de ce fichier, comme les autres utilitaires)
sys.path.insert(0, str(Path(__file__).parent))
from bucketed_batching import build_buckets, BucketBatchSampler, crop_episode, min_class_count  # noqa: E402
from load_variable_corpus import load_manifest  # noqa: E402


# ----------------------------------------------------------------------
# Modèle -- identique à train_classification.py, num_classes = MAX_CLASSES
# ----------------------------------------------------------------------

def build_model(args: argparse.Namespace, max_classes: int) -> EncoderICLClassifier:
    encoder = UnivariatePerceiverEncoder(
        input_dim=1, num_channels=1,
        num_latents=args.num_latents, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        depth=args.encoder_depth, latent_heads=args.latent_heads, latent_dim_head=args.latent_dim_head,
        cross_heads=args.cross_heads, cross_dim_head=args.cross_dim_head,
        max_pos_encoding_freq=args.max_pos_encoding_freq, num_freq=args.num_freq,
        encode_geo=args.encode_geo, include_pos_in_value=args.include_pos_in_value,
        use_kl=args.use_kl, use_cls_token=(args.pooling == "cls"),
    )
    pooler = SeriesPooler(
        pooling=args.pooling, d_model=encoder.latent_out_dim, num_classes=max_classes,
        include_label_token=args.include_label_token, heads=args.pooler_heads, dim_head=args.pooler_dim_head,
    )
    head = ICLearningClassification(
        d_model=args.icl_d_model, num_blocks=args.icl_blocks, nhead=args.icl_nhead,
        dim_feedforward=args.icl_dim_feedforward, num_classes=max_classes, dropout=args.icl_dropout,
        inject_labels=not args.include_label_token,
    )
    return EncoderICLClassifier(encoder=encoder, pooler=pooler, head=head)


# ----------------------------------------------------------------------
# Chargement + split train/val/test au niveau ÉPISODE (le corpus variable
# n'a pas de split intégré comme generate_synthetic_corpus.py -- chaque
# "tâche" ici est un épisode unique, jamais rééchantillonné).
#
#   train : utilisé pour la descente de gradient
#   val   : surveillé PENDANT l'entraînement -- choix de best.pt, early
#           stopping (voir la boucle principale). Vu indirectement plusieurs
#           fois au cours du run (via la sélection du meilleur checkpoint),
#           donc pas un chiffre final "propre".
#   test  : JAMAIS touché avant la toute fin -- à évaluer une seule fois,
#           après la fin de l'entraînement (voir evaluate_split ci-dessous
#           ou un script séparé façon eval_ou_holdout.py), pour le chiffre
#           à rapporter.
# ----------------------------------------------------------------------

def load_and_split_manifest(
    corpus_dir: Path, val_frac: float, test_frac: float, seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    manifest = load_manifest(corpus_dir)
    rng = random.Random(seed)  # même seed => même split à chaque relance -- NE PAS changer
    manifest = manifest[:]     # entre deux runs du même run_id, sous peine de fuite train/test
    rng.shuffle(manifest)

    n_val = max(1, int(round(val_frac * len(manifest))))
    n_test = max(1, int(round(test_frac * len(manifest))))
    val_rows = manifest[:n_val]
    test_rows = manifest[n_val:n_val + n_test]
    train_rows = manifest[n_val + n_test:]

    logger.info(f"[Data] {len(manifest)} épisodes -- {len(train_rows)} train / {len(val_rows)} val / "
                f"{len(test_rows)} test (split au niveau épisode, seed={seed})")
    return train_rows, val_rows, test_rows


def evaluate_split(model: EncoderICLClassifier, corpus_dir: Path, rows: list[dict], args: argparse.Namespace,
                    device: torch.device, train_frac: float | None = None) -> tuple[float, float]:
    """Un seul passage complet sur `rows` (val OU test), sans gradient.
    Utilisé pendant l'entraînement pour `val`, et UNE SEULE FOIS à la fin
    pour `test`."""
    buckets = build_buckets(rows, multiple=args.bucket_multiple, min_val=args.bucket_min_val)
    sampler = BucketBatchSampler(buckets, batch_size=args.batch_size, seed=args.seed, drop_last=False)
    tf = train_frac if train_frac is not None else sum(args.train_frac_range) / 2

    model.eval()
    loss_sum, acc_sum, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for bucket_key, batch_rows in sampler:
            if not batch_rows:
                continue
            batch = make_batch_variable(corpus_dir, bucket_key, batch_rows, args.max_classes, tf, device)
            ce_loss, acc = forward_batch_variable(model, batch)
            loss_sum += ce_loss.item(); acc_sum += acc.item(); n_batches += 1
    return loss_sum / max(1, n_batches), acc_sum / max(1, n_batches)


# ----------------------------------------------------------------------
# Batching : bucket -> charge -> rogne -> padde à MAX_CLASSES -> split
# contexte/requête par bloc de classe -- voir le docstring du module
# ----------------------------------------------------------------------

def load_bucket_batch(corpus_dir: Path, rows: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Charge+rogne les épisodes d'un batch (tous du même bucket T_b/npc_b,
    déjà déterminés par le sampler). Retourne (values, labels, n_classes_reel)
    par épisode, encore SANS padding MAX_CLASSES ni split contexte/requête."""
    by_shard: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_idx"]].append(row)

    out = []
    for shard_idx, shard_rows in by_shard.items():
        shard = torch.load(corpus_dir / f"shard_{shard_idx:06d}.pt", weights_only=False)
        for row in shard_rows:
            task = shard[row["idx_in_shard"]]
            npc_b = min_class_count(row)
            # crop_episode rogne à npc_b par classe -- npc_b vient de
            # min_class_count (classe la moins représentée), jamais dépassé
            values, _raw, labels = crop_episode(task, T_b=row["_T_b"], npc_b=npc_b)
            out.append((values, labels, row["n_classes"]))
    return out


def make_batch_variable(
    corpus_dir: Path,
    bucket_key: tuple[int, int],
    rows: list[dict],
    max_classes: int,
    train_frac: float,
    device: torch.device,
) -> dict:
    """Construit un batch complet : padding MAX_CLASSES + masque d'attention
    + split contexte/requête par bloc de classe (train_size constant pour
    tout le batch, voir docstring du module)."""
    T_b, npc_b = bucket_key
    for row in rows:
        row["_T_b"] = T_b  # transmis à load_bucket_batch (évite un 2e passage sur le manifest)

    episodes = load_bucket_batch(corpus_dir, rows)  # [(values (n_c*npc_b, T_b), labels, n_classes), ...]

    n_train_pc = max(1, min(npc_b - 1, round(train_frac * npc_b)))  # >=1 côté contexte ET requête
    B = len(episodes)
    N_pad = max_classes * npc_b
    train_size = max_classes * n_train_pc

    values_batch = torch.zeros(B, N_pad, T_b)
    labels_batch = torch.zeros(B, N_pad, dtype=torch.long)
    key_padding_mask = torch.ones(B, N_pad, dtype=torch.bool)  # True = padding, par défaut tout masqué
    n_classes_batch = torch.zeros(B, dtype=torch.long)

    for b, (values, labels, n_classes) in enumerate(episodes):
        n_classes_batch[b] = n_classes
        ctx_slots, qry_slots = [], []  # positions dans le tenseur final (0..N_pad-1)
        ctx_write, qry_write = 0, train_size  # curseurs d'écriture

        for c in range(max_classes):
            is_real_class = c < n_classes
            if is_real_class:
                block_values = values[c * npc_b:(c + 1) * npc_b]
                block_labels = labels[c * npc_b:(c + 1) * npc_b]
            else:
                block_values = torch.zeros(npc_b, T_b)
                block_labels = torch.zeros(npc_b, dtype=torch.long)

            values_batch[b, ctx_write:ctx_write + n_train_pc] = block_values[:n_train_pc]
            labels_batch[b, ctx_write:ctx_write + n_train_pc] = block_labels[:n_train_pc]
            key_padding_mask[b, ctx_write:ctx_write + n_train_pc] = not is_real_class
            ctx_write += n_train_pc

            n_q = npc_b - n_train_pc
            values_batch[b, qry_write:qry_write + n_q] = block_values[n_train_pc:]
            labels_batch[b, qry_write:qry_write + n_q] = block_labels[n_train_pc:]
            key_padding_mask[b, qry_write:qry_write + n_q] = not is_real_class
            qry_write += n_q

    series = values_batch.unsqueeze(-1).to(device)  # (B, N_pad, T_b, 1)
    coords = torch.linspace(0.0, 1.0, T_b, device=device).view(1, 1, T_b, 1).expand_as(series).contiguous()
    labels_batch = labels_batch.to(device)
    key_padding_mask = key_padding_mask.to(device)
    n_classes_batch = n_classes_batch.to(device)

    y_train = labels_batch[:, :train_size]
    y_query = labels_batch[:, train_size:]
    query_pad_mask = key_padding_mask[:, train_size:]  # True = ligne de requête fictive, à exclure de la loss

    return {
        "series": series, "coords": coords, "y_train": y_train, "y_query": y_query,
        "key_padding_mask": key_padding_mask, "query_pad_mask": query_pad_mask,
        "n_classes_batch": n_classes_batch, "max_classes": max_classes,
    }


def forward_batch_variable(model: EncoderICLClassifier, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    logits, _ = model(
        series=batch["series"], coords=batch["coords"], y_train=batch["y_train"],
        key_padding_mask=batch["key_padding_mask"],
    )  # (B, n_query, MAX_CLASSES)

    max_classes = batch["max_classes"]
    class_idx = torch.arange(max_classes, device=logits.device).view(1, 1, max_classes)
    invalid_class = class_idx >= batch["n_classes_batch"].view(-1, 1, 1)  # (B, 1, MAX_CLASSES)
    logits = logits.masked_fill(invalid_class, float("-inf"))

    real_row_mask = ~batch["query_pad_mask"]  # (B, n_query) -- True = vraie ligne de requête
    logits_flat = logits[real_row_mask]         # (n_real, MAX_CLASSES)
    labels_flat = batch["y_query"][real_row_mask]  # (n_real,)

    ce_loss = nn.functional.cross_entropy(logits_flat, labels_flat)
    acc = (logits_flat.argmax(dim=-1) == labels_flat).float().mean()
    return ce_loss, acc


def infinite_bucket_batches(sampler: BucketBatchSampler):
    while True:
        for bucket_key, rows in sampler:
            yield bucket_key, rows


# ----------------------------------------------------------------------
# Visualisation -- identique à train_classification.py
# ----------------------------------------------------------------------

def plot_history(history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["step"], history["train_loss"], label="train")
    axes[0].plot(history["step"], history["val_loss"], label="val")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("cross-entropy"); axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["step"], history["train_acc"], label="train")
    axes[1].plot(history["step"], history["val_acc"], label="val")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy (hasard variable par batch, non tracé)"); axes[1].legend()

    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--corpus-dir", type=Path, required=True, help="dossier généré par generate_variable_corpus.py (manifest.jsonl + shard_*.pt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-classes", type=int, default=15, help="taille fixe de la tête de sortie -- doit couvrir le n_classes max réellement présent dans le corpus (voir n_classes_range de dag.py, borne haute par défaut 15)")

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--bucket-multiple", type=int, default=16)
    ap.add_argument("--bucket-min-val", type=int, default=16)
    ap.add_argument("--train-frac-range", type=float, nargs=2, default=[0.3, 0.8], help="train_frac tiré uniformément dans cette plage, UNE FOIS PAR BATCH")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction des épisodes réservée à la validation (surveillée PENDANT l'entraînement -- best.pt, early stopping)")
    ap.add_argument("--test-frac", type=float, default=0.1, help="fraction des épisodes réservée au test (JAMAIS touchée avant la fin -- un seul passage, chiffre final)")

    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip-value", type=float, default=1.0)
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="none")

    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--num-latents", type=int, default=16)
    ap.add_argument("--encoder-depth", type=int, default=2)
    ap.add_argument("--latent-heads", type=int, default=4)
    ap.add_argument("--latent-dim-head", type=int, default=32)
    ap.add_argument("--cross-heads", type=int, default=4)
    ap.add_argument("--cross-dim-head", type=int, default=32)
    ap.add_argument("--max-pos-encoding-freq", type=int, default=4)
    ap.add_argument("--num-freq", type=int, default=12)
    ap.add_argument("--encode-geo", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include-pos-in-value", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use-kl", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--pooling", choices=POOLING_TYPES, default="mean")
    ap.add_argument("--include-label-token", action="store_true")
    ap.add_argument("--pooler-heads", type=int, default=4)
    ap.add_argument("--pooler-dim-head", type=int, default=32)

    ap.add_argument("--icl-d-model", type=int, default=64)
    ap.add_argument("--icl-blocks", type=int, default=2)
    ap.add_argument("--icl-nhead", type=int, default=4)
    ap.add_argument("--icl-dim-feedforward", type=int, default=128)
    ap.add_argument("--icl-dropout", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)

    args = ap.parse_args()
    if args.pooling == "cls" and args.include_label_token:
        ap.error("--pooling cls et --include-label-token sont mutuellement exclusifs")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ckpt").mkdir(exist_ok=True)
    (args.out / "plots").mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=args.out / "tensorboard")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Device] using {device}")

    # 1/3 DONNÉES

    train_rows, val_rows, test_rows = load_and_split_manifest(args.corpus_dir, args.val_frac, args.test_frac, args.seed)
    train_buckets = build_buckets(train_rows, multiple=args.bucket_multiple, min_val=args.bucket_min_val)

    train_sampler = BucketBatchSampler(train_buckets, batch_size=args.batch_size, seed=args.seed, drop_last=True)
    logger.info(f"[Data] {len(train_buckets)} buckets train, {len(train_sampler)} batchs pleins/epoch "
                f"(taille {args.batch_size}) -- hasard = 1/n_classes, variable par épisode")

    # 2/3 MODÈLE

    model = build_model(args, args.max_classes).to(device)
    num_params_encoder = sum(p.numel() for p in model.encoder.parameters())
    num_params_head = sum(p.numel() for p in model.head.parameters())
    logger.info(f"[Model] EncoderICLClassifier -- encodeur: {num_params_encoder:,d} params, "
                f"tête ICL: {num_params_head:,d} params (MAX_CLASSES={args.max_classes})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps) if args.scheduler == "cosine" else None

    # 3/3 BOUCLE D'ENTRAÎNEMENT

    history = {"step": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    evals_since_improvement = 0
    running_loss, running_acc, n_running = 0.0, 0.0, 0

    train_iter = infinite_bucket_batches(train_sampler)
    logger.info(f"[Training] start -- {args.max_steps} steps max, éval toutes les {args.eval_every} steps")

    for step in range(args.max_steps):
        model.train()
        bucket_key, rows = next(train_iter)
        train_frac = random.uniform(*args.train_frac_range)  # UN tirage par batch, cf. docstring
        batch = make_batch_variable(args.corpus_dir, bucket_key, rows, args.max_classes, train_frac, device)
        ce_loss, acc = forward_batch_variable(model, batch)

        optimizer.zero_grad()
        ce_loss.backward()
        if args.grad_clip_value > 0:
            nn.utils.clip_grad_value_(model.parameters(), clip_value=args.grad_clip_value)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("loss/train_step", ce_loss.item(), step)
        writer.add_scalar("accuracy/train_step", acc.item(), step)
        running_loss += ce_loss.item(); running_acc += acc.item(); n_running += 1

        is_last_step = (step == args.max_steps - 1)
        if ((step + 1) % args.eval_every != 0) and not is_last_step:
            continue

        train_loss = running_loss / max(1, n_running)
        train_acc = running_acc / max(1, n_running)
        running_loss, running_acc, n_running = 0.0, 0.0, 0

        val_loss, val_acc = evaluate_split(model, args.corpus_dir, val_rows, args, device)

        history["step"].append(step)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        writer.add_scalar("loss/val", val_loss, step)
        writer.add_scalar("accuracy/val", val_acc, step)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

        logger.info(f"[Training] step {step:05d} -- train loss {train_loss:.4f} acc {train_acc:.3f} "
                    f"| val loss {val_loss:.4f} acc {val_acc:.3f}")
        plot_history(history, args.out / "plots" / "training_curves.png")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            evals_since_improvement = 0
            torch.save({
                "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
                "args": vars(args), "max_classes": args.max_classes,
            }, args.out / "ckpt" / "best.pt")
            logger.info(f"[Training] step {step:05d} -- nouveau meilleur val_acc={val_acc:.3f}, ckpt sauvegardé")
        else:
            evals_since_improvement += 1
            if args.patience > 0 and evals_since_improvement >= args.patience:
                logger.info(f"[Training] step {step:05d} -- early stopping (meilleur={best_val_acc:.3f})")
                break

    writer.close()
    logger.info(f"[Training] terminé -- meilleur val_acc={best_val_acc:.3f}")

    # ÉVALUATION FINALE SUR TEST -- un seul passage, jamais utilisé avant ce
    # point (ni pour l'entraînement, ni pour le choix du checkpoint). Recharge
    # le MEILLEUR checkpoint (best.pt, sélectionné sur val), pas le modèle en
    # mémoire au dernier step (qui peut avoir régressé depuis).
    best_ckpt = torch.load(args.out / "ckpt" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    test_loss, test_acc = evaluate_split(model, args.corpus_dir, test_rows, args, device)
    logger.info(f"[Test] (best.pt, step={best_ckpt['step']}) -- loss={test_loss:.4f} acc={test_acc:.3f} "
                f"sur {len(test_rows)} épisodes jamais vus avant cette évaluation")
    with (args.out / "test_result.json").open("w") as f:
        import json
        json.dump({"test_loss": test_loss, "test_acc": test_acc, "best_step": best_ckpt["step"],
                    "n_test_episodes": len(test_rows)}, f, indent=2)


if __name__ == "__main__":
    main()
