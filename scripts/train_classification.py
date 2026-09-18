"""
Lit le corpus au format PARQUET via `corpus_io.py`.

python -m scripts.train_classification \
    --corpus-dir out/tasks/linear_ou_n12_c4_seed0_len512/ \
    --out runs/first_classif_run/ \
    --pooling mean --batch-size 8 --max-steps 2000 --eval-every 100

-corpus-dir accepte PLUSIEURS dossiers (ex: plusieurs graines/racines de la
même famille -- même mechanism/n_nodes/n_classes, seed différente), combinés
en un seul pool train/eval -- plus de variété entre épisodes, toujours
in-distribution :
    python -m scripts.train_classification \
        --corpus-dir out/tasks/mlp_ou_n12_c4_seed0_len512/ \
                     out/tasks/mlp_ou_n12_c4_seed1_len512/ \
                     out/tasks/mlp_ou_n12_c4_seed2_len512/ \
        --out runs/multi_seed_run/ ...
(n_classes et length doivent être identiques entre tous les dossiers combinés
-- voir corpus_io.load_corpus_shards_multi)

(chaque --corpus-dir doit contenir des shard_*.parquet + metadata.json,
générés côté projet génération avec `python generate_synthetic_corpus.py --format parquet ...`)


TensorBoard : logs écrits dans --out/tensorboard/ -- loss/acc train à CHAQUE
step (courbe brute, bruitée), loss/acc/lr val à chaque `--eval-every`.
Nécessite `tensorboard` installé (`uv add tensorboard`).
Visualiser : `tensorboard --logdir <--out>/tensorboard/`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless (noeud de calcul SLURM sans serveur X)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.modules.aroma import EncoderICLClassifier
from src.modules.aroma.aroma.encoder import UnivariatePerceiverEncoder
from src.modules.icl_learning import ICLearningClassification, POOLING_TYPES, SeriesPooler

from scripts.corpus_io import load_corpus_shards_parquet, load_corpus_shards_multi, make_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------

def build_model(args: argparse.Namespace, n_classes: int) -> EncoderICLClassifier:
    encoder = UnivariatePerceiverEncoder(
        input_dim       = 1,
        num_channels    = 1,
        num_latents     = args.num_latents,
        hidden_dim      = args.hidden_dim,
        latent_dim      = args.latent_dim,
        depth           = args.encoder_depth,
        latent_heads    = args.latent_heads,
        latent_dim_head = args.latent_dim_head,
        cross_heads     = args.cross_heads,
        cross_dim_head  = args.cross_dim_head,
        max_pos_encoding_freq = args.max_pos_encoding_freq,
        num_freq              = args.num_freq,
        encode_geo            = args.encode_geo,
        include_pos_in_value  = args.include_pos_in_value,
        use_kl          = args.use_kl,
        use_cls_token   = (args.pooling == "cls"),
    )
    pooler = SeriesPooler(
        pooling             = args.pooling,
        d_model             = encoder.latent_out_dim,
        num_classes         = n_classes,
        include_label_token = args.include_label_token,
        heads               = args.pooler_heads,
        dim_head            = args.pooler_dim_head,
    )
    head = ICLearningClassification(
        d_model         = args.icl_d_model,
        num_blocks      = args.icl_blocks,
        nhead           = args.icl_nhead,
        dim_feedforward = args.icl_dim_feedforward,
        num_classes     = n_classes,
        dropout         = args.icl_dropout,
        # early fusion (pooler.include_label_token) et late fusion (inject_labels)
        # sont mutuellement exclusives, sinon le label serait ajouté deux fois :
        inject_labels   = not args.include_label_token,
    )
    return EncoderICLClassifier(encoder=encoder, pooler=pooler, head=head)


# ----------------------------------------------------------------------
# Batching (échantillonne des épisodes, pas des lignes -- un "batch" est un
# groupe d'épisodes indépendants, chacun avec son propre split contexte/requête)
# ----------------------------------------------------------------------

def batches(episodes: list, batch_size: int, shuffle: bool):
    n = len(episodes)
    order = torch.randperm(n).tolist() if shuffle else list(range(n))
    for i in range(0, n, batch_size):
        idx = order[i : i + batch_size]
        yield [episodes[j] for j in idx]


def infinite_batches(episodes: list, batch_size: int):
    """Repioche indéfiniment dans `episodes`, en remélangeant à chaque tour
    complet -- pour un entraînement cadencé en steps, sans notion d'epoch."""
    while True:
        yield from batches(episodes, batch_size, shuffle=True)


def forward_batch(
    model: EncoderICLClassifier,
    episodes: list,
    train_frac: float,
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    series, coords, y_train, y_query = make_batch(episodes, train_frac, length)
    series, coords   = series.to(device), coords.to(device)
    y_train, y_query = y_train.to(device), y_query.to(device)

    # kl_loss ignoré : encodeur en --use-kl False (déterministe) par défaut, donc toujours 0.
    logits, _ = model(series=series, coords=coords, y_train=y_train)
    ce_loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_query.reshape(-1))
    acc = (logits.argmax(dim=-1) == y_query).float().mean()

    return ce_loss, acc


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------

def plot_history(history: dict, n_classes: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["step"], history["train_loss"], label="train")
    axes[0].plot(history["step"], history["val_loss"], label="val")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["step"], history["train_acc"], label="train")
    axes[1].plot(history["step"], history["val_acc"], label="val")
    axes[1].axhline(1 / n_classes, color="gray", linestyle="--", linewidth=1, label="hasard")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ap.add_argument("--corpus-dir", type=Path, required=True, help="dossier contenant les shard_*.parquet + metadata.json (voir generate_synthetic_corpus.py --format parquet)")
    ap.add_argument("--corpus-dir", type=Path, nargs="+", required=True, help="un ou plusieurs dossiers shard_*.parquet + metadata.json -- plusieurs = pool combiné (mêmes n_classes/length requis, voir docstring)")
    ap.add_argument("--out", type=Path, required=True, help="dossier de sortie (ckpt/, plots/)")

    # données
    ap.add_argument("--batch-size", type=int, default=8, help="nb d'épisodes par batch")
    ap.add_argument("--train-frac", type=float, default=0.6, help="fraction contexte/requête au sein d'un épisode (indépendant du split train/eval des épisodes eux-mêmes)")

    # optimisation
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=100, help="cadence (en steps) de la validation, du log, du plot et du checkpoint")
    ap.add_argument("--patience", type=int, default=0, help="arrête l'entraînement si val_acc ne s'améliore pas pendant N évaluations consécutives, càd N * --eval-every steps (0 = désactivé, entraîne jusqu'à --max-steps)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip-value", type=float, default=1.0, help="0 désactive le clipping (même primitive que aroma_train_fn: clip_grad_value_)")
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="none")

    # modèle -- encodeur 
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--num-latents", type=int, default=16)
    ap.add_argument("--encoder-depth", type=int, default=2)
    ap.add_argument("--latent-heads", type=int, default=4)
    ap.add_argument("--latent-dim-head", type=int, default=32)
    ap.add_argument("--cross-heads", type=int, default=4)
    ap.add_argument("--cross-dim-head", type=int, default=32)
    # encode_geo/include_pos_in_value/max_pos_encoding_freq/num_freq sont utilisés dans le
    # géo-encoder et le value-encoder (bloc①), donc AVANT le early-return de `return_latents=True` --
    # contrairement à ce qui suit le bottleneck (scales/mlp_feature_dim/decoder_ff, bloc③, jamais
    # atteint par notre forward), ils affectent bien la qualité de Z_val. Défauts alignés sur la
    # config TS-ICL forecasting (max_pos_encoding_freq=10, num_freq=32, encode_geo/include_pos_in_value=True) ;
    # laissés à False/valeurs plus petites par défaut ici pour un premier run plus léger.
    ap.add_argument("--max-pos-encoding-freq", type=int, default=4)
    ap.add_argument("--num-freq", type=int, default=12)
    ap.add_argument("--encode-geo", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include-pos-in-value", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use-kl", action=argparse.BooleanOptionalAction, default=False, help="bottleneck VAE stochastique (défaut: déterministe, plus simple pour un premier run)")

    # modèle -- pooling (Z_val -> une représentation par série)
    ap.add_argument("--pooling", choices=POOLING_TYPES, default="mean")
    ap.add_argument("--include-label-token", action="store_true", help="fusion précoce du label (voir SeriesPooler); incompatible avec --pooling cls")
    ap.add_argument("--pooler-heads", type=int, default=4)
    ap.add_argument("--pooler-dim-head", type=int, default=32)

    # modèle -- tête ICL
    ap.add_argument("--icl-d-model", type=int, default=64)
    ap.add_argument("--icl-blocks", type=int, default=2)
    ap.add_argument("--icl-nhead", type=int, default=4)
    ap.add_argument("--icl-dim-feedforward", type=int, default=128)
    ap.add_argument("--icl-dropout", type=float, default=0.0)

    # divers
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, help="défaut: cuda si disponible, sinon cpu")

    args = ap.parse_args()

    if args.pooling == "cls" and args.include_label_token:
        ap.error("--pooling cls et --include-label-token sont mutuellement exclusifs (voir SeriesPooler)")

    torch.manual_seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ckpt").mkdir(exist_ok=True)
    (args.out / "plots").mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=args.out / "tensorboard")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[Device] using {}".format(device))

    # 1/3 DONNÉES

    # data = load_corpus_shards_parquet(args.corpus_dir)
    data = load_corpus_shards_multi(args.corpus_dir)
    train_episodes = data["train_episodes"]
    eval_episodes  = data["eval_episodes"]
    n_classes      = data["n_classes"]
    length         = data["length"]

    logger.info(
        "[Data] {} corpus combiné(s) -- {} épisodes train / {} épisodes eval, n_classes={}, length={} "
        "(mécanismes={}, dag_modes={})".format(
            data["n_corpora"], len(train_episodes), len(eval_episodes), n_classes, length,
            data["mechanism"], data["dag_mode"]
        )
    )
    logger.info("[Data] hasard = {:.3f} (accuracy d'un classifieur aléatoire à {} classes)".format(1 / n_classes, n_classes))

    # 2/3 MODÈLE

    model = build_model(args, n_classes).to(device)
    num_params_encoder = sum(p.numel() for p in model.encoder.parameters())
    num_params_head     = sum(p.numel() for p in model.head.parameters())
    logger.info(
        "[Model] EncoderICLClassifier -- encodeur: {:,d} params, tête ICL: {:,d} params "
        "(pooling={}, include_label_token={})".format(
            num_params_encoder, num_params_head, args.pooling, args.include_label_token
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps) if args.scheduler == "cosine" else None

    # 3/3 BOUCLE D'ENTRAÎNEMENT (par steps)

    history = {"step": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    evals_since_improvement = 0
    running_loss, running_acc, n_running = 0.0, 0.0, 0  # accumulés depuis le dernier eval

    train_iter = infinite_batches(train_episodes, args.batch_size)

    logger.info("[Training] start -- {} steps max, éval toutes les {} steps".format(args.max_steps, args.eval_every))

    for step in range(args.max_steps):

        model.train()
        batch_episodes = next(train_iter)
        ce_loss, acc = forward_batch(model, batch_episodes, args.train_frac, length, device)

        optimizer.zero_grad()
        ce_loss.backward()
        if args.grad_clip_value > 0:
            nn.utils.clip_grad_value_(model.parameters(), clip_value=args.grad_clip_value)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("loss/train_step", ce_loss.item(), step)
        writer.add_scalar("accuracy/train_step", acc.item(), step)

        running_loss += ce_loss.item()
        running_acc  += acc.item()
        n_running    += 1

        is_last_step = (step == args.max_steps - 1)
        if ((step + 1) % args.eval_every != 0) and not is_last_step:
            continue

        train_loss = running_loss / max(1, n_running)
        train_acc  = running_acc / max(1, n_running)
        running_loss, running_acc, n_running = 0.0, 0.0, 0

        # validation -- pool eval, lui-même artificiellement split train/query
        # par make_batch (voir --train-frac), jamais vu pendant l'entraînement
        model.eval()
        val_loss, val_acc, n_val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch_episodes in batches(eval_episodes, args.batch_size, shuffle=False):
                ce_loss, acc = forward_batch(model, batch_episodes, args.train_frac, length, device)
                val_loss += ce_loss.item()
                val_acc  += acc.item()
                n_val_batches += 1
        val_loss /= max(1, n_val_batches)
        val_acc  /= max(1, n_val_batches)

        history["step"].append(step)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        writer.add_scalar("loss/val", val_loss, step)
        writer.add_scalar("accuracy/val", val_acc, step)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

        logger.info(
            "[Training] step {:05d} -- train loss {:.4f} acc {:.3f} | val loss {:.4f} acc {:.3f}".format(
                step, train_loss, train_acc, val_loss, val_acc
            )
        )
        plot_history(history, n_classes, args.out / "plots" / "training_curves.png")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            evals_since_improvement = 0
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "args": vars(args),
                    "n_classes": n_classes,
                },
                args.out / "ckpt" / "best.pt",
            )
            logger.info("[Training] step {:05d} -- nouveau meilleur val_acc={:.3f}, ckpt sauvegardé".format(step, val_acc))
        else:
            evals_since_improvement += 1
            if args.patience > 0 and evals_since_improvement >= args.patience:
                logger.info(
                    "[Training] step {:05d} -- early stopping, val_acc n'a pas progressé depuis {} évaluations "
                    "(meilleur={:.3f})".format(step, evals_since_improvement, best_val_acc)
                )
                break

    writer.close()
    logger.info("[Training] terminé -- meilleur val_acc={:.3f} (hasard={:.3f})".format(best_val_acc, 1 / n_classes))


if __name__ == "__main__":
    main()
