"""
Agrège les résultats de la grille OOD "mécanisme progressif" lancée par
`ood_mechanism_progressive.sbatch` -- lit chaque
`runs/*_ood_mech_holdout-*_k*/eval_ood_*/summary_eval.json` +
`eval_indist/summary_eval.json`, et trace accuracy vs nombre de mécanismes
d'entraînement vus (k), une courbe par mécanisme exclu (root_family=ar fixée
partout).

Variante de `aggregate_ood_results.py` (expérience "famille exclue") --
préfixe de nom de run différent (`ood_mech_holdout` vs `ood_holdout`) pour
que les deux expériences ne se mélangent jamais dans une même agrégation.

Usage :
    python aggregate_ood_mechanism_results.py --runs-dir runs/ --out ood_mechanism_curves.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_RE = re.compile(r"_ood_mech_holdout-(?P<held_out>[a-zA-Z0-9]+)_k(?P<k>\d+)_")


def collect(runs_dir: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """Retourne {held_out_mechanism: [(k, ood_acc, ood_std, indist_acc, ood_baseline), ...]}."""

    results: dict[str, list[tuple[int, float, float, float, float]]] = {}

    for run_dir in sorted(runs_dir.iterdir()):
        m = RUN_RE.search(run_dir.name)
        if not m:
            continue
        held_out = m.group("held_out")
        k = int(m.group("k"))

        ood_glob = list(run_dir.glob("eval_ood_*/summary_eval.json"))
        indist_summary = run_dir / "eval_indist" / "summary_eval.json"
        if not ood_glob or not indist_summary.exists():
            print(f"[skip] {run_dir.name} -- résultats manquants (job pas encore fini ?)")
            continue

        ood = json.loads(ood_glob[0].read_text())
        indist = json.loads(indist_summary.read_text())

        results.setdefault(held_out, []).append(
            (k, ood["acc_mean"], ood["acc_std"], indist["acc_mean"], ood["baseline_knn_acc"])
        )

    for held_out in results:
        results[held_out].sort(key=lambda row: row[0])  # trier par k croissant

    return results


def plot(results: dict[str, list[tuple[int, float, float, float, float]]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10.colors

    for i, (held_out, rows) in enumerate(sorted(results.items())):
        ks          = [r[0] for r in rows]
        ood_acc     = [r[1] for r in rows]
        ood_std     = [r[2] for r in rows]
        indist_acc  = [r[3] for r in rows]
        c = colors[i % len(colors)]

        ax.errorbar(ks, ood_acc, yerr=ood_std, marker="o", color=c,
                    label=f"OOD -- mécanisme exclu: {held_out}", linestyle="-")
        ax.plot(ks, indist_acc, marker="s", color=c, alpha=0.4, linestyle="--",
                label=f"in-distribution -- exclu: {held_out}")

    if results:
        n_classes_guess = 4  # cf. n{n_nodes}_c{n_classes} -- ajuster si besoin
        ax.axhline(1 / n_classes_guess, color="gray", linestyle=":", label="hasard")

    ax.set_xlabel("nombre de mécanismes d'entraînement (k) -- root_family=ar fixée")
    ax.set_ylabel("accuracy (séries requête)")
    ax.set_title("Généralisation OOD par mécanisme exclu, en fonction de k")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[OK] figure -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("ood_mechanism_curves.png"))
    args = ap.parse_args()

    results = collect(args.runs_dir)
    if not results:
        print("Aucun run OOD mécanisme complet trouvé sous", args.runs_dir)
        return

    for held_out, rows in sorted(results.items()):
        print(f"\n== mécanisme exclu: {held_out} ==")
        for k, ood_acc, ood_std, indist_acc, baseline in rows:
            print(f"  k={k}  OOD={ood_acc:.3f}±{ood_std:.3f}  in-dist={indist_acc:.3f}  baseline-kNN(OOD)={baseline:.3f}")

    plot(results, args.out)


if __name__ == "__main__":
    main()
