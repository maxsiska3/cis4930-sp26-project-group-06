"""E4: transfer matrix - train on one attack family, test on every family.

For each of the nine attack families (the "source"), a binary detector is
trained on normal traffic plus ONLY that family. It is then scored on the full
official test set, and we record how much of every family (the "target") it
flags as an attack. The result is a 9 x 9 matrix:

    rows    = family the model was trained on
    columns = family it was tested on
    diagonal       = detection of the family it saw (seen attack)
    off-diagonal   = detection of families it never saw (transfer)

The off-diagonal cells explain E3: if no other family's detector catches a
family here, we expect E3 to miss it when it is left out of training.

Threshold: chosen on out-of-fold predictions from the training subset only
(maximum attack F1, same rule as E2). Test labels never choose anything.

Usage:
    python e4_transfer.py                          # raw CSVs in data/
    python e4_transfer.py --processed              # quick run on processed/ CSVs
    python e4_transfer.py --classifiers xgboost --methods none smote
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import (ATTACK_FAMILIES, CLASSIFIERS, IMBALANCE_METHODS, N_FOLDS,
                    SEED, TEST_PATH, TRAIN_PATH, get_fit_params, make_pipeline)
from e2_rare_family import choose_threshold
from evaluate import evaluate_predictions
from preprocess import Preprocessor, load_raw, split_xy

METHODS = [m for m in IMBALANCE_METHODS if m != "focal_loss"]
KEY = ["source", "classifier", "method"]


def source_subset(train, source):
    """Training rows for one source family: all Normal rows plus that family."""
    keep = train.attack_cat.isin(["Normal", source])
    return train[keep].reset_index(drop=True)


def fit_scores(train, target, classifier, method, raw):
    """Fit a binary attack detector on `train`; return P(attack) for `target`.

    In raw mode the preprocessor is fitted on `train` only, so nothing about
    the target rows (validation or test) leaks into encoding or medians.
    """
    if raw:
        pre = Preprocessor().fit(train)
        train, target = pre.transform(train), pre.transform(target)
    X, y, _ = split_xy(train)
    Xt, _, _ = split_xy(target)
    pipe = make_pipeline(classifier, method)
    pipe.fit(X, y, **get_fit_params(method, y))
    attack_column = list(pipe.classes_).index(1)
    return pipe.predict_proba(Xt)[:, attack_column]


def pick_threshold(subset, classifier, method, raw):
    """Out-of-fold threshold using only the training subset."""
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    truth, scores = [], []
    for tr, va in folds.split(subset, subset.label):
        scores.extend(fit_scores(subset.iloc[tr], subset.iloc[va], classifier, method, raw))
        truth.extend(subset.label.iloc[va])
    return choose_threshold(truth, scores)


def run_source(train, test, source, classifier, method, raw):
    """Train on one source family and return one result row per target family."""
    subset = source_subset(train, source)
    threshold = pick_threshold(subset, classifier, method, raw)
    scores = fit_scores(subset, test, classifier, method, raw)
    result = evaluate_predictions(test.label.to_numpy(), (scores >= threshold).astype(int),
                                  y_score=scores, family_true=test.attack_cat.to_numpy())
    rows = []
    for target in ATTACK_FAMILIES:
        rows.append({"source": source, "classifier": classifier, "method": method,
                     "target": target, "seen": target == source,
                     "detection_recall": result.per_family.loc[target, "detection_recall"],
                     "target_support": int(result.per_family.loc[target, "support"]),
                     "source_train_rows": int((subset.attack_cat == source).sum()),
                     "threshold": threshold,
                     "false_positives_per_100k_normal": result.summary["false_positives_per_100k_normal"],
                     "roc_auc": result.summary["roc_auc"]})
    return rows


def matrix(rows, classifier, method):
    """9 x 9 table of detection recall: rows = trained on, columns = tested on."""
    part = rows[(rows.classifier == classifier) & (rows.method == method)]
    return part.pivot(index="source", columns="target", values="detection_recall") \
               .reindex(index=ATTACK_FAMILIES, columns=ATTACK_FAMILIES)


def coverage(rows):
    """For each target family, the best detection by a model that never saw it.

    A low value means no other family 'covers' it, so E3 should miss it too.
    """
    unseen = rows[~rows.seen]
    best = unseen.loc[unseen.groupby(["classifier", "method", "target"]).detection_recall.idxmax()]
    return best.rename(columns={"source": "best_other_source",
                                "detection_recall": "best_unseen_recall"})[
        ["classifier", "method", "target", "best_other_source", "best_unseen_recall"]]


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (classifier, method), _ in rows.groupby(["classifier", "method"]):
        table = matrix(rows, classifier, method)
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(table.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(9), ATTACK_FAMILIES, rotation=45, ha="right")
        ax.set_yticks(range(9), ATTACK_FAMILIES)
        ax.set(xlabel="Tested on (family)", ylabel="Trained on (normal + this family)",
               title=f"E4 transfer matrix: {classifier}, {method}")
        for i in range(9):
            for j in range(9):
                value = table.iat[i, j]
                ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if value > 0.6 else "black")
            ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, lw=1.5, ec="black"))
        fig.colorbar(image, ax=ax, label="Detection recall (share flagged as attack)")
        fig.tight_layout()
        fig.savefig(out.with_name(f"{out.stem}_{classifier}_{method}.png"), dpi=150)
        plt.close(fig)


def run(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.raw:
        paths = [Path(TRAIN_PATH), Path(TEST_PATH)]
    else:
        paths = [Path(args.processed_dir) / f"{s}_processed.csv" for s in ("train", "test")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
    # Same data handling as E2: raw CSVs with exact duplicates dropped.
    if args.raw:
        train, test = [load_raw(p).drop_duplicates().reset_index(drop=True) for p in paths]
    else:
        train, test = [pd.read_csv(p) for p in paths]

    # Resume: skip (source, classifier, method) combinations already saved.
    done = set()
    if out.exists():
        done = set(pd.read_csv(out)[KEY].drop_duplicates().itertuples(index=False, name=None))

    for classifier in args.classifiers:
        for method in args.methods:
            for source in args.sources:
                if (source, classifier, method) in done:
                    continue
                start = time.monotonic()
                rows = run_source(train, test, source, classifier, method, args.raw)
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{classifier} {method} trained on {source}: "
                      f"{time.monotonic() - start:.0f}s", flush=True)

    rows = pd.read_csv(out)
    coverage(rows).to_csv(out.with_name(out.stem + "_coverage.csv"), index=False)
    plot(rows, out)
    print(f"Saved {out}, coverage table and heatmaps.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--raw", dest="raw", action="store_true", default=True,
                      help="default: official CSVs in data/, preprocessing fitted per training subset")
    mode.add_argument("--processed", dest="raw", action="store_false",
                      help="quick mode: reuse processed/ CSVs (preprocessing fitted on full train)")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--out", default="results/e4_transfer.csv")
    parser.add_argument("--sources", nargs="+", choices=ATTACK_FAMILIES, default=ATTACK_FAMILIES)
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS,
                        default=["random_forest", "xgboost"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["none", "smote"])
    run(parser.parse_args())


if __name__ == "__main__":
    main()