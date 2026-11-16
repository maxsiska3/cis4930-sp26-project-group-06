"""E2: one-family-at-a-time learning curves, paired across methods and seeds.

By default use fold-local preprocessing of official raw CSVs; --processed uses
E3's processed CSVs. The latter has training-wide preprocessing and is explicitly
marked as such in outputs. No test labels are used to choose thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from config import (CLASSIFIERS, IMBALANCE_METHODS, N_FOLDS, RARE_FAMILIES,
                    SEED, SMOTE_K, SUBSAMPLE_LEVELS, SUBSAMPLE_SEEDS,
                    TRAIN_PATH, TEST_PATH, make_pipeline, get_fit_params)
from evaluate import evaluate_predictions
from preprocess import Preprocessor, load_raw, split_xy

METHODS = [m for m in IMBALANCE_METHODS if m != "focal_loss"]
KEY = ["family", "classifier", "method", "fraction", "seed", "phase", "fold"]


def retained_count(n, fraction):
    """Round half up; retain one example rather than turn E2 into E3."""
    return min(n, max(1, int(np.floor(n * fraction + 0.5))))


def reduce_family(frame, family, fraction, seed):
    indices = np.flatnonzero(frame.attack_cat.to_numpy() == family)
    if not len(indices):
        raise ValueError(f"No training examples for {family}")
    # Same ordering at every level makes the subsets nested for each seed.
    selected = np.random.default_rng(seed).permutation(indices)[:retained_count(len(indices), fraction)]
    keep = frame.attack_cat.to_numpy() != family
    keep[selected] = True
    return frame.iloc[np.flatnonzero(keep)].reset_index(drop=True)


def choose_threshold(truth, scores):
    """Maximize attack F1 on out-of-fold predictions; ties favor specificity."""
    p, r, thresholds = precision_recall_curve(truth, scores)
    f1 = np.divide(2 * p[:-1] * r[:-1], p[:-1] + r[:-1],
                   out=np.zeros(len(thresholds)), where=(p[:-1] + r[:-1]) != 0)
    return float(thresholds[np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]])


def fit_scores(train, target, classifier, method, raw):
    X, _, families = split_xy(train)
    Xt, _, _ = split_xy(target)
    names = sorted(families.unique())
    codes = pd.Categorical(families, categories=names).codes
    counts = np.bincount(codes)
    if method in {"smote", "adasyn", "smote_enn"} and counts.min() <= SMOTE_K:
        raise ValueError(f"unsupported: smallest class has {counts.min()} examples; fixed k={SMOTE_K} requires {SMOTE_K + 1}")
    if raw:
        pre = Preprocessor().fit(train)
        X, _, _ = split_xy(pre.transform(train))
        Xt, _, _ = split_xy(pre.transform(target))
    pipe = make_pipeline(classifier, method)
    pipe.fit(X, codes, **get_fit_params(method, codes))
    probabilities = pipe.predict_proba(Xt)
    normal_code = names.index("Normal")
    scores = 1 - probabilities[:, list(pipe.classes_).index(normal_code)]
    predictions = np.asarray(names)[pipe.predict(Xt).astype(int)]
    return scores, predictions


def metric_row(target, scores, predicted, threshold, family):
    binary = evaluate_predictions(target.label.to_numpy(), (scores >= threshold).astype(int),
                                  y_score=scores, family_true=target.attack_cat.to_numpy())
    mask = target.attack_cat.to_numpy() == family
    correct = int(((predicted == family) & mask).sum())
    predicted_count = int((predicted == family).sum())
    return {**binary.summary, "threshold": threshold,
            "family_support": int(mask.sum()),
            "family_argmax_detection_recall": float(np.mean(predicted[mask] != "Normal")) if mask.any() else np.nan,
            "family_classification_precision": correct / predicted_count if predicted_count else 0.0,
            "family_detection_recall": binary.per_family.loc[family, "detection_recall"],
            "family_classification_recall": float(np.mean(predicted[mask] == family)) if mask.any() else np.nan}


def recall_curve(group):
    # Fractions rounding to the same count reuse exactly the same seeded sample.
    independent = group.drop_duplicates(["retained", "seed"])
    return independent.groupby("retained").family_detection_recall.agg(["mean", "std", "count"]).sort_index()


def summarize(out, breakdown_drop=0.1):
    rows = pd.read_csv(out)
    rows[(rows.phase == "test") & (rows.status != "ok")].to_csv(
        out.with_name(out.stem + "_unavailable.csv"), index=False)
    final = rows[(rows.phase == "test") & (rows.status == "ok")]
    if final.empty:
        return
    columns = ["retained", "family_detection_recall", "family_classification_recall",
               "family_classification_precision",
               "macro_f1", "pr_auc", "roc_auc", "false_positives_per_100k_normal"]
    summary = final.groupby(["family", "classifier", "method", "fraction"])[columns].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out.with_name(out.stem + "_summary.csv"))
    # Paired differences preserve the same subsampling seed for comparisons.
    baseline = final[final.fraction == 1.0][["family", "classifier", "method", "seed", "family_detection_recall"]]
    paired = final.merge(baseline, on=["family", "classifier", "method", "seed"], suffixes=("", "_baseline"))
    paired["recall_change_from_full"] = paired.family_detection_recall - paired.family_detection_recall_baseline
    paired.to_csv(out.with_name(out.stem + "_paired.csv"), index=False)
    # An operational breakdown on the tested grid, not a universal threshold.
    reduced = paired[paired.fraction < 1].copy()
    floor_rows = []
    for keys, group in reduced.groupby(["family", "classifier", "method"]):
        levels = group.groupby("fraction").agg(
            retained=("retained", "mean"), delta=("recall_change_from_full", "mean"),
            seeds=("seed", "nunique")).sort_index(ascending=False)
        broken = levels[levels.delta <= -breakdown_drop]
        floor_rows.append(dict(zip(["family", "classifier", "method"], keys)) | {
            "recall_drop_criterion": breakdown_drop,
            "first_breakdown_fraction": broken.index[0] if len(broken) else np.nan,
            "first_breakdown_retained_mean": broken.iloc[0].retained if len(broken) else np.nan,
            "minimum_tested_retained_mean": levels.retained.min(),
            "largest_mean_recall_loss": max(0.0, -levels.delta.min()),
            "minimum_successful_seed_count": int(levels.seeds.min()),
            "interpretation": "breakdown observed on tested grid" if len(broken) else "no breakdown observed on successful tested levels"})
    pd.DataFrame(floor_rows).to_csv(out.with_name(out.stem + "_floors.csv"), index=False)
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for classifier, table in final.groupby("classifier"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
        for ax, family in zip(axes.flat, RARE_FAMILIES):
            for method, group in table[table.family == family].groupby("method"):
                curve = recall_curve(group)
                ax.errorbar(curve.index, curve["mean"], yerr=curve["std"].fillna(0), marker="o", label=method)
            ax.set(title=family, xlabel="Actual retained training examples", ylabel="Attack-detection recall", xscale="log", ylim=(0, 1.05))
            if ax.lines:
                ax.legend(fontsize=7)
        fig.suptitle(f"E2 — {classifier}; error bars: sample SD across subsampling seeds")
        fig.tight_layout()
        fig.savefig(out.with_name(f"{out.stem}_{classifier}.png"), dpi=150)
        plt.close(fig)


def run(args):
    out = Path(args.out)
    paths = [Path(TRAIN_PATH), Path(TEST_PATH)] if args.raw else [Path(args.processed_dir) / f"{s}_processed.csv" for s in ("train", "test")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Supply official CSVs with --raw, or E3 processed data with --processed-dir.")
    train, test = [load_raw(p).drop_duplicates().reset_index(drop=True) if args.raw else pd.read_csv(p) for p in paths]
    if not set(RARE_FAMILIES + ["Normal"]).issubset(set(train.attack_cat)):
        raise ValueError("Training data is missing required families")
    if train.attack_cat.value_counts().min() < N_FOLDS:
        raise ValueError("Unreduced training data needs at least five records per family for stratified folds")
    manifest = {"version": 1, "raw": args.raw, "folds": N_FOLDS,
                "libraries": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost"]},
                "data": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                "config": hashlib.sha256(Path(__file__).with_name("config.py").read_bytes()).hexdigest(),
                "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "shared_sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ["preprocess.py", "evaluate.py"]},
                "threshold": "maximum OOF attack F1; highest threshold breaks ties",
                "preprocessing": "fold-local" if args.raw else "E3 training-wide (validation preprocessing leakage limitation)"}
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() and (not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest):
        raise ValueError("Output belongs to a different experiment; choose a new --out path")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=KEY)
    done = set(existing[KEY].itertuples(index=False, name=None))
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, train.attack_cat))
    def save(row):
        row_key = tuple(row[c] for c in KEY)
        if row_key in done:
            return
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        done.add(row_key)
    for family in args.families:
        for classifier in args.classifiers:
            for method in args.methods:
                for fraction in [1.0, *args.levels]:
                    for seed in args.seeds:
                        base = dict(family=family, classifier=classifier, method=method, fraction=fraction, seed=seed)
                        key = (family, classifier, method, fraction, seed, "test", -1)
                        if key in done:
                            continue
                        start = time.monotonic()
                        cv_rows = []
                        try:
                            truth, scores_all = [], []
                            for fold, (tr, va) in enumerate(folds):
                                reduced = reduce_family(train.iloc[tr], family, fraction, seed)
                                validation = train.iloc[va]
                                scores, predicted = fit_scores(reduced, validation, classifier, method, args.raw)
                                truth.extend(validation.label.to_numpy())
                                scores_all.extend(scores)
                                cv_rows.append((fold, reduced, validation, scores, predicted))
                            threshold = choose_threshold(truth, scores_all)
                            reduced = reduce_family(train, family, fraction, seed)
                            scores, predicted = fit_scores(reduced, test, classifier, method, args.raw)
                            for fold, reduced_fold, validation, fold_scores, fold_predicted in cv_rows:
                                save({**base, "phase": "cv", "fold": fold, "status": "ok", "reason": "",
                                      "retained": int((reduced_fold.attack_cat == family).sum()),
                                      "original": int((train.iloc[folds[fold][0]].attack_cat == family).sum()),
                                      "train_rows": len(reduced_fold), "seconds": np.nan,
                                      **metric_row(validation, fold_scores, fold_predicted, threshold, family)})
                            row = {**base, "phase": "test", "fold": -1, "status": "ok", "reason": "",
                                   "retained": int((reduced.attack_cat == family).sum()),
                                   "original": int((train.attack_cat == family).sum()), "train_rows": len(reduced),
                                   "seconds": time.monotonic() - start,
                                   **metric_row(test, scores, predicted, threshold, family)}
                        except ValueError as exc:
                            # Preserve a fixed CSV schema even when the very first run fails.
                            empty = metric_row(test, np.zeros(len(test)), np.repeat("Normal", len(test)), 1.0, family)
                            row = {**base, "phase": "test", "fold": -1,
                                   "status": "unsupported" if str(exc).startswith("unsupported:") else "failed",
                                   "reason": str(exc), "retained": retained_count(int((train.attack_cat == family).sum()), fraction),
                                   "original": int((train.attack_cat == family).sum()), "train_rows": np.nan,
                                   "seconds": time.monotonic() - start, **dict.fromkeys(empty, np.nan)}
                        save(row)
                        print(f"{family} {classifier} {method} {fraction:g} seed={seed}: {row['status']} {row['reason']}", flush=True)
    summarize(out, args.breakdown_drop)
    rows = pd.read_csv(out)
    expected = pd.DataFrame([
        dict(family=f, classifier=c, method=m, fraction=p, seed=s)
        for f in args.families for c in args.classifiers for m in args.methods
        for p in [1.0, *args.levels] for s in args.seeds]).drop_duplicates()
    coverage = expected.merge(rows[rows.phase == "test"], how="left",
                              on=["family", "classifier", "method", "fraction", "seed"])
    coverage["status"] = coverage.status.fillna("missing")
    coverage.to_csv(out.with_name(out.stem + "_coverage.csv"), index=False)
    print("Coverage:", coverage.status.value_counts().to_dict(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--raw", dest="raw", action="store_true", default=True,
                      help="default: fit preprocessing within each reduced training fold")
    mode.add_argument("--processed", dest="raw", action="store_false",
                      help="exploratory E3-data mode; preprocessing is not isolated from validation")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--out", default="results/e2_rare_family.csv")
    parser.add_argument("--families", nargs="+", choices=RARE_FAMILIES, default=RARE_FAMILIES)
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--levels", nargs="+", type=float, default=SUBSAMPLE_LEVELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SUBSAMPLE_SEEDS)
    parser.add_argument("--breakdown-drop", type=float, default=0.1,
                        help="absolute mean detection-recall loss defining breakdown (default .1)")
    args = parser.parse_args()
    if any(not 0 < level < 1 for level in args.levels):
        parser.error("Reduction levels must be strictly between zero and one; the full baseline is automatic")
    if not 0 < args.breakdown_drop <= 1:
        parser.error("--breakdown-drop must be in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
