"""E6: paired Wilcoxon tests of E1 balancing treatments against no balancing.

Run with run_e6.py. Reads saved training-side CV metrics, never raw/test data.
Every comparison needs all matching, successful folds from one E1 run. Holm
correction covers the full selected plan, including unavailable comparisons.

Two-sided signed-rank tests assume symmetric, independent paired differences.
CV training sets overlap, so the fold-level inference is exploratory. Five
nonzero pairs have a minimum exact two-sided p-value of 0.0625; nonsignificance
does not establish equivalence. E2 seeds and E3 families are not CV replicates.

API/method references:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from config import ATTACK_FAMILIES, CLASSIFIERS, IMBALANCE_METHODS, SEED


IDENTITY = ["dataset_id", "seed", "expected_folds", "inner_splits"]
DEFAULT_METRICS = ["rare_family_macro_recall", "macro_f1", "false_positives_per_100k_normal"]
METRIC_DIRECTIONS = {
    name: "higher" for name in (
        "macro_f1", "pr_auc", "average_precision", "roc_auc",
        "detection_macro_f1", "detection_pr_auc", "detection_average_precision", "detection_roc_auc",
        "rare_family_macro_recall", *(f"recall_{family}" for family in ATTACK_FAMILIES),
    )
}
METRIC_DIRECTIONS["false_positives_per_100k_normal"] = "lower"
METRIC_DIRECTIONS.update({"default_" + name: direction for name, direction in list(METRIC_DIRECTIONS.items())})
PAIR_COLUMNS = [*IDENTITY, "task", "classifier", "method", "metric", "fold", "fold_id",
                "baseline_value", "treatment_value", "delta"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_folds(folds):
    """Reject mixed protocols, duplicate observations and invalid split metadata."""
    required = {*IDENTITY, "task", "classifier", "method", "fold", "fold_id", "status",
                "train_rows", "validation_rows"}
    missing = required - set(folds.columns)
    if missing or folds.empty:
        raise ValueError(f"Nonempty E1 fold results required; missing columns: {sorted(missing)}.")
    folds = folds.copy()
    for name in IDENTITY:
        if folds[name].isna().any() or folds[name].nunique() != 1:
            raise ValueError(f"Use one dataset/split protocol per analysis; mixed or missing {name}.")
    for name in ("seed", "expected_folds", "inner_splits", "fold", "train_rows", "validation_rows"):
        values = pd.to_numeric(folds[name], errors="raise")
        if not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError(f"{name} must contain finite integers.")
        folds[name] = values.astype(int)
    expected = int(folds.expected_folds.iloc[0])
    if expected < 2 or folds.inner_splits.iloc[0] < 2:
        raise ValueError("At least two outer and inner folds are required.")
    if not folds.fold.between(1, expected).all() or (folds[["train_rows", "validation_rows"]] <= 0).any().any():
        raise ValueError("Invalid fold numbers or split sizes.")
    for name in ("dataset_id", "fold_id", "task", "classifier", "method", "status"):
        if folds[name].isna().any() or folds[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} must not be missing.")
    if set(folds.status) - {"ok", "error"}:
        raise ValueError("Fold status must be ok or error.")
    if set(folds.task) - {"binary", "multiclass"} or set(folds.classifier) - set(CLASSIFIERS):
        raise ValueError("Unknown task or classifier in E1 results.")
    if set(folds.method) - set(IMBALANCE_METHODS):
        raise ValueError("Unknown treatment in E1 results.")
    for identifier in ("fold", "fold_id"):
        if folds.duplicated(["task", "classifier", "method", identifier]).any():
            raise ValueError(f"Duplicate {identifier} within a classifier/treatment/task.")
    return folds


def load_e1_results(directory):
    """Load one E1 experiment, checking the manifest, split plan and checkpoint rows.

    The archived training source hashes describe the experiment analyzed, even
    if today's E1 code has changed. No training or prediction files are opened.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    folds = pd.read_csv(directory / "fold_results.csv", dtype={"dataset_id": str, "fold_id": str})
    expected = {
        "dataset_id": manifest["dataset_id"], "seed": manifest["seed"],
        "expected_folds": manifest["n_splits"], "inner_splits": manifest["inner_splits"],
    }
    filled = []
    # The earlier E1 baseline omitted this column; its manifest still records it.
    if "inner_splits" not in folds:
        folds["inner_splits"] = expected["inner_splits"]
        filled.append("inner_splits")
    folds = validate_folds(folds)
    for name, value in expected.items():
        if not folds[name].eq(value).all():
            raise ValueError(f"Fold results disagree with the E1 manifest: {name}.")
    if not (folds.train_rows + folds.validation_rows).eq(manifest["rows"]).all():
        raise ValueError("Split sizes disagree with the manifest's training population.")
    split_plan = json.loads((directory / "split_plan.json").read_text())
    if len(split_plan) != expected["expected_folds"] or {s["fold"] for s in split_plan} != set(range(1, len(split_plan) + 1)):
        raise ValueError("Invalid E1 outer split plan.")
    validation_memberships = []
    for split in split_plan:
        fit = np.asarray(split["train_rows"])
        validation = np.asarray(split["validation_rows"])
        if any(values.ndim != 1 or values.dtype.kind not in "iu" for values in (fit, validation)):
            raise ValueError("E1 split memberships must be one-dimensional integer indices.")
        fit, validation = fit.astype(np.int64), validation.astype(np.int64)
        population = np.r_[fit, validation]
        if (len(population) != manifest["rows"] or len(np.unique(population)) != manifest["rows"]
                or population.min() != 0 or population.max() != manifest["rows"] - 1):
            raise ValueError("E1 split has duplicate, overlapping or missing training/validation members.")
        validation_memberships.extend(validation)
        fold_id = hashlib.sha256(validation.tobytes()).hexdigest()[:16]
        rows = folds[folds.fold == split["fold"]]
        if (not rows.fold_id.eq(fold_id).all() or not rows.train_rows.eq(len(fit)).all()
                or not rows.validation_rows.eq(len(validation)).all()):
            raise ValueError("Fold results disagree with the saved outer split plan.")
    if (len(validation_memberships) != manifest["rows"]
            or len(np.unique(validation_memberships)) != manifest["rows"]):
        raise ValueError("Outer validation folds must cover every selected record exactly once.")
    if "checkpoint" not in folds:
        raise ValueError("E1 checkpoint keys are required to verify exported metrics.")
    for _, row in folds.iterrows():
        key = str(row.checkpoint)
        if Path(key).name != key:
            raise ValueError("Checkpoint keys must be filenames, not paths.")
        saved = json.loads((directory / "checkpoints" / (key + ".json")).read_text())
        for name in (*IDENTITY, "task", "classifier", "method", "fold", "fold_id", "train_rows", "validation_rows"):
            if name not in filled and saved.get(name) != row[name]:
                raise ValueError(f"Fold table disagrees with checkpoint {key}: {name}.")
        if row.status != "ok":
            continue  # E1 may have demoted an old success after detecting corrupt artifacts.
        if saved.get("status") != "ok":
            raise ValueError(f"Successful fold has no successful checkpoint: {key}.")
        for name in METRIC_DIRECTIONS:
            if name not in folds:
                continue
            actual, original = row[name], saved.get(name)
            if pd.isna(actual) and (original is None or pd.isna(original)):
                continue
            if original is None or not np.isclose(float(actual), float(original), rtol=1e-12, atol=1e-12):
                raise ValueError(f"Fold table disagrees with checkpoint {key}: {name}.")
    provenance = {
        "e1_manifest": manifest, "filled_from_manifest": filled,
        "input_sha256": {name: file_hash(directory / name)
                         for name in ("manifest.json", "fold_results.csv", "split_plan.json")},
    }
    return folds, provenance


def holm_adjust(pvalues, planned_count):
    """Holm step-down FWER correction; unavailable planned tests count as p=1."""
    pvalues = np.asarray(pvalues, dtype=float)
    finite = np.flatnonzero(np.isfinite(pvalues))
    if (pvalues.ndim != 1 or np.isinf(pvalues).any()
            or not isinstance(planned_count, (int, np.integer)) or planned_count < len(pvalues)
            or ((pvalues[finite] < 0) | (pvalues[finite] > 1)).any()):
        raise ValueError("Invalid p-values or planned hypothesis count.")
    order = finite[np.argsort(pvalues[finite], kind="stable")]
    adjusted = np.full(len(pvalues), np.nan)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        (planned_count - np.arange(len(order))) * pvalues[order]
    ))
    return adjusted


def signed_rank(differences, *, seed=SEED):
    """Two-sided signed-rank permutation test with ties and zero handling.

    Round to 12 decimals before ranking; drop exact zeros (Wilcox convention).
    Enumerate all signs up to 16 nonzero pairs. Larger inputs use 65,535 random
    permutations with the recorded seed. All-zero differences return p=1.
    """
    differences = np.round(np.asarray(differences, dtype=float), 12)
    if differences.ndim != 1 or len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("Signed-rank tests need at least two finite paired differences.")
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return dict(statistic=0.0, p_value=1.0, nonzero_pairs=0, rank_biserial=np.nan,
                    p_method="all_zero", note="All paired differences are zero.")
    ranks = stats.rankdata(abs(nonzero), method="average")
    rank_biserial = float(np.sum(ranks * np.sign(nonzero)) / ranks.sum())
    if len(nonzero) == 1:
        statistic, p_value, method_name = 0.0, 1.0, "exhaustive"
    else:
        exact = len(nonzero) <= 16
        method = stats.PermutationMethod(n_resamples=np.inf if exact else 65_535, random_state=seed)
        result = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method=method)
        statistic, p_value = float(result.statistic), float(result.pvalue)
        method_name = "exhaustive" if exact else "Monte Carlo (65535, fixed seed)"
    return dict(statistic=statistic, p_value=p_value, nonzero_pairs=len(nonzero),
                rank_biserial=rank_biserial, p_method=method_name, note="")


def _pair_rows(baseline, treatment, expected, metric):
    if len(baseline) != expected or len(treatment) != expected:
        return None, "Missing baseline or treatment folds; all expected folds are required."
    if not baseline.status.eq("ok").all() or not treatment.status.eq("ok").all():
        return None, "Baseline or treatment contains failed folds."
    if set(baseline.fold_id) != set(treatment.fold_id):
        return None, "Baseline and treatment fold IDs differ."
    pair = baseline.merge(treatment, on="fold_id", suffixes=("_baseline", "_treatment"), validate="one_to_one")
    for name in ("fold", "train_rows", "validation_rows"):
        if not pair[name + "_baseline"].eq(pair[name + "_treatment"]).all():
            return None, f"Paired {name} values differ."
    if metric not in baseline or metric not in treatment:
        return None, f"Metric {metric} was not recorded."
    for suffix in ("_baseline", "_treatment"):
        pair[metric + suffix] = pd.to_numeric(pair[metric + suffix], errors="coerce")
        if not np.isfinite(pair[metric + suffix]).all():
            return None, f"Metric {metric} is undefined in at least one fold; no pairs were dropped."
        upper = 100_000 if METRIC_DIRECTIONS[metric] == "lower" else 1
        if not pair[metric + suffix].between(0, upper).all():
            return None, f"Metric {metric} lies outside its expected units/range."
    return pair.sort_values("fold_baseline"), ""


def analyze(folds, *, tasks=("binary", "multiclass"), classifiers=CLASSIFIERS,
            methods=IMBALANCE_METHODS, metrics=DEFAULT_METRICS, alpha=0.05):
    """Execute a fixed comparison plan; never infer from a partial set of pairs."""
    folds = validate_folds(folds)
    tasks, classifiers, methods, metrics = [list(dict.fromkeys(x)) for x in (tasks, classifiers, methods, metrics)]
    if (not tasks or not classifiers or not metrics or set(tasks) - {"binary", "multiclass"}
            or set(classifiers) - set(CLASSIFIERS) or set(methods) - set(IMBALANCE_METHODS)
            or set(metrics) - set(METRIC_DIRECTIONS) or not 0 < alpha < 1):
        raise ValueError("Invalid analysis plan or alpha.")
    plan = [(task, classifier, method, metric) for task in tasks for classifier in classifiers
            for method in methods if method != "none"
            and (method != "focal_loss" or classifier in ("mlp", "xgboost")) for metric in metrics]
    if not plan:
        raise ValueError("Select at least one valid treatment other than none.")
    identity = {name: folds[name].iloc[0] for name in IDENTITY}
    expected = int(identity["expected_folds"])
    comparisons, paired_rows = [], []
    for task, classifier, method, metric in plan:
        subset = folds[(folds.task == task) & (folds.classifier == classifier)]
        base = dict(**identity, task=task, classifier=classifier, method=method, baseline="none",
                    metric=metric, better=METRIC_DIRECTIONS[metric], alpha=alpha,
                    test="wilcoxon", alternative="two-sided", planned_hypotheses=len(plan))
        pair, reason = _pair_rows(subset[subset.method == "none"], subset[subset.method == method], expected, metric)
        if pair is None:
            comparisons.append(dict(**base, status="unavailable", reason=reason, n_pairs=0, p_value=np.nan))
            continue
        baseline, treatment = pair[metric + "_baseline"].to_numpy(), pair[metric + "_treatment"].to_numpy()
        differences = np.round(treatment - baseline, 12)
        test = signed_rank(differences, seed=int(identity["seed"]))
        mean = float(differences.mean())
        comparisons.append(dict(
            **base, **test, status="ok", reason="", n_pairs=len(pair),
            baseline_mean=float(baseline.mean()), treatment_mean=float(treatment.mean()),
            mean_delta=mean, median_delta=float(np.median(differences)), std_delta=float(differences.std(ddof=1)),
            mean_improvement=mean * (1 if base["better"] == "higher" else -1),
            folds_improved=int(np.sum(differences > 0 if base["better"] == "higher" else differences < 0)),
            folds_tied=int(np.sum(differences == 0)),
        ))
        for (_, row), difference in zip(pair.iterrows(), differences):
            paired_rows.append(dict(**identity, task=task, classifier=classifier, method=method, metric=metric,
                                    fold=int(row.fold_baseline), fold_id=row.fold_id,
                                    baseline_value=row[metric + "_baseline"], treatment_value=row[metric + "_treatment"],
                                    delta=float(difference)))
    results = pd.DataFrame(comparisons)
    results["p_holm"] = holm_adjust(results.p_value, len(plan))
    results["significant_holm"] = results.status.eq("ok") & results.p_holm.le(alpha)
    return results, pd.DataFrame(paired_rows, columns=PAIR_COLUMNS)


def write_summary(path, results):
    available = results[results.status == "ok"]
    significant = available[available.significant_holm]
    lines = ["# E6: paired Wilcoxon comparisons against no balancing", "",
             f"Planned comparisons: {len(results)}; available tests: {len(available)}.",
             f"Holm-adjusted significant comparisons: {len(significant)}.", "",
             "Each test compares one treatment with none on every matching E1 fold, within",
             "the same dataset, training task and classifier. Failed/missing/undefined folds",
             "make that comparison unavailable; they are never silently removed. Holm",
             "correction includes all planned comparisons, including unavailable ones.", "",
             "Differences are treatment minus baseline. Positive recall/F1 differences are",
             "gains; positive false-alarm differences are costs. Recall/F1 differences of",
             "0.01 mean one percentage point. False alarms remain per 100,000 normal",
             "connections. macro_f1 means binary macro-F1 for binary training and family",
             "classification macro-F1 for multiclass training; detection_macro_f1 is binary.", "",
             "Limitations: Wilcoxon assumes independent, symmetric paired differences; CV",
             "training sets overlap, so inference here is exploratory. Five nonzero pairs",
             "have a minimum exact two-sided p-value of 0.0625, already above 0.05 before",
             "Holm adjustment. Fewer nonzero pairs provide still less resolution. Do not",
             "change the test direction after inspecting results. Nonsignificance does not",
             "establish equivalence, and E1 cannot establish unseen-family generalization.", "",
             "The current experiment's source hashes and protocol are recorded in manifest.json;",
             "these may describe an older E1 run rather than today's implementation."]
    if not len(available):
        lines.extend(["", "No significance conclusions are available. Complete the matching E1 baseline",
                      "and treatment folds first. See comparisons.csv for each unavailable reason."])
    elif not significant.empty:
        lines.extend(["", "Significant comparisons:"])
        for _, row in significant.iterrows():
            lines.append(f"- {row.task} / {row.classifier} / {row.method}: {row.metric}, "
                         f"delta={row.mean_delta:.6g}, adjusted p={row.p_holm:.6g}.")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("results/e1"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/e6"))
    parser.add_argument("--tasks", nargs="+", choices=["binary", "multiclass"], default=["binary", "multiclass"])
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS)
    parser.add_argument("--methods", nargs="+", choices=IMBALANCE_METHODS, default=IMBALANCE_METHODS)
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRIC_DIRECTIONS), default=DEFAULT_METRICS)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.out_dir.exists() and (not args.out_dir.is_dir() or any(args.out_dir.iterdir())):
        parser.error("Choose a new --out-dir; existing results will not be overwritten.")
    print("Reading saved E1 fold results; no training or test-partition access.", flush=True)
    try:
        folds, provenance = load_e1_results(args.input_dir)
        results, pairs = analyze(folds, tasks=args.tasks, classifiers=args.classifiers,
                                 methods=args.methods, metrics=args.metrics, alpha=args.alpha)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "comparisons.csv", index=False)
    pairs.to_csv(args.out_dir / "paired_differences.csv", index=False)
    write_summary(args.out_dir / "E6_SUMMARY.md", results)
    metadata = dict(
        schema_version=1, **provenance, analysis_source_sha256=file_hash(__file__),
        tasks=list(dict.fromkeys(args.tasks)), classifiers=list(dict.fromkeys(args.classifiers)),
        methods=list(dict.fromkeys(args.methods)), metrics=list(dict.fromkeys(args.metrics)),
        alpha=args.alpha, test="wilcoxon", alternative="two-sided", zero_method="wilcox",
        correction="Holm across every selected task/classifier/treatment/metric combination",
        difference_rounding_decimals=12, permutation_exact_max_nonzero=16,
        permutation_mc_resamples=65_535,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
    )
    (args.out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    complete = int(results.status.eq("ok").sum())
    print(f"Available tests: {complete}/{len(results)}; Holm-significant: {int(results.significant_holm.sum())}.", flush=True)
    print(f"Saved comparisons, paired differences, summary and provenance to {args.out_dir.resolve()}.", flush=True)
    return int(complete != len(results))
