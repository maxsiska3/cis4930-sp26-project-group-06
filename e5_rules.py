"""E5: rule extraction - how resampling moves Decision Tree and Random Forest boundaries.

For Decision Tree and Random Forest, with no balancing and with each resampling
method, a family classifier (Normal + nine attack families) is trained on the
official training partition. We then compare, before and after resampling:

    feature importances   impurity (MDI) importance; for RF also how often
                          each feature is the root split of a tree
    decision rules        the top of each tree and the leaf rules for the rare
                          families (and Fuzzers, the family E3/E4 miss),
                          with thresholds converted back to original units
    synthetic leaves      leaves predicting a family that contain NO real
                          training record of that family - regions created
                          only by synthetic (SMOTE/ADASYN) records

The Random Forest has 200 deep trees, so its readable rules come from a
surrogate tree (256 leaves) fitted to the forest's predictions on the real training data;
its fidelity (agreement with the forest on the test set, overall and per family) is reported.

Detection uses 1 - P(Normal) with a threshold chosen on out-of-fold training
predictions (maximum attack F1, as in E2/E4). Test labels never choose
anything; they are only used to describe how many test records reach a rule.

Usage:
    python e5_rules.py                          # raw CSVs in data/
    python e5_rules.py --processed              # quick run on processed/ CSVs
    python e5_rules.py --classifiers decision_tree --methods none smote
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from config import (ATTACK_FAMILIES, CATEGORICAL_COLUMNS, IMBALANCE_METHODS, LOG_COLUMNS,
                    N_FOLDS, RARE_FAMILIES, SEED, TEST_PATH, TRAIN_PATH, get_fit_params,
                    make_pipeline)
from e2_rare_family import choose_threshold
from evaluate import evaluate_predictions
from preprocess import Preprocessor, load_raw, split_xy

CLASSIFIERS = ["decision_tree", "random_forest"]
METHODS = [m for m in IMBALANCE_METHODS if m != "focal_loss"]
FAMILIES = ["Normal"] + ATTACK_FAMILIES
RULE_FAMILIES = RARE_FAMILIES + ["Fuzzers"]   # rare families + the one E3/E4 miss
KEY = ["classifier", "method"]
SURROGATE_LEAVES = 256  # size of the readable stand-in for the forest (see E5_WRITEUP.md)
TOP_LEVELS = 3          # depth of the "top of the tree" printout
TOP_RULES = 5           # leaf rules printed per family
ONE_HOT_PREFIXES = tuple(f"{c}_" for c in CATEGORICAL_COLUMNS)


# Fitting and scoring
class Model:
    """A fitted family classifier plus what is needed to read its rules."""

    def __init__(self, train, classifier, method, raw):
        self.pre = Preprocessor().fit(train) if raw else None
        X, _, families = split_xy(self.frame(train))
        self.codes = pd.Categorical(families, categories=FAMILIES).codes
        if (self.codes < 0).any():
            raise ValueError(f"unknown family labels: {sorted(set(families) - set(FAMILIES))}")
        self.features = list(X.columns)
        self.pipe = make_pipeline(classifier, method)
        self.pipe.fit(X, self.codes, **get_fit_params(method, self.codes))
        self.scaler = self.pipe.named_steps["scale"]
        self.clf = self.pipe.named_steps["clf"]
        self.X_scaled = self.scaled(train, transformed=X)   # real (non-resampled) training rows

    def frame(self, df):
        return self.pre.transform(df) if self.pre is not None else df

    def scaled(self, df, transformed=None):
        X = transformed if transformed is not None else split_xy(self.frame(df))[0]
        return self.scaler.transform(X)

    def predict(self, df):
        """Return (attack score = 1 - P(Normal), argmax family name) for each row."""
        X = split_xy(self.frame(df))[0]
        proba = self.pipe.predict_proba(X)
        normal = list(self.pipe.classes_).index(FAMILIES.index("Normal"))
        names = np.asarray(FAMILIES)[self.pipe.classes_[proba.argmax(1)]]
        return 1 - proba[:, normal], names


def pick_threshold(train, classifier, method, raw):
    """Out-of-fold threshold on 1 - P(Normal), using the training partition only."""
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    truth, scores = [], []
    for tr, va in folds.split(train, train.attack_cat):
        model = Model(train.iloc[tr], classifier, method, raw)
        scores.extend(model.predict(train.iloc[va])[0])
        truth.extend(train.label.iloc[va])
    return choose_threshold(truth, scores)


# Reading trees
def original_value(model, feature, value):
    """Undo the pipeline scaler (and log1p) so a threshold is in original units."""
    i = model.features.index(feature)
    value = value * model.scaler.scale_[i] + model.scaler.mean_[i]
    return float(np.expm1(value)) if feature in LOG_COLUMNS else float(value)


def path_conditions(tree, leaf):
    """(feature index, went_right, threshold) for every split from the root to `leaf`."""
    t = tree.tree_
    parent = {}
    for node in range(t.node_count):
        if t.children_left[node] != -1:
            parent[t.children_left[node]] = (node, False)
            parent[t.children_right[node]] = (node, True)
    steps = []
    while leaf in parent:
        node, right = parent[leaf]
        steps.append((t.feature[node], right, t.threshold[node]))
        leaf = node
    return steps[::-1]


def describe_rule(model, steps):
    """Merge a path's splits into readable conditions in original units."""
    lower, upper, is_cat, not_cat = {}, {}, {}, {}
    for index, right, threshold in steps:
        feature = model.features[index]
        if feature.startswith(ONE_HOT_PREFIXES):
            column, category = feature.split("_", 1)
            (is_cat if right else not_cat).setdefault(column, set()).add(category)
            continue
        value = original_value(model, feature, threshold)
        if right:
            lower[feature] = max(lower.get(feature, -np.inf), value)
        else:
            upper[feature] = min(upper.get(feature, np.inf), value)
    parts = []
    for feature in dict.fromkeys(model.features[s[0]] for s in steps):
        if feature in lower and feature in upper:
            parts.append(f"{lower[feature]:.4g} < {feature} <= {upper[feature]:.4g}")
        elif feature in lower:
            parts.append(f"{feature} > {lower[feature]:.4g}")
        elif feature in upper:
            parts.append(f"{feature} <= {upper[feature]:.4g}")
    for column in CATEGORICAL_COLUMNS:
        if column in is_cat:
            parts.append(f"{column} = {'/'.join(sorted(is_cat[column]))}")
        elif column in not_cat:
            parts.append(f"{column} not in {{{', '.join(sorted(not_cat[column]))}}}")
    return " AND ".join(parts) if parts else "(all records)"


def top_of_tree(model, tree, depth=TOP_LEVELS):
    """Text outline of the first `depth` levels, thresholds in original units."""
    t, lines = tree.tree_, []

    def walk(node, level):
        if t.children_left[node] == -1 or level == depth:
            label = FAMILIES[int(tree.classes_[t.value[node, 0].argmax()])]
            lines.append("|   " * level + f"-> {label}  ({t.n_node_samples[node]:,} training rows)")
            return
        feature = model.features[t.feature[node]]
        if feature.startswith(ONE_HOT_PREFIXES):
            column, category = feature.split("_", 1)
            left, right = f"{column} != {category}", f"{column} = {category}"
        else:
            value = original_value(model, feature, t.threshold[node])
            left, right = f"{feature} <= {value:.4g}", f"{feature} > {value:.4g}"
        lines.append("|   " * level + left)
        walk(t.children_left[node], level + 1)
        lines.append("|   " * level + right)
        walk(t.children_right[node], level + 1)

    walk(0, 0)
    return "\n".join(lines)


def leaf_table(tree, real_X, real_codes):
    """One row per leaf: predicted family code and real training records of that family in it."""
    t = tree.tree_
    leaves = np.flatnonzero(t.children_left == -1)
    predicted = tree.classes_[t.value[leaves, 0].argmax(1)].astype(int)
    counts = np.zeros((t.node_count, len(FAMILIES)), dtype=int)
    np.add.at(counts, (tree.apply(real_X), real_codes), 1)
    return pd.DataFrame({"leaf": leaves, "predicted": predicted,
                         "real_support": counts[leaves, predicted],
                         "real_total": counts[leaves].sum(1),
                         "fit_samples": t.n_node_samples[leaves]})


def boundary_stats(model, test_X, test_codes):
    """Per family: leaves predicting it, how many have no real record of it, and
    which share of each family's test records lands in such synthetic-only leaves.

    For the forest every number is averaged over its trees.
    """
    trees = model.clf.estimators_ if hasattr(model.clf, "estimators_") else [model.clf]
    leaves = np.zeros(len(FAMILIES))
    synthetic = np.zeros(len(FAMILIES))
    in_synthetic = np.zeros(len(FAMILIES))
    test_support = np.bincount(test_codes, minlength=len(FAMILIES))
    for tree in trees:
        table = leaf_table(tree, model.X_scaled, model.codes)
        only = table[table.real_support == 0]
        leaves += np.bincount(table.predicted, minlength=len(FAMILIES))
        synthetic += np.bincount(only.predicted, minlength=len(FAMILIES))
        hit = np.isin(tree.apply(test_X), only.leaf)
        in_synthetic += np.bincount(test_codes[hit], minlength=len(FAMILIES))
    n = len(trees)
    return pd.DataFrame({"family": FAMILIES, "leaves": leaves / n,
                         "synthetic_only_leaves": synthetic / n,
                         "test_share_in_synthetic_leaves":
                             np.divide(in_synthetic / n, test_support, out=np.full(len(FAMILIES), np.nan),
                                       where=test_support > 0)})


def surrogate(model):
    """Small tree imitating the forest on the real training rows.

    Grown best-first to SURROGATE_LEAVES leaves without class weights. Balanced
    weights were tried first: they gave the rare families more leaves but cut
    test agreement with the forest from 0.85 to 0.73 (no balancing, 256 leaves),
    so fidelity is instead reported per family (`surrogate_fidelity` in
    e5_families.csv).
    """
    labels = model.clf.predict(model.X_scaled)
    return DecisionTreeClassifier(max_leaf_nodes=SURROGATE_LEAVES,
                                  random_state=SEED).fit(model.X_scaled, labels)


def family_fidelity(readable, forest, test_X):
    """Per family: share of test rows the forest assigns to it that the surrogate also does."""
    want, got = forest.predict(test_X).astype(int), readable.predict(test_X).astype(int)
    hits = np.bincount(want[want == got], minlength=len(FAMILIES))
    total = np.bincount(want, minlength=len(FAMILIES))
    return np.divide(hits, total, out=np.full(len(FAMILIES), np.nan), where=total > 0)


def family_rules(model, tree, test_X, test_codes, classifier, method):
    """Top leaf rules for each RULE_FAMILY, ranked by real training support (no test labels)."""
    table = leaf_table(tree, model.X_scaled, model.codes)
    test_leaf = tree.apply(test_X)
    normal = FAMILIES.index("Normal")
    rows = []
    for family in RULE_FAMILIES:
        code = FAMILIES.index(family)
        mine = table[table.predicted == code].sort_values(["real_support", "fit_samples"],
                                                         ascending=False).head(TOP_RULES)
        for rank, leaf in enumerate(mine.itertuples(), 1):
            here = test_leaf == leaf.leaf
            rows.append({"classifier": classifier, "method": method, "family": family, "rank": rank,
                         "rule": describe_rule(model, path_conditions(tree, leaf.leaf)),
                         "real_train_support": int(leaf.real_support),
                         "real_train_rows_in_leaf": int(leaf.real_total),
                         "fit_rows_in_leaf": int(leaf.fit_samples),
                         "test_family_rows": int((here & (test_codes == code)).sum()),
                         "test_normal_rows": int((here & (test_codes == normal)).sum())})
    return rows


def importances(model, classifier, method):
    """MDI importance per feature; for RF also the share of trees that split on it first."""
    out = pd.DataFrame({"classifier": classifier, "method": method, "feature": model.features,
                        "importance": model.clf.feature_importances_})
    trees = model.clf.estimators_ if hasattr(model.clf, "estimators_") else [model.clf]
    roots = pd.Series([model.features[t.tree_.feature[0]] for t in trees if t.tree_.node_count > 1])
    out["root_split_share"] = out.feature.map(roots.value_counts(normalize=True)).fillna(0.0)
    out["rank"] = out.importance.rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank")


# One configuration
def run_config(train, test, classifier, method, raw):
    """Fit one configuration and return (summary row, family rows, importance rows, rules, text)."""
    start = time.monotonic()
    threshold = pick_threshold(train, classifier, method, raw)
    model = Model(train, classifier, method, raw)
    scores, names = model.predict(test)
    test_X = model.scaled(test)
    test_codes = pd.Categorical(test.attack_cat, categories=FAMILIES).codes

    binary = evaluate_predictions(test.label.to_numpy(), (scores >= threshold).astype(int),
                                  y_score=scores, family_true=test.attack_cat.to_numpy())
    multi = evaluate_predictions(test.attack_cat.to_numpy(), names, task="multiclass")

    families = boundary_stats(model, test_X, test_codes).set_index("family")
    families["detection_recall"] = binary.per_family.detection_recall
    for column in ("classification_precision", "classification_recall", "classification_f1"):
        families[column] = multi.per_family[column]
    families.loc["Normal", "classification_precision"] = multi.per_class.loc["Normal", "precision"]
    families.loc["Normal", "classification_recall"] = multi.per_class.loc["Normal", "recall"]
    families["test_support"] = np.bincount(test_codes, minlength=len(FAMILIES))

    if hasattr(model.clf, "estimators_"):
        readable = surrogate(model)
        fidelity = float((readable.predict(test_X) == model.clf.predict(test_X)).mean())
        families["surrogate_fidelity"] = family_fidelity(readable, model.clf, test_X)
        depths = [t.get_depth() for t in model.clf.estimators_]
        n_leaves = [t.get_n_leaves() for t in model.clf.estimators_]
    else:
        readable, fidelity = model.clf, 1.0
        families["surrogate_fidelity"] = 1.0
        depths, n_leaves = [model.clf.get_depth()], [model.clf.get_n_leaves()]
    rare_fidelity = float(families.loc[RARE_FAMILIES, "surrogate_fidelity"].mean())
    per_family_fidelity = families.surrogate_fidelity.copy()
    families = families.reset_index().assign(classifier=classifier, method=method)
    rules = family_rules(model, readable, test_X, test_codes, classifier, method)

    rare = binary.per_family.loc[RARE_FAMILIES]
    summary = {"classifier": classifier, "method": method, "threshold": threshold,
               "false_positives_per_100k_normal": binary.summary["false_positives_per_100k_normal"],
               "macro_f1": binary.summary["macro_f1"], "roc_auc": binary.summary["roc_auc"],
               "rare_detection_recall": rare.detected_as_attack.sum() / rare.support.sum(),
               "rare_classification_recall": multi.per_family.loc[RARE_FAMILIES, "classification_recall"].mean(),
               "rare_classification_precision": multi.per_family.loc[RARE_FAMILIES, "classification_precision"].mean(),
               "mean_leaves_per_tree": float(np.mean(n_leaves)), "mean_depth": float(np.mean(depths)),
               "rules_from": "surrogate" if readable is not model.clf else "model",
               "surrogate_fidelity": fidelity,
               "surrogate_rare_fidelity": rare_fidelity,
               "seconds": round(time.monotonic() - start, 1)}

    title = f"{classifier}, {method}"
    source = (f"surrogate tree ({SURROGATE_LEAVES} leaves) fitted to the forest's predictions; "
              f"test agreement with the forest {fidelity:.3f}") if readable is not model.clf else "the model itself"
    text = [f"E5 rules: {title}",
            f"threshold on 1 - P(Normal): {threshold:.4f}; false alarms per 100k normal: "
            f"{summary['false_positives_per_100k_normal']:,.0f}; rare detection recall: "
            f"{summary['rare_detection_recall']:.3f}",
            f"rules read from: {source}", "",
            f"Top {TOP_LEVELS} levels (thresholds in original units):", top_of_tree(model, readable), ""]
    for family in RULE_FAMILIES:
        note = (f" (surrogate agrees with the forest on {per_family_fidelity[family]:.2f} of test rows "
                f"the forest assigns to {family})") if readable is not model.clf else ""
        text.append(f"{family}: top leaf rules by real training support{note}")
        mine = [r for r in rules if r["family"] == family]
        if not mine:
            text.append("    (no leaf predicts this family)")
        for r in mine:
            text.append(f"  {r['rank']}. IF {r['rule']}")
            text.append(f"     real {family} train rows {r['real_train_support']:,} of "
                        f"{r['real_train_rows_in_leaf']:,} real rows; fitted on {r['fit_rows_in_leaf']:,} rows; "
                        f"test: {r['test_family_rows']:,} {family}, {r['test_normal_rows']:,} Normal")
        text.append("")
    return summary, families, importances(model, classifier, method), rules, "\n".join(text)


# Comparison with no balancing
def importance_shift(imp, top=10):
    """Per configuration: Spearman rank correlation and top-k overlap with `none`."""
    rows = []
    for (classifier, method), part in imp.groupby(KEY, sort=False):
        base = imp[(imp.classifier == classifier) & (imp.method == "none")]
        if base.empty:
            continue
        joined = part.merge(base, on="feature", suffixes=("", "_none"))
        top_now = set(part.nsmallest(top, "rank").feature)
        top_base = set(base.nsmallest(top, "rank").feature)
        change = (joined.importance - joined.importance_none)
        rows.append({"classifier": classifier, "method": method,
                     "spearman_vs_none": joined.importance.corr(joined.importance_none, method="spearman"),
                     f"top{top}_overlap_vs_none": len(top_now & top_base),
                     "top_feature": part.nsmallest(1, "rank").feature.iloc[0],
                     "largest_gain": joined.feature[change.idxmax()],
                     "largest_gain_change": change.max(),
                     "largest_loss": joined.feature[change.idxmin()],
                     "largest_loss_change": change.min()})
    return pd.DataFrame(rows)


def plot(imp, fig_dir, top=6):
    """Heatmap per classifier: importance of each method's top features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    for classifier, part in imp.groupby("classifier"):
        methods = [m for m in METHODS if m in set(part.method)]
        keep = list(dict.fromkeys(f for m in methods
                                  for f in part[part.method == m].nsmallest(top, "rank").feature))
        table = part.pivot(index="feature", columns="method", values="importance").reindex(
            index=keep, columns=methods)
        fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 3, 0.38 * len(keep) + 1.6))
        image = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, aspect="auto")
        ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
        ax.set_yticks(range(len(keep)), keep)
        vmax = np.nanmax(table.to_numpy())
        for i in range(len(keep)):
            for j in range(len(methods)):
                value = table.iat[i, j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if value > 0.6 * vmax else "black")
        ax.set(xlabel="Balancing method", title=f"E5 feature importance (MDI): {classifier}")
        fig.colorbar(image, ax=ax, label="Impurity importance")
        fig.tight_layout()
        fig.savefig(fig_dir / f"e5_importance_{classifier}.png", dpi=150)
        plt.close(fig)


def plot_synthetic(families, fig_dir):
    """Share of each family's leaves that hold no real training record of it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = families[families.family != "Normal"].assign(
        share=lambda d: np.divide(d.synthetic_only_leaves, d.leaves,
                                  out=np.zeros(len(d)), where=d.leaves > 0))
    classifiers = [c for c in CLASSIFIERS if c in set(families.classifier)]
    fig, axes = plt.subplots(1, len(classifiers), figsize=(7.0 * len(classifiers), 4.6), squeeze=False)
    fig.subplots_adjust(wspace=0.45)   # room for the second panel's family labels
    for ax, classifier in zip(axes[0], classifiers):
        part = families[families.classifier == classifier]
        methods = [m for m in METHODS if m in set(part.method)]
        table = part.pivot(index="family", columns="method", values="share").reindex(
            index=ATTACK_FAMILIES, columns=methods)
        image = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
        ax.set_yticks(range(len(ATTACK_FAMILIES)), ATTACK_FAMILIES)
        for i in range(len(ATTACK_FAMILIES)):
            for j in range(len(methods)):
                value = table.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if value > 0.6 else "black")
        ax.set(xlabel="Balancing method", title=classifier)
    axes[0][0].set_ylabel("Family predicted by the leaf")
    fig.suptitle("E5: share of a family's leaves with no real training record of that family")
    fig.colorbar(image, ax=axes[0].tolist(), label="Synthetic-only share of leaves")
    fig.savefig(fig_dir / "e5_synthetic_leaves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Driver
def load(args):
    if args.raw:
        paths = [Path(TRAIN_PATH), Path(TEST_PATH)]
    else:
        paths = [Path(args.processed_dir) / f"{s}_processed.csv" for s in ("train", "test")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
    if args.raw:   # same data handling as E2/E4: exact duplicates dropped
        return [load_raw(p).drop_duplicates().reset_index(drop=True) for p in paths]
    return [pd.read_csv(p) for p in paths]


def run(args):
    out = Path(args.out_dir)
    (out / "e5_rules").mkdir(parents=True, exist_ok=True)
    paths = {name: out / f"e5_{name}.csv" for name in ("summary", "families", "importances", "rules")}
    train, test = load(args)

    # Resume: configurations are complete once their summary row exists. Drop any
    # partial rows that an interrupted run left in the other tables.
    done = set()
    if paths["summary"].exists():
        done = set(pd.read_csv(paths["summary"])[KEY].itertuples(index=False, name=None))
    for name in ("families", "importances", "rules"):
        if paths[name].exists():
            table = pd.read_csv(paths[name])
            keep = [k in done for k in table[KEY].itertuples(index=False, name=None)]
            table[keep].to_csv(paths[name], index=False)

    for classifier in args.classifiers:
        for method in args.methods:
            if (classifier, method) in done:
                continue
            summary, families, imp, rules, text = run_config(train, test, classifier, method, args.raw)
            (out / "e5_rules" / f"{classifier}_{method}.txt").write_text(text + "\n", encoding="utf-8")
            for name, table in (("families", families), ("importances", imp),
                                ("rules", pd.DataFrame(rules)), ("summary", pd.DataFrame([summary]))):
                if table.empty:
                    continue
                table.to_csv(paths[name], mode="a", index=False,
                             header=not paths[name].exists() or paths[name].stat().st_size == 0)
            print(f"{classifier} {method}: {summary['seconds']:.0f}s", flush=True)

    imp = pd.read_csv(paths["importances"])
    importance_shift(imp).to_csv(out / "e5_importance_shift.csv", index=False)
    plot(imp, Path(args.fig_dir))
    plot_synthetic(pd.read_csv(paths["families"]), Path(args.fig_dir))
    print(f"Saved E5 tables to {out}/ and figures to {args.fig_dir}/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--raw", dest="raw", action="store_true", default=True,
                      help="default: official CSVs in data/, preprocessing fitted per training set")
    mode.add_argument("--processed", dest="raw", action="store_false",
                      help="quick mode: reuse processed/ CSVs (preprocessing fitted on full train)")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
