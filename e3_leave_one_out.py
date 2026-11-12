#!/usr/bin/env python3
"""
e3_leave_one_out.py - E3 leave-one-attack-out (objective O3).

For every attack family F, every balancing method and every classifier:
  * drop all F records from the official training partition
  * train a family classifier (Normal + the eight remaining families) on
    what is left, balancing across those families
  * score the full official test partition, where F is now an unseen attack;
    a record counts as flagged if it is predicted as ANY attack family

The key number is unseen_recall: the share of F's test records still flagged
as attacks. seen_recall (the other eight families) and the usual binary
metrics are kept so a high unseen_recall bought with false alarms is visible.

Each run is appended to results/e3_leave_one_out.csv as soon as it finishes,
and runs already in that file are skipped, so the script can be stopped and
restarted. Once every run is done, rerunning only redraws the figures.

Outputs:
    results/e3_leave_one_out.csv     one row per (held_out, classifier, method)
    figures/e3_unseen_recall.png     heatmap: held-out family x method
    figures/e3_tradeoff.png          unseen recall vs false alarms per method

Usage (run preprocess.py first):
    python e3_leave_one_out.py
    python e3_leave_one_out.py --classifiers xgboost random_forest
    python e3_leave_one_out.py --methods none smote --families Worms
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # write PNGs without needing a display
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import ATTACK_FAMILIES, CLASSIFIERS, IMBALANCE_METHODS, get_fit_params, make_pipeline
from evaluate import evaluate_predictions
from preprocess import load_processed

OUT_PATH = Path("results/e3_leave_one_out.csv")
FIG_DIR = Path("figures")
KEY = ["held_out", "classifier", "method"]   # columns that identify one run
# ponytail: focal_loss is skipped until config.get_resampler implements it
METHODS = [m for m in IMBALANCE_METHODS if m != "focal_loss"]


def run_one(family, classifier, method, X_tr, yb_tr, yf_tr, X_te, yb_te, yf_te):
	"""Train without `family`, score the full test set, return one result row."""
	# Withhold the family: the model never sees a single record of it
	keep = (yf_tr != family).to_numpy()

	# Train on family labels so balancing acts on the family imbalance
	# (127 Worms vs ~52k Normal); normal-vs-attack alone is already ~50/50,
	# so balancing would change almost nothing (and ADASYN crashes on it).
	# Integer codes because XGBoost only accepts 0..n-1 class labels.
	codes, names = pd.factorize(yf_tr[keep], sort=True)
	X, y = X_tr[keep], codes

	# Shared pipeline from config.py: scaler -> resampler -> classifier.
	# Resampling happens inside fit, so only training data is resampled.
	start = time.time()
	pipe = make_pipeline(classifier, method)
	pipe.fit(X, y, **get_fit_params(method, y))
	fit_seconds = time.time() - start

	# Turn family predictions into attack detection:
	#   alert = predicted as ANY attack family (a Worms record called
	#           Shellcode still counts as caught)
	#   score = 1 - P(Normal), used for the ranking metrics (PR/ROC AUC)
	normal = list(names).index("Normal")
	proba = pipe.predict_proba(X_te)
	score = 1 - proba[:, list(pipe.classes_).index(normal)]
	alert = (pipe.predict(X_te) != normal).astype(int)

	# Shared scorer from evaluate.py, in binary mode, with family labels so it
	# also reports per-family detection recall
	result = evaluate_predictions(yb_te, alert, y_score=score, family_true=yf_te)

	fam = result.per_family
	seen = fam.drop(index=family)        # the eight families it did train on
	s = result.summary
	return {
		"held_out": family,
		"classifier": classifier,
		"method": method,
		"unseen_support": int(fam.loc[family, "support"]),
		"unseen_detected": int(fam.loc[family, "detected_as_attack"]),
		"unseen_recall": fam.loc[family, "detection_recall"],       # headline metric
		"unseen_mean_score": float(score[(yf_te == family).to_numpy()].mean()),
		# pooled over records, so big families weigh more than small ones
		"seen_recall": seen["detected_as_attack"].sum() / seen["support"].sum(),
		"macro_f1": s["macro_f1"],
		"pr_auc": s["pr_auc"],
		"roc_auc": s["roc_auc"],
		"false_positives_per_100k_normal": s["false_positives_per_100k_normal"],
		"train_rows": int(keep.sum()),
		"fit_seconds": round(fit_seconds, 1),
	}


def plot(results):
	"""Draw the two write-up figures from the finished results table."""
	FIG_DIR.mkdir(exist_ok=True)
	methods = [m for m in METHODS if m in results["method"].unique()]
	families = [f for f in ATTACK_FAMILIES if f in results["held_out"].unique()]

	# Figure 1: heatmap of unseen recall (mean over classifiers), plus a
	# bottom row with the mean over families
	t = results.pivot_table(index="held_out", columns="method", values="unseen_recall")
	t = t.reindex(index=families, columns=methods)
	t.loc["Mean"] = t.mean()
	blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#2a78d6", "#0d366b"])
	fig, ax = plt.subplots(figsize=(7.5, 5))
	im = ax.imshow(t.to_numpy(), cmap=blues, vmin=0.3, vmax=1.0, aspect="auto")
	ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
	ax.set_yticks(range(len(t)), t.index)
	ax.axhline(len(families) - 0.5, color="white", linewidth=3)   # separate the Mean row
	for i in range(len(t)):
		for j in range(len(methods)):
			v = t.iat[i, j]
			# white text on dark cells, dark text on light cells
			ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
					color="white" if v > 0.7 else "#0b0b0b")
	ax.set_xlabel("Balancing method")
	ax.set_ylabel("Family withheld from training")
	ax.set_title("Share of the unseen family still flagged (mean of 4 classifiers)",
				 loc="left", fontsize=11)
	fig.colorbar(im, ax=ax, fraction=0.04, label="Detection recall")
	fig.savefig(FIG_DIR / "e3_unseen_recall.png", dpi=150, bbox_inches="tight")
	plt.close(fig)

	# Figure 2: the tradeoff. One point per method (mean over families and
	# classifiers): more unseen attacks caught vs more normal traffic flagged.
	m = results.groupby("method")[["unseen_recall", "false_positives_per_100k_normal"]].mean()
	m = m.reindex(methods)
	fig, ax = plt.subplots(figsize=(6.5, 4.2))
	ax.scatter(m["false_positives_per_100k_normal"] / 1000, m["unseen_recall"],
			   s=60, color="#2a78d6", zorder=3)
	for name, row in m.iterrows():
		ax.annotate(name, (row["false_positives_per_100k_normal"] / 1000, row["unseen_recall"]),
					xytext=(6, 4), textcoords="offset points", fontsize=8, color="#0b0b0b")
	ax.grid(color="#e4e3df")
	ax.spines[["top", "right"]].set_visible(False)
	ax.set_xlabel("False alarms per 100 normal connections")
	ax.set_ylabel("Unseen-family recall")
	ax.set_title("Catching more unseen attacks costs false alarms", loc="left", fontsize=11)
	fig.savefig(FIG_DIR / "e3_tradeoff.png", dpi=150, bbox_inches="tight")
	plt.close(fig)


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--classifiers", nargs="+", default=CLASSIFIERS, choices=CLASSIFIERS)
	ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
	ap.add_argument("--families", nargs="+", default=ATTACK_FAMILIES, choices=ATTACK_FAMILIES)
	ap.add_argument("--processed-dir", default="processed")
	ap.add_argument("--out", default=str(OUT_PATH))
	args = ap.parse_args()

	# X_tr, yb_tr, yf_tr, X_te, yb_te, yf_te from preprocess.py's output
	data = load_processed(args.processed_dir)
	out = Path(args.out)
	out.parent.mkdir(parents=True, exist_ok=True)

	# Resume support: skip any run already saved in the output CSV
	done = set()
	if out.exists():
		done = set(map(tuple, pd.read_csv(out)[KEY].itertuples(index=False)))

	todo = [(f, c, m) for c in args.classifiers for m in args.methods for f in args.families
			if (f, c, m) not in done]
	print(f"{len(todo)} runs to do ({len(done)} already in {out})")

	for i, (f, c, m) in enumerate(todo, 1):
		row = run_one(f, c, m, *data)
		# Append right away so a crash or Ctrl-C loses at most one run
		pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
		print(f"[{i}/{len(todo)}] {c:<13} {m:<13} held out {f:<15} "
			  f"unseen recall {row['unseen_recall']:.3f}  seen {row['seen_recall']:.3f}  "
			  f"FP/100k {row['false_positives_per_100k_normal']:.0f}  ({row['fit_seconds']}s)")

	# Summary table + figures over everything in the CSV
	results = pd.read_csv(out)
	table = results.pivot_table(index="held_out", columns="method", values="unseen_recall", aggfunc="mean")
	print("\nUnseen-family detection recall (mean over classifiers):")
	print(table.reindex(index=[f for f in ATTACK_FAMILIES if f in table.index],
						columns=[m for m in METHODS if m in table.columns]).round(3).to_string())
	plot(results)
	print(f"\nFigures written to {FIG_DIR.resolve()}")


if __name__ == "__main__":
	main()
