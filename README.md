# cis4930-sp26-project-group-06
CIS 4930 Data Mining Project: Investigating whether fixing class imbalance improves rare and unseen attack detection in network intrusion systems.

## E2: rare-family thresholds

Run `python e2_rare_family.py` with the official raw CSVs in `data/`.
Use `--processed` only for exploratory runs with E3 processed CSVs. See
[E2_WRITEUP.md](E2_WRITEUP.md) for the sampling policy, tiny-class limitations,
validation protocol, commands, and result files. Install `requirements.txt` and
run `python -m unittest -v test_e2 test_evaluate` to verify the implementation.

## Shared scoring

`evaluate.py` requires NumPy, pandas, and scikit-learn. It also requires
imbalanced-learn and XGBoost because it imports shared constants from `config.py`,
which loads those training libraries. Scoring does not require the dataset.

```python
from evaluate import evaluate_predictions

result = evaluate_predictions(
    y_true=[0, 0, 1, 1],
    y_pred=[0, 1, 0, 1],
    task="binary",
    y_score=[0.1, 0.8, 0.4, 0.9],  # probability of attack (class 1)
    family_true=["Normal", "Normal", "Worms", "Shellcode"],
)

print(result.summary)
print(result.per_class)
print(result.per_family)
print(result.confusion_matrix)
```

Pass aligned, one-dimensional arrays. For pandas Series, indices must also
match. For a fitted model, select the `predict_proba` column corresponding to
class `1`; do not pass the whole probability matrix. Set any decision threshold
using training-side validation data before calling the scorer.

- `summary` includes macro-F1 over both binary classes, PR-AUC (trapezoidal),
  average precision (AP), ROC-AUC, confusion counts, and false positives per
  100,000 true normal connections. The example has one false alarm among two
  normal records, so its scaled false-alarm count is 50,000.
- `per_class` provides precision, recall, F1, true support, and predicted counts
  for labels `0` (normal) and `1` (attack), using zero for undefined class ratios.
- `per_family` provides support, detected attack counts, and detection recall for
  all nine attack families. A binary model cannot report family-specific
  classification precision. An absent family's detection recall is `NaN`.
- `y_score` and `family_true` are optional. Missing scores or a single true class
  produce `NaN` AUCs. Missing family labels produce unavailable family counts and
  recalls. With no true normal records, the false-alarm metric is `NaN`.
  `result.metadata["unavailable_metrics"]` explains unavailable results.

For multiclass family classification, pass family names and identify every
probability column explicitly:

```python
result = evaluate_predictions(
    y_true=["Normal", "Worms", "Shellcode"],
    y_pred=["Normal", "Shellcode", "Shellcode"],
    task="multiclass",
    y_score=[
        [0.8, 0.1, 0.1],
        [0.1, 0.3, 0.6],
        [0.1, 0.2, 0.7],
    ],
    score_labels=["Normal", "Worms", "Shellcode"],
    labels=["Normal", "Worms", "Shellcode"],  # this example's target universe
)

print(result.summary)
print(result.per_family[["classification_precision", "classification_recall", "detection_recall"]])
```

For fitted models, `score_labels` must match `predict_proba` column order,
usually `model.classes_`. Decode integer-encoded family names in `y_true`,
`y_pred`, and `score_labels` first, for example with a fitted label encoder's
`inverse_transform`. Each probability row must sum to one. A probability
DataFrame must also have columns matching `score_labels` and aligned row indices.

- Multiclass defaults to all ten project classes: `Normal` followed by
  `config.ATTACK_FAMILIES`. Set `labels` only to declare a smaller complete
  target universe; keep it identical across comparisons. It must include all
  observed true/predicted classes and all supplied score columns.
- Macro-F1 uses the declared class list, assigning absent classes zero F1.
  Consequently, perfect predictions on just three observed classes give 0.3
  macro-F1 under the default ten-class universe, or 1.0 with those three classes
  explicitly declared as the complete universe.
- Multiclass PR-AUC, AP, and ROC-AUC are unweighted macro averages of each
  declared class versus all the other records, including `Normal`. Each class's
  scores appear in `per_class` (and attack-family scores in `per_family`). If any
  declared class lacks positive/negative records or a probability column, its
  AUCs and the corresponding summary AUCs are `NaN`, with reasons in metadata.
  Missing classes are never silently removed from the average or given invented
  probabilities. Other classes' valid AUCs remain available.
- Multiclass `per_family` adds `classification_precision`,
  `classification_recall`, `classification_f1`, `predicted_count`, and AUCs.
  `detection_recall` measures any attack alert: a Worms record predicted as
  Shellcode counts as detected, but incorrectly classified.
- Summary confusion counts and false alarms always describe binary detection:
  `Normal` versus any attack. The returned `confusion_matrix` instead describes
  the requested classification task. In the example, Worms has detection recall
  1.0 and classification recall 0.0. No normal record produces a false alarm.

Run the binary and multiclass scoring checks without training models:

```bash
python -m unittest -v test_evaluate
```

## E1: imbalance benchmark

E1 has two implementation/entry-point files:

- `e1.py`: preprocessing, treatment pipelines, focal-loss classifiers,
  cross-validation, threshold selection, checkpointing, tables and figures.
  Other experiments can import `fit_detector()` or `run_cv()` from this module.
- `run_e1.py`: launches the experiment with command-line options.

`test_e1.py` contains the tests, and `requirements-e1.txt` lists dependencies.
Shared settings and scoring remain in `config.py` and `evaluate.py`.

E1 reads the **raw official training CSV**, not `processed/train_processed.csv`.
`preprocess.py` does not download data, and it does not need to be run before E1.
Obtain `UNSW_NB15_training-set.csv` from the
[official UNSW dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
(the linked SharePoint download may require sign-in), or use the same original
CSV as your teammates. Put it at `data/UNSW_NB15_training-set.csv`, or provide
`--train-path /path/to/UNSW_NB15_training-set.csv`. E1 never opens the official
testing CSV. Do not concatenate the two partitions.

Install E1 dependencies in your project environment:

```bash
python3 -m pip install -r requirements-e1.txt
```

Start with one classifier, one method and one training target:

```bash
python3 run_e1.py --classifiers decision_tree --methods none --tasks multiclass
```

For a smaller pilot, use a stratified training subset and a separate output
directory. The subset size is recorded, so sampled results cannot be silently
mixed with full-data results:

```bash
python3 run_e1.py --sample-size 20000 --classifiers decision_tree --methods none smote --tasks multiclass --out-dir results/e1_pilot
```

Run the complete benchmark or resume interrupted work:

```bash
python3 run_e1.py
python3 run_e1.py --resume
```

The full sweep has 26 valid classifier/method pairs for each training task:
four classifiers times six ordinary treatments, plus focal loss for MLP and
XGBoost. Decision Tree and Random Forest never receive focal loss. Both binary
and multiclass training are enabled by default. Five outer folds and two fits
per fold give 520 fits in total. Runtime is hardware-dependent; the command
prints progress and updates its remaining-time estimate after each outer fold.
Run a pilot before the full sweep. Models use the settings in `config.py`;
optional `--model-params overrides.json` accepts a mapping such as
`{"random_forest": {"n_estimators": 20, "n_jobs": 1}}`, recorded in the manifest.

Protocol:

- Only training data is deduplicated (exact rows, matching the shared default).
  Five folds are stratified by family. `StratifiedGroupKFold` keeps identical raw
  feature vectors in the same partition even if their family labels conflict.
  All methods reuse the same fold memberships and seed 42. A split missing a
  family fails explicitly instead of changing the evaluation population.
- An inner grouped validation fold (approximately 20% of each outer training
  partition) selects the attack threshold maximizing **binary macro-F1**, with
  ties favoring fewer alerts. Its provisional model sees only the inner training
  data. A fresh pipeline is then fitted on the whole outer training partition
  and evaluated on the untouched outer validation fold.
- Medians, protocol grouping, numeric scaling, encoders, class weights and
  resampling are fitted separately for each fit. IDs and both target columns
  are excluded from features. E1 has its own factory in `e1.py` because
  the shared pipeline cannot preserve categorical values for SMOTENC.
- Binary training balances the normal/attack target; multiclass training
  balances individual families. Report these as separate experiments rather
  than pooling their results. Multiclass detection uses `1 - P(Normal)` and the
  selected threshold, then chooses the highest-probability attack family when
  an alert is raised. Default argmax decisions are also scored and saved.
- Every result uses `evaluate.py`. Fold metrics are the primary estimates;
  PR/ROC figures pool out-of-fold scores and therefore need not equal the mean
  fold AUC. No threshold is chosen using an outer validation fold or final test
  labels. Hyperparameter selection using these CV results requires separate
  final evaluation on the still-reserved test partition.

Treatment details that must be reported:

- `smote` is **SMOTENC**, applied to scaled numerics plus original categorical
  fields before final one-hot encoding. `smote_enn` runs SMOTENC, encoding, then
  edited-nearest-neighbor cleaning. Sampling/cleaning that removes an entire
  training class is recorded as a failed fold, not repaired silently.
- `adasyn` runs imbalanced-learn's numerical ADASYN on encoded features and then
  projects each categorical dummy block to its largest component. This is an
  explicitly documented adaptation, not native categorical ADASYN. If the
  library's integer allocation produces zero synthetic records, that no-op is
  recorded. Other sampling failures remain errors. Definitions and mixed-data
  limitations: [imbalanced-learn documentation](https://imbalanced-learn.org/stable/over_sampling.html).
- `focal_loss` uses balanced class weights as alpha and gamma 2. XGBoost uses
  analytic gradients with diagonal Hessians clipped positive. The custom MLP
  uses ReLU, Adam, the configured hidden sizes, a softmax output (including binary
  training), and validation focal loss for early stopping. Its output and
  stopping criterion differ from sklearn's MLP and must be disclosed. Loss:
  [Lin et al., 2017](https://arxiv.org/abs/1708.02002); objective API:
  [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html).

Outputs under `results/e1/`:

- `manifest.json`: dataset fingerprint, source hashes, package versions,
  sampling choices, seeds, model overrides and protocol.
- `split_plan.json`: exact outer and threshold-selection row memberships.
- `fold_results.csv`: one row per training task/classifier/method/outer fold,
  selected threshold, tuned/default metrics, family detection recalls, runtime,
  warnings, sampling counts and explicit success/error status.
- `family_results.csv`: family metrics for tuned and default decisions.
- `family_summary.csv`: per-family mean/std, pooled recall, total support and
  paired recall differences against `none`, identifying the four rare families.
- `summary.csv`: mean/std across complete folds and paired differences against
  `none` for the same classifier, training task and fold IDs. Incomplete or
  undefined comparisons remain unavailable, not silently averaged over fewer
  folds. This is descriptive; E6 significance testing is a separate step.
- `checkpoints/`: immediate per-fold records, family tables and compressed
  out-of-fold predictions, with checksums. Resume validates the manifest and
  artifacts; it reruns failed/missing/corrupt folds. Changed data/code/settings
  require a new output directory. Additional classifier/method selections can
  be added with `--resume` when the core protocol is unchanged.
- `figures/`: pooled out-of-fold detection PR/ROC curves and rare-recall versus
  false-alarm plots for complete runs.

Reuse one method in another experiment (for example E3):

```python
from e1 import fit_detector, run_cv
from preprocess import load_raw
from evaluate import evaluate_predictions

train = load_raw("data/UNSW_NB15_training-set.csv").drop_duplicates()
cv = run_cv(train, classifier="random_forest", method="smote", task="multiclass")
print(cv.folds)

# E3 should filter the training family BEFORE any fitting or threshold selection.
seen_train = train[train.attack_cat != "Worms"]
detector = fit_detector(seen_train, classifier="random_forest", method="smote")
# For a separate E3 test evaluation: prediction = detector.predict(test_frame)
# result = evaluate_predictions(test_frame.label.to_numpy(), prediction.binary,
#     y_score=prediction.attack_score, family_true=test_frame.attack_cat.to_numpy())
```

Meaningful E1 tests use synthetic raw-format data, including all valid treatment
pairs, numerical checks of focal gradients/Hessians, preprocessing isolation,
group separation, threshold selection, paired folds, CLI outputs and resume:

```bash
python3 -m unittest -v test_evaluate test_e1
```
