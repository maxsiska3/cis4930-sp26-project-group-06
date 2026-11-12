# cis4930-sp26-project-group-06
CIS 4930 Data Mining Project: Investigating whether fixing class imbalance improves rare and unseen attack detection in network intrusion systems.

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
