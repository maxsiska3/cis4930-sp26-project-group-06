"""Shared scoring for binary and multiclass network-intrusion predictions.

Binary labels are 0 (normal) and 1 (attack). Multiclass labels are Normal and the
nine configured attack families. Pass final predictions and optional probability
scores. Threshold selection happens in the experiment's training/validation code,
never here. Inputs are paired by position; pandas Series/DataFrame inputs must
also have identical row indices so reordered rows are caught.

The scorer fits nothing, uses no randomness, and writes no files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import ATTACK_FAMILIES, FALSE_ALARM_SCALE


@dataclass
class EvaluationResult:
    """Metric values, labeled tables, and the conventions used to compute them.

    summary contains scalar metrics and binary attack-detection confusion counts
    for either task. per_class is indexed by the task's target labels. per_family
    contains the nine attack families, even when a family is absent. The labeled
    confusion_matrix describes the requested task, with true labels on rows and
    predicted labels on columns. metadata explains undefined values and scoring
    conventions. Values retain full precision for aggregation.
    """

    summary: dict[str, int | float]
    per_class: pd.DataFrame
    per_family: pd.DataFrame
    confusion_matrix: pd.DataFrame
    metadata: dict[str, Any]


def _vector(values: ArrayLike, name: str, n_samples: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if n_samples is not None and len(array) != n_samples:
        raise ValueError(f"{name} must have {n_samples} rows; got {len(array)}.")
    return array


def _binary_labels(values: ArrayLike, name: str, n_samples: int | None = None) -> np.ndarray:
    array = _vector(values, name, n_samples)
    if array.dtype.kind not in "biuf" or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only numeric 0 (normal) and 1 (attack).")
    return array.astype(np.int64)


def _check_pandas_indices(**inputs: Any) -> None:
    frames = [
        (name, value) for name, value in inputs.items()
        if isinstance(value, (pd.Series, pd.DataFrame))
    ]
    if frames:
        first_name, first = frames[0]
        for name, value in frames[1:]:
            if not first.index.equals(value.index):
                raise ValueError(f"{name} and {first_name} must have identical pandas row indices.")


def _family_names(values: ArrayLike, name: str, n_samples: int | None = None) -> np.ndarray:
    array = _vector(values, name, n_samples)
    if pd.isna(array).any():
        raise ValueError(f"{name} must not contain missing values.")
    # Match the spelling normalization in preprocess.py.
    families = pd.Series(array, dtype="string").str.strip().replace({"Backdoors": "Backdoor"})
    unknown = sorted(set(families) - {"Normal", *ATTACK_FAMILIES})
    if unknown:
        raise ValueError(f"{name} contains unknown families: {unknown}.")
    return families.to_numpy(dtype=str)


def _label_order(values: ArrayLike, name: str) -> list[str]:
    labels = _family_names(values, name).tolist()
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} must contain unique family names.")
    return labels


def _probabilities(values: ArrayLike, n_samples: int, ndim: int) -> np.ndarray:
    scores = np.asarray(values)
    if scores.ndim != ndim:
        raise ValueError(f"y_score must be {ndim}-dimensional; got shape {scores.shape}.")
    if len(scores) != n_samples:
        raise ValueError(f"y_score must have {n_samples} rows; got {len(scores)}.")
    if scores.dtype.kind not in "biuf":
        raise ValueError("y_score must contain numeric probabilities.")
    scores = scores.astype(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("y_score must contain finite probabilities in [0, 1].")
    if ndim == 2 and not np.allclose(scores.sum(axis=1), 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("Each multiclass y_score row must sum to 1.")
    return scores


def _ranking_metrics(
    truth: np.ndarray, scores: np.ndarray | None
) -> tuple[dict[str, float], str | None]:
    if scores is None or np.unique(truth).size != 2:
        reason = (
            "y_score was not provided."
            if scores is None
            else "Both positive and negative true records are required."
        )
        return dict.fromkeys(("pr_auc", "average_precision", "roc_auc"), float("nan")), reason
    precision, recall, _ = precision_recall_curve(truth, scores, pos_label=1)
    return {
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)),
    }, None


def evaluate_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    task: str = "binary",
    y_score: ArrayLike | None = None,
    family_true: ArrayLike | None = None,
    score_labels: ArrayLike | None = None,
    labels: ArrayLike | None = None,
) -> EvaluationResult:
    """Score predictions using the same conventions in every experiment.

    Parameters
    ----------
    y_true, y_pred:
        Nonempty, equally sized 1D arrays. Binary targets are numeric 0/1;
        multiclass targets are canonical family names, including Normal. Decode
        integer-encoded family targets before calling. y_pred contains final
        decisions, including any threshold chosen outside this function.
    task:
        "binary" (default) or "multiclass".
    y_score:
        Optional finite probabilities in [0, 1]. Binary: 1D probabilities for
        class 1. Multiclass: a 2D matrix, rows summing to 1, with columns named
        by score_labels. AUCs use scores independently of y_pred.
    family_true:
        Optional aligned attack_cat values for binary per-family recall.
        Multiclass infers these from y_true; if supplied they must match y_true.
        Family inputs accept surrounding whitespace and Backdoors as an alias
        for Backdoor. Missing values and label disagreements raise.
    score_labels:
        Required with multiclass y_score: unique family names identifying its
        columns in their actual order (usually decoded model.classes_). A score
        DataFrame must have matching column names and aligned row indices.
        Missing class columns are allowed, but their AUCs remain undefined.
    labels:
        Optional complete multiclass target universe, including every observed
        true label, prediction, and supplied score column. Defaults to Normal
        followed by all nine ATTACK_FAMILIES. Use the same labels across paired
        runs. Macro-F1 averages over this fixed list, assigning absent classes
        zero F1. Binary always uses [0, 1]; labels/score_labels are multiclass-only.

    Returns
    -------
    EvaluationResult
        summary includes macro_f1, trapezoidal pr_auc, average_precision (AP),
        roc_auc, and false_positives_per_100k_normal. Binary AUCs use attack as
        positive; multiclass AUCs are the unweighted mean of one-versus-rest
        scores for EVERY declared label, including Normal. If any label's AUC
        is undefined, the corresponding macro AUC is NaN (no class dropping).
        Multiclass per-class AUCs remain available independently.

        per_class columns: precision, recall, f1, support, predicted_count.
        Class metrics use zero_division=0, including absent binary classes.
        Multiclass adds pr_auc, average_precision, roc_auc. per_family includes
        support, detected_as_attack, detection_recall; multiclass adds explicitly
        named classification_precision, classification_recall, classification_f1,
        predicted_count, and per-family AUCs. These are NaN outside labels.
        Binary predictions cannot provide family-classification precision.

        Summary confusion counts ALWAYS describe binary attack detection. For
        multiclass, any prediction other than Normal is an attack alert: Worms
        predicted as Generic is detected but is a family-classification error.
        The returned confusion_matrix describes the requested task instead.

        AUCs are NaN without scores or positive/negative true records. False
        alarms are NaN without true normals. A family's detection recall is NaN
        when absent. Binary without family_true has missing family statistics.
        Each unavailable result has a reason in metadata['unavailable_metrics'].
    """
    if task not in ("binary", "multiclass"):
        raise ValueError("task must be 'binary' or 'multiclass'.")
    _check_pandas_indices(y_true=y_true, y_pred=y_pred, y_score=y_score, family_true=family_true)
    scores = None
    score_order = []
    if task == "binary":
        if labels is not None or score_labels is not None:
            raise ValueError("labels and score_labels are only used with task='multiclass'.")
        target_labels = [0, 1]
        truth = _binary_labels(y_true, "y_true")
        prediction = _binary_labels(y_pred, "y_pred", len(truth))
        families = None
        if family_true is not None:
            families = _family_names(family_true, "family_true", len(truth))
            if not np.array_equal(families != "Normal", truth == 1):
                raise ValueError("family_true must be 'Normal' exactly where y_true is 0.")
        binary_truth, binary_prediction = truth, prediction
        if y_score is not None:
            scores = _probabilities(y_score, len(truth), ndim=1)
    else:
        truth = _family_names(y_true, "y_true")
        prediction = _family_names(y_pred, "y_pred", len(truth))
        target_labels = ["Normal", *ATTACK_FAMILIES] if labels is None else _label_order(labels, "labels")
        outside = sorted((set(truth) | set(prediction)) - set(target_labels))
        if outside:
            raise ValueError(f"labels must include every true and predicted family: {outside}.")
        families = truth
        if family_true is not None:
            supplied_families = _family_names(family_true, "family_true", len(truth))
            if not np.array_equal(supplied_families, truth):
                raise ValueError("family_true must match multiclass y_true exactly.")
        binary_truth = (truth != "Normal").astype(np.int64)
        binary_prediction = (prediction != "Normal").astype(np.int64)
        if y_score is None:
            if score_labels is not None:
                raise ValueError("score_labels requires y_score.")
        else:
            if score_labels is None:
                raise ValueError("score_labels is required with multiclass y_score.")
            score_order = _label_order(score_labels, "score_labels")
            outside = sorted(set(score_order) - set(target_labels))
            if outside:
                raise ValueError(f"score_labels contains families outside labels: {outside}.")
            scores = _probabilities(y_score, len(truth), ndim=2)
            if scores.shape[1] != len(score_order):
                raise ValueError("score_labels must name every y_score column exactly once.")
            if isinstance(y_score, pd.DataFrame):
                columns = _label_order(y_score.columns, "y_score columns")
                if columns != score_order:
                    raise ValueError("y_score DataFrame columns must match score_labels in order.")
            outside = sorted(set(prediction) - set(score_order))
            if outside:
                raise ValueError(f"score_labels must include every predicted family: {outside}.")

    # A declared one-class universe necessarily places every record in its sole
    # cell. Construct it directly to avoid sklearn's single-label warning.
    matrix = (
        np.array([[len(truth)]], dtype=np.int64)
        if len(target_labels) == 1
        else confusion_matrix(truth, prediction, labels=target_labels)
    )
    detection_matrix = confusion_matrix(binary_truth, binary_prediction, labels=[0, 1])
    tn, fp, fn, tp = (int(count) for count in detection_matrix.ravel())
    n_normal, n_attack = tn + fp, fn + tp
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=target_labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": matrix.sum(axis=0),
        },
        index=pd.Index(target_labels, name="label"),
    )

    unavailable: dict[str, str] = {}
    if task == "binary":
        ranking_metrics, reason = _ranking_metrics(binary_truth, scores)
        if reason:
            unavailable.update({name: reason for name in ranking_metrics})
    else:
        # Score each class against all other records. Explicit column mapping
        # avoids depending on alphabetical order or an estimator's encoding.
        columns = {label: i for i, label in enumerate(score_order)}
        for label in target_labels:
            class_scores = None if scores is None or label not in columns else scores[:, columns[label]]
            metrics, reason = _ranking_metrics((truth == label).astype(np.int64), class_scores)
            if scores is not None and label not in columns:
                reason = "No probability column was provided for this class."
            for metric, value in metrics.items():
                per_class.loc[label, metric] = value
                if reason:
                    unavailable[f"per_class.{label}.{metric}"] = reason
        ranking_metrics = {}
        for metric in ("pr_auc", "average_precision", "roc_auc"):
            if per_class[metric].isna().any():
                ranking_metrics[metric] = float("nan")
                unavailable[metric] = (
                    "Undefined for one or more declared labels; see per_class reasons. "
                    "The macro average requires every label."
                )
            else:
                ranking_metrics[metric] = float(per_class[metric].mean())

    false_alarms = float("nan")
    if n_normal:
        false_alarms = float(FALSE_ALARM_SCALE * fp / n_normal)
    else:
        unavailable["false_positives_per_100k_normal"] = "No true normal connections."

    family_rows = []
    if families is None:
        unavailable["per_family"] = "family_true was not provided."
    for family in ATTACK_FAMILIES:
        count = detected = None
        detection_recall = float("nan")
        if families is not None:
            mask = families == family
            count = int(mask.sum())
            detected = int((binary_prediction[mask] == 1).sum())
            if count:
                detection_recall = detected / count
            else:
                unavailable[f"per_family.{family}.detection_recall"] = "No true records of this family."
        family_rows.append((count, detected, detection_recall))
    per_family = pd.DataFrame(
        family_rows,
        columns=["support", "detected_as_attack", "detection_recall"],
        index=pd.Index(ATTACK_FAMILIES, name="family"),
    ).astype({"support": "Int64", "detected_as_attack": "Int64"})
    if task == "multiclass":
        classification = per_class.reindex(ATTACK_FAMILIES).drop(columns="support").rename(
            columns={
                "precision": "classification_precision",
                "recall": "classification_recall",
                "f1": "classification_f1",
            }
        )
        classification["predicted_count"] = classification["predicted_count"].astype("Int64")
        per_family = per_family.join(classification)
        for family in ATTACK_FAMILIES:
            if family not in target_labels:
                unavailable[f"per_family.{family}.classification_metrics"] = "Family is outside labels."

    return EvaluationResult(
        summary={
            "n_samples": len(truth),
            "n_normal": n_normal,
            "n_attack": n_attack,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "macro_f1": float(f1.mean()),
            **ranking_metrics,
            "false_positives_per_100k_normal": false_alarms,
        },
        per_class=per_class,
        per_family=per_family,
        confusion_matrix=pd.DataFrame(
            matrix,
            index=pd.Index(target_labels, name="true_label"),
            columns=pd.Index(target_labels, name="predicted_label"),
        ),
        metadata={
            "task": task,
            "labels": target_labels,
            "positive_label": 1 if task == "binary" else None,
            "family_labels": list(ATTACK_FAMILIES),
            "zero_division": 0,
            "pr_auc_method": "trapezoidal",
            "macro_f1_labels": target_labels,
            "auc_strategy": "attack_vs_normal" if task == "binary" else "one_vs_rest",
            "auc_average": None if task == "binary" else "macro",
            "auc_labels": [1] if task == "binary" else target_labels,
            "score_labels": score_order if task == "multiclass" else None,
            "confusion_counts_task": "binary_detection",
            "false_alarm_scale": FALSE_ALARM_SCALE,
            "unavailable_metrics": unavailable,
        },
    )
