"""E1 imbalance benchmark: all implementation code in one importable module.

Run the experiment through run_e1.py, for example:
    python run_e1.py --classifiers decision_tree --methods none --tasks multiclass
    python run_e1.py --sample-size 20000
    python run_e1.py --resume

Import fit_detector() to fit one treatment (also reusable by E3 after withholding
an attack family), or run_cv() to evaluate one classifier/method/task. Only the
raw official training CSV is read; the final test partition stays reserved.

The sections below contain focal classifiers, fold-local preprocessing/sampling,
benchmark execution, reports, and command-line/checkpoint support. Shared
config.py, preprocess.py, and evaluate.py supply settings, loading, and scoring.

SMOTE uses SMOTENC before final one-hot encoding. ADASYN's numerical output is
projected to valid categorical blocks, an adaptation recorded in result metadata.
Focal loss is -alpha[y] * (1 - p[y])**gamma * log(p[y]), using balanced alpha and
gamma=2 by default. The MLP uses NumPy Adam; XGBoost uses an analytic custom
objective with clipped diagonal Hessians.

E1 disables internal MLP validation splits: those would occur after resampling.
Ordinary MLP uses training-loss convergence, up to max_iter; focal MLP uses the
configured epoch budget. Thresholds still use a separate raw, grouped validation
partition. The custom focal MLP differs from sklearn's MLP in training and output
implementation, so comparisons do not isolate the loss function alone.

References:
https://imbalanced-learn.org/stable/over_sampling.html
https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html
https://arxiv.org/abs/1708.02002
https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from imblearn.over_sampling import ADASYN, SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_array, check_is_fitted

from config import (
    ATTACK_FAMILIES, BINARY_LABEL, CATEGORICAL_COLUMNS, CLASSIFIERS, FAMILY_LABEL,
    ID_COLUMN, IMBALANCE_METHODS, LOG_COLUMNS, N_FOLDS, PROTO_MIN_COUNT,
    RARE_FAMILIES, SEED, SMOTE_K, TRAIN_PATH, get_classifier, get_fit_params,
)
from evaluate import evaluate_predictions
from preprocess import load_raw


# ============================================================================
# Focal-loss objective and classifiers
# ============================================================================


def focal_objective(y, logits, *, gamma=2.0, class_weights=None):
    """Return per-record losses, logit gradients and diagonal Hessians.

    A vector of logits represents binary sigmoid outputs; a matrix represents
    multiclass softmax outputs. Hessians here are exact and are NOT clipped,
    allowing numerical verification. The XGBoost adapter clips them separately.
    """
    y = np.asarray(y, dtype=int)
    logits = np.asarray(logits, dtype=float)
    if gamma < 0:
        raise ValueError("gamma must be nonnegative.")
    binary = logits.ndim == 1
    probabilities = expit(logits) if binary else softmax(logits, axis=1)
    pt = np.where(y == 1, probabilities, 1 - probabilities) if binary else probabilities[np.arange(len(y)), y]
    pt = np.clip(pt, 1e-12, 1 - 1e-12)
    q = 1 - pt
    weights = np.ones(len(y)) if class_weights is None else np.asarray(class_weights)[y]
    losses = -weights * q**gamma * np.log(pt)
    if gamma == 0:
        factor = np.ones_like(pt)
        derivative = np.zeros_like(pt)
    else:
        factor = q**gamma - gamma * pt * q ** (gamma - 1) * np.log(pt)
        derivative = (
            -2 * gamma * q ** (gamma - 1)
            - gamma * q ** (gamma - 1) * np.log(pt)
        )
        if gamma != 1:
            derivative += gamma * (gamma - 1) * pt * q ** (gamma - 2) * np.log(pt)
    residual = probabilities - y if binary else probabilities - np.eye(logits.shape[1])[y]
    if binary:
        gradient = weights * factor * residual
        hessian = weights * (factor * probabilities * (1 - probabilities) - pt * derivative * residual**2)
    else:
        gradient = (weights * factor)[:, None] * residual
        hessian = weights[:, None] * (
            factor[:, None] * probabilities * (1 - probabilities)
            - (pt * derivative)[:, None] * residual**2
        )
    return losses, gradient, hessian


def _targets(y):
    classes, encoded = np.unique(np.asarray(y), return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Focal loss needs at least two training classes.")
    return classes, encoded


class FocalXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost with a genuine focal objective and explicit margin conversion."""

    def __init__(self, *, gamma=2.0, random_state=SEED, n_estimators=300, n_jobs=-1,
                 tree_method="hist", max_depth=6, learning_rate=0.3):
        self.gamma = gamma
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y):
        from xgboost import XGBClassifier

        X = check_array(X)
        self.classes_, encoded = _targets(y)
        self.n_features_in_ = X.shape[1]
        self.class_weights_ = compute_class_weight(
            "balanced", classes=np.arange(len(self.classes_)), y=encoded
        )

        def objective(labels, margins):
            _, gradient, hessian = focal_objective(
                labels, margins, gamma=self.gamma, class_weights=self.class_weights_
            )
            return gradient, np.maximum(hessian, 1e-6)

        self.model_ = XGBClassifier(
            objective=objective, n_estimators=self.n_estimators, n_jobs=self.n_jobs,
            tree_method=self.tree_method, random_state=self.random_state,
            max_depth=self.max_depth, learning_rate=self.learning_rate,
            base_score=0.5,  # binary logistic margin 0; equal multiclass margins
        )
        self.model_.fit(X, encoded)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        margins = self.model_.predict(check_array(X), output_margin=True)
        if margins.ndim == 1:
            positive = expit(margins)
            return np.column_stack([1 - positive, positive])
        return softmax(margins, axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class FocalMLPClassifier(ClassifierMixin, BaseEstimator):
    """ReLU MLP with focal loss, Adam, L2 regularization and early stopping.

    Hidden-layer sizes and defaults match config.py's MLP. Unlike sklearn's
    binary MLP, the output always has one softmax unit per class. Standalone early
    stopping monitors validation focal loss instead of accuracy. E1's factory
    disables that split and uses max_iter epochs. These implementation differences
    must be reported with the focal-loss treatment.
    """

    def __init__(self, *, hidden_layer_sizes=(64, 32), gamma=2.0, alpha=0.0001,
                 learning_rate_init=0.001, batch_size=200, max_iter=200,
                 early_stopping=True, validation_fraction=0.1, tol=1e-4,
                 n_iter_no_change=10, random_state=SEED):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def _forward(self, X):
        activations = [X]
        for W, b in zip(self.coefs_[:-1], self.intercepts_[:-1]):
            activations.append(np.maximum(0, activations[-1] @ W + b))
        logits = activations[-1] @ self.coefs_[-1] + self.intercepts_[-1]
        return activations, logits

    def fit(self, X, y):
        X = check_array(X)
        self.classes_, encoded = _targets(y)
        if self.max_iter < 1 or self.batch_size < 1 or self.gamma < 0:
            raise ValueError("max_iter/batch_size must be positive and gamma nonnegative.")
        self.n_features_in_ = X.shape[1]
        indices = np.arange(len(encoded))
        if self.early_stopping:
            train, validation = train_test_split(
                indices, test_size=self.validation_fraction, stratify=encoded,
                random_state=self.random_state,
            )
        else:
            train, validation = indices, indices
        self.class_weights_ = compute_class_weight(
            "balanced", classes=np.arange(len(self.classes_)), y=encoded[train]
        )
        rng = np.random.default_rng(self.random_state)
        sizes = (X.shape[1], *self.hidden_layer_sizes, len(self.classes_))
        self.coefs_, self.intercepts_ = [], []
        for before, after in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6 / (before + after))
            self.coefs_.append(rng.uniform(-bound, bound, (before, after)))
            self.intercepts_.append(np.zeros(after))
        parameters = self.coefs_ + self.intercepts_
        first_moments = [np.zeros_like(p) for p in parameters]
        second_moments = [np.zeros_like(p) for p in parameters]
        self.loss_curve_ = []
        best_loss, best_parameters, stale, step = np.inf, None, 0, 0
        for epoch in range(self.max_iter):
            order = rng.permutation(train)
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                activations, logits = self._forward(X[batch])
                _, delta, _ = focal_objective(
                    encoded[batch], logits, gamma=self.gamma, class_weights=self.class_weights_
                )
                delta /= len(batch)
                grad_W, grad_b = [], []
                for layer in range(len(self.coefs_) - 1, -1, -1):
                    grad_W.append(activations[layer].T @ delta + self.alpha * self.coefs_[layer] / len(train))
                    grad_b.append(delta.sum(axis=0))
                    if layer:
                        delta = (delta @ self.coefs_[layer].T) * (activations[layer] > 0)
                gradients = list(reversed(grad_W)) + list(reversed(grad_b))
                step += 1
                for p, g, m, v in zip(parameters, gradients, first_moments, second_moments):
                    m *= 0.9
                    m += 0.1 * g
                    v *= 0.999
                    v += 0.001 * g * g
                    p -= self.learning_rate_init * (m / (1 - 0.9**step)) / (
                        np.sqrt(v / (1 - 0.999**step)) + 1e-8
                    )
            _, logits = self._forward(X[validation])
            losses, _, _ = focal_objective(
                encoded[validation], logits, gamma=self.gamma, class_weights=self.class_weights_
            )
            loss = float(losses.mean())
            if not np.isfinite(loss):
                raise ValueError("Focal MLP training produced a non-finite loss.")
            self.loss_curve_.append(loss)
            if loss < best_loss - self.tol:
                best_loss, stale = loss, 0
                best_parameters = [p.copy() for p in parameters]
            else:
                stale += 1
            if self.early_stopping and stale >= self.n_iter_no_change:
                break
        else:
            if self.early_stopping:
                warnings.warn("Focal MLP reached max_iter before early stopping.", ConvergenceWarning)
        self.n_iter_ = epoch + 1
        self.loss_ = self.loss_curve_[-1]
        if self.early_stopping and best_parameters is not None:
            for p, best in zip(parameters, best_parameters):
                p[:] = best
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "coefs_")
        return softmax(self._forward(check_array(X))[1], axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ============================================================================
# Fold-local preprocessing and treatment pipelines
# ============================================================================


IMPLEMENTATIONS = {
    "none": "no balancing",
    "class_weights": "balanced sample weights from this fit's targets",
    "undersample": "RandomUnderSampler",
    "smote": "SMOTENC before one-hot encoding",
    "adasyn": "ADASYN on one-hot features, then categorical block projection",
    "smote_enn": "SMOTENC, one-hot encoding, then EditedNearestNeighbours(all)",
    "focal_loss": "balanced-alpha focal loss, gamma=2",
}


class FoldPreprocessor(TransformerMixin, BaseEstimator):
    """Fit medians, protocol grouping, log transforms and numeric scaling locally.

    Output remains a DataFrame containing scaled numerics and categorical
    strings. Encoding is a later pipeline step. Infinities count as missing;
    entirely missing numeric training columns use zero as their fill value.
    """

    def __init__(self, min_proto_count=PROTO_MIN_COUNT):
        self.min_proto_count = min_proto_count

    @staticmethod
    def _categories(X):
        categories = X[CATEGORICAL_COLUMNS].astype("string").fillna("__missing__")
        for column in CATEGORICAL_COLUMNS:
            categories[column] = categories[column].str.strip().str.lower()
        categories["service"] = categories["service"].replace({"-": "no_service"})
        return categories

    def _numbers(self, X):
        numeric = X[self.numeric_columns_].apply(pd.to_numeric, errors="raise")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        return numeric

    def _logged(self, numeric):
        numeric = numeric.fillna(self.medians_).copy()
        for column in self.log_columns_:
            numeric[column] = np.log1p(numeric[column].clip(lower=0))
        return numeric

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FoldPreprocessor requires raw features in a DataFrame.")
        if {BINARY_LABEL, FAMILY_LABEL} & set(X.columns):
            raise ValueError("Target columns must not be passed as model features.")
        missing = set(CATEGORICAL_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Raw categorical columns are required: {sorted(missing)}.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.n_fit_rows_ = len(X)
        self.numeric_columns_ = [c for c in X.columns if c not in CATEGORICAL_COLUMNS]
        if not self.numeric_columns_:
            raise ValueError("At least one numeric feature is required.")
        self.log_columns_ = [c for c in LOG_COLUMNS if c in self.numeric_columns_]
        self.medians_ = self._numbers(X).median().fillna(0)
        self.scaler_ = StandardScaler().fit(self._logged(self._numbers(X)))
        counts = self._categories(X)["proto"].value_counts()
        self.keep_protocols_ = set(counts[counts >= self.min_proto_count].index)
        return self

    def transform(self, X):
        check_is_fitted(self, "scaler_")
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Feature columns and their order must match the fit data.")
        numeric = pd.DataFrame(
            self.scaler_.transform(self._logged(self._numbers(X))),
            columns=self.numeric_columns_, index=X.index,
        )
        categories = self._categories(X)
        categories["proto"] = categories["proto"].where(
            categories["proto"].isin(self.keep_protocols_), "other"
        )
        return pd.concat([numeric, categories], axis=1)


class ProjectedADASYN(BaseEstimator):
    """Public imblearn ADASYN followed by one-hot categorical block projection.

    ADASYN's neighbor difficulty and interpolation are unchanged. Projection
    chooses the largest dummy value within each original categorical feature;
    this preserves valid categories but is an adaptation of numeric ADASYN.
    If the library's rounded allocation creates zero samples, record that fact
    and return the unmodified data instead of substituting a different sampler.
    """

    def __init__(self, n_neighbors=SMOTE_K, random_state=SEED):
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ProjectedADASYN requires named one-hot columns.")
        self.resampler_ = ADASYN(n_neighbors=self.n_neighbors, random_state=self.random_state)
        self.no_synthetic_samples_ = False
        try:
            sampled, labels = self.resampler_.fit_resample(X, y)
        except ValueError as error:
            if "No samples will be generated" not in str(error):
                raise
            self.no_synthetic_samples_ = True
            sampled, labels = X.copy(), np.asarray(y).copy()
        sampled = pd.DataFrame(sampled, columns=X.columns).copy()
        for column in CATEGORICAL_COLUMNS:
            block = [name for name in X.columns if name.startswith(f"cat__{column}_")]
            if not block:
                raise ValueError(f"Cannot find encoded categorical block {column}.")
            choices = sampled[block].to_numpy().argmax(axis=1)
            sampled.loc[:, block] = np.eye(len(block))[choices]
        self.n_generated_ = len(sampled) - len(X)
        return sampled, labels


class ClassCoverageGuard(BaseEstimator):
    """Fail visibly if sampling removes a training class (including Normal)."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit_resample(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        if not np.array_equal(classes, np.arange(self.n_classes)):
            raise ValueError("Resampling removed a target class; this fit cannot be compared fairly.")
        self.class_counts_ = dict(zip(classes.tolist(), counts.tolist()))
        return X, y


def make_training_pipeline(classifier, method, n_classes, *, seed=SEED, model_params=None):
    """Return a pipeline; MLP must not split already-resampled data for validation."""
    if classifier not in CLASSIFIERS or method not in IMBALANCE_METHODS:
        raise ValueError("Unknown classifier or balancing method.")
    if method == "focal_loss" and classifier not in ("mlp", "xgboost"):
        raise ValueError("Focal loss is supported only for MLP and XGBoost.")
    model_params = dict(model_params or {})
    if "random_state" in model_params:
        raise ValueError("Use the shared seed argument instead of overriding random_state.")
    if classifier == "mlp":
        if model_params.get("early_stopping", False):
            raise ValueError(
                "E1 disables MLP early_stopping because its validation split would "
                "happen after preprocessing and resampling. Use max_iter to set the budget."
            )
        model_params["early_stopping"] = False
    if method == "focal_loss":
        model = FocalMLPClassifier() if classifier == "mlp" else FocalXGBClassifier()
    else:
        model = get_classifier(classifier)
    model.set_params(random_state=seed, **model_params)

    raw_sampler = "passthrough"
    if method == "undersample":
        raw_sampler = RandomUnderSampler(random_state=seed)
    elif method in ("smote", "smote_enn"):
        raw_sampler = SMOTENC(
            categorical_features=CATEGORICAL_COLUMNS,
            k_neighbors=SMOTE_K, random_state=seed,
        )
    encoder = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS)],
        remainder="passthrough", sparse_threshold=0,
    ).set_output(transform="pandas")
    return Pipeline([
        ("prepare", FoldPreprocessor()),
        ("sample_raw", raw_sampler),
        ("encode", encoder),
        ("sample_encoded", ProjectedADASYN(random_state=seed) if method == "adasyn" else "passthrough"),
        ("clean", EditedNearestNeighbours(sampling_strategy="all") if method == "smote_enn" else "passthrough"),
        ("coverage", ClassCoverageGuard(n_classes)),
        ("clf", model),
    ])


# ============================================================================
# Training, thresholds, and cross-validation
# ============================================================================


def training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize labels/raw categories without fitting anything."""
    frame = frame.drop(columns=[ID_COLUMN], errors="ignore").copy().reset_index(drop=True)
    required = {BINARY_LABEL, FAMILY_LABEL, *CATEGORICAL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"E1 needs raw training columns {sorted(missing)}; do not pass one-hot processed CSVs.")
    if frame.empty or frame[FAMILY_LABEL].isna().any():
        raise ValueError("Training data must be nonempty and have family labels.")
    frame[FAMILY_LABEL] = frame[FAMILY_LABEL].astype(str).str.strip().replace({"Backdoors": "Backdoor"})
    if set(frame[FAMILY_LABEL]) - {"Normal", *ATTACK_FAMILIES}:
        raise ValueError("Unknown attack family in training data.")
    if not frame[BINARY_LABEL].isin([0, 1]).all():
        raise ValueError("Binary labels must be 0 or 1.")
    frame[BINARY_LABEL] = frame[BINARY_LABEL].astype(int)
    if not np.array_equal(frame[FAMILY_LABEL] != "Normal", frame[BINARY_LABEL] == 1):
        raise ValueError("Binary and family labels disagree.")
    if frame[BINARY_LABEL].nunique() != 2:
        raise ValueError("Training needs normal traffic and at least one attack family.")
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("string").fillna("__missing__").str.strip().str.lower()
    frame["service"] = frame["service"].replace({"-": "no_service"})
    return frame


def features(frame):
    return frame.drop(columns=[BINARY_LABEL, FAMILY_LABEL, ID_COLUMN], errors="ignore")


def feature_groups(frame):
    """Identical raw feature vectors share a split, even if labels conflict."""
    return pd.util.hash_pandas_object(features(frame), index=False).to_numpy()


def dataset_id(frame):
    digest = hashlib.sha256(json.dumps(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def make_folds(frame, n_splits=N_FOLDS, seed=SEED):
    """Five stratified folds by default, grouping identical raw feature rows.

    Group stratification is approximate. Every training and validation part must
    contain every family present in the supplied data; fail rather than silently
    use an invalid split when a family has too few independent feature groups.
    """
    if n_splits < 2 or frame[FAMILY_LABEL].value_counts().min() < n_splits:
        raise ValueError(f"Each family needs at least {n_splits} records for this split.")
    groups = feature_groups(frame)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    expected = set(frame[FAMILY_LABEL])
    splits = list(splitter.split(features(frame), frame[FAMILY_LABEL], groups))
    for train, validation in splits:
        if set(groups[train]) & set(groups[validation]):
            raise AssertionError("Feature groups overlap between training and validation.")
        if set(frame.iloc[train][FAMILY_LABEL]) != expected or set(frame.iloc[validation][FAMILY_LABEL]) != expected:
            raise ValueError("A grouped split is missing a family; more independent records are needed.")
    return splits


def select_threshold(y_true, scores):
    """Maximize binary macro-F1 on inner validation data; ties prefer fewer alerts.

    Evaluate every distinct score plus an all-normal endpoint. No outer fold or
    official test labels enter this function. The vectorized count calculation
    avoids calling sklearn once for each of tens of thousands of thresholds.
    """
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if truth.ndim != 1 or scores.shape != truth.shape or set(truth) != {0, 1}:
        raise ValueError("Threshold selection needs aligned scores and both binary classes.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Threshold scores must be finite probabilities.")
    order = np.argsort(-scores, kind="stable")
    ranked_scores, ranked_truth = scores[order], truth[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores) != 0), len(scores) - 1]
    tp = np.r_[0, np.cumsum(ranked_truth)[ends]]
    fp = np.r_[0, ends + 1 - np.cumsum(ranked_truth)[ends]]
    fn, tn = truth.sum() - tp, (len(truth) - truth.sum()) - fp
    attack_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    normal_f1 = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) != 0)
    values = (attack_f1 + normal_f1) / 2
    thresholds = np.r_[np.nextafter(1.0, np.inf), ranked_scores[ends]]
    best = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))[0]
    return float(thresholds[best]), float(values[best])


@dataclass
class Predictions:
    attack_score: np.ndarray
    binary: np.ndarray
    default_binary: np.ndarray
    probabilities: np.ndarray
    score_labels: np.ndarray
    families: np.ndarray | None
    default_families: np.ndarray | None


@dataclass
class FittedDetector:
    pipeline: Any
    encoder: LabelEncoder
    task: str
    threshold: float
    threshold_validation_macro_f1: float
    calibration_train_rows: int
    calibration_validation_rows: int
    fit_warnings: list[str]

    def predict(self, frame: pd.DataFrame) -> Predictions:
        probabilities = np.asarray(self.pipeline.predict_proba(features(frame)))
        decoded = self.encoder.inverse_transform(np.asarray(self.pipeline.classes_, dtype=int))
        if self.task == "binary":
            normal = list(decoded).index(0)
            attack_score = probabilities[:, list(decoded).index(1)]
            default_binary = decoded[probabilities.argmax(axis=1)].astype(int)
            family_prediction = default_families = None
            score_labels = np.where(decoded == 0, "Normal", "Attack")
        else:
            normal = list(decoded).index("Normal")
            attack_score = np.clip(1 - probabilities[:, normal], 0, 1)
            default_families = decoded[probabilities.argmax(axis=1)]
            default_binary = (default_families != "Normal").astype(int)
            attack_columns = np.flatnonzero(decoded != "Normal")
            family_prediction = decoded[attack_columns[probabilities[:, attack_columns].argmax(axis=1)]]
            family_prediction = np.where(attack_score >= self.threshold, family_prediction, "Normal")
            score_labels = decoded
        return Predictions(
            attack_score=attack_score, binary=(attack_score >= self.threshold).astype(int),
            default_binary=default_binary, probabilities=probabilities,
            score_labels=score_labels, families=family_prediction, default_families=default_families,
        )


def fit_detector(train, *, classifier, method, task="multiclass", inner_splits=5,
                 seed=SEED, model_params=None) -> FittedDetector:
    """Fit one treatment; reusable by E3 with a prefiltered training frame.

    First fit on an inner training partition and select a detection threshold
    using only its inner validation partition. Then refit a fresh pipeline on
    all supplied training rows. Preprocessing and sampling exclude threshold
    validation data in the provisional fit and outer validation data in both
    fits. E1 MLP pipelines disable internal validation splits after resampling.
    """
    train = training_frame(train)
    if task not in ("binary", "multiclass"):
        raise ValueError("task must be binary or multiclass.")
    target = train[BINARY_LABEL] if task == "binary" else train[FAMILY_LABEL]
    encoder = LabelEncoder().fit(target)
    proper, calibration = make_folds(train, inner_splits, seed)[0]
    messages = []

    def fit_part(part):
        y = encoder.transform(part[BINARY_LABEL] if task == "binary" else part[FAMILY_LABEL])
        pipeline = make_training_pipeline(
            classifier, method, len(encoder.classes_), seed=seed, model_params=model_params,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(features(part), y, **get_fit_params(method, y))
        messages.extend(f"{w.category.__name__}: {w.message}" for w in caught)
        return pipeline

    provisional = FittedDetector(
        fit_part(train.iloc[proper]), encoder, task, 0.5, np.nan,
        len(proper), len(calibration), messages,
    )
    calibration_scores = provisional.predict(train.iloc[calibration]).attack_score
    threshold, score = select_threshold(train.iloc[calibration][BINARY_LABEL], calibration_scores)
    return FittedDetector(
        fit_part(train), encoder, task, threshold, score, len(proper), len(calibration),
        list(dict.fromkeys(messages)),
    )


@dataclass
class FoldOutput:
    row: dict
    family_metrics: pd.DataFrame
    predictions: pd.DataFrame


@dataclass
class CVResult:
    folds: pd.DataFrame
    families: pd.DataFrame
    predictions: pd.DataFrame


def _score_fold(detector, validation):
    prediction = detector.predict(validation)
    evaluations = {}
    for rule in ("tuned", "default"):
        binary = prediction.binary if rule == "tuned" else prediction.default_binary
        detection = evaluate_predictions(
            validation[BINARY_LABEL].to_numpy(), binary,
            y_score=prediction.attack_score, family_true=validation[FAMILY_LABEL].to_numpy(),
        )
        classification = detection
        if detector.task == "multiclass":
            families = prediction.families if rule == "tuned" else prediction.default_families
            classification = evaluate_predictions(
                validation[FAMILY_LABEL].to_numpy(), families, task="multiclass",
                y_score=prediction.probabilities, score_labels=prediction.score_labels,
            )
        evaluations[rule] = (detection, classification)
    return prediction, evaluations


def run_cv(train, *, classifier, method, task="multiclass", n_splits=N_FOLDS,
           inner_splits=5, seed=SEED, model_params=None,
           on_fold: Callable[[FoldOutput], None] | None = None,
           skip_folds=()) -> CVResult:
    """Evaluate ONE classifier/method/task on fixed stratified grouped folds.

    Failures become status='error' rows and are never assigned invented scores.
    on_fold permits immediate checkpointing. E6 should pair successful rows by
    dataset_id, fold_id, task and classifier, comparing method against 'none'.
    """
    train = training_frame(train)
    data_hash = dataset_id(train)
    records, family_tables, prediction_tables = [], [], []
    for fold, (fit_indices, validation_indices) in enumerate(make_folds(train, n_splits, seed), 1):
        if fold in skip_folds:
            continue
        fold_hash = hashlib.sha256(validation_indices.tobytes()).hexdigest()[:16]
        row = {
            "dataset_id": data_hash, "fold_id": fold_hash, "fold": fold,
            "task": task, "classifier": classifier, "method": method, "seed": seed,
            "expected_folds": n_splits, "inner_splits": inner_splits,
            "implementation": IMPLEMENTATIONS.get(method, method),
            "train_rows": len(fit_indices), "validation_rows": len(validation_indices),
        }
        if classifier == "mlp":
            row["mlp_stopping"] = (
                "configured epoch budget; no internal validation split" if method == "focal_loss"
                else "training-loss convergence up to max_iter; no internal validation split"
            )
            row["comparison_limitations"] = (
                "Custom focal MLP differs from sklearn MLP; this comparison does not "
                "isolate the loss function alone." if method == "focal_loss" else ""
            )
        start = time.perf_counter()
        family_table = prediction_table = pd.DataFrame()
        try:
            detector = fit_detector(
                train.iloc[fit_indices], classifier=classifier, method=method, task=task,
                inner_splits=inner_splits, seed=seed, model_params=model_params,
            )
            validation = train.iloc[validation_indices]
            prediction, evaluations = _score_fold(detector, validation)
            family_parts = []
            for rule, (detection, classification) in evaluations.items():
                prefix = "" if rule == "tuned" else "default_"
                row.update({prefix + name: value for name, value in classification.summary.items()})
                for metric in ("macro_f1", "pr_auc", "average_precision", "roc_auc"):
                    row[f"{prefix}detection_{metric}"] = detection.summary[metric]
                row[prefix + "rare_family_macro_recall"] = float(
                    detection.per_family.loc[RARE_FAMILIES, "detection_recall"].mean(skipna=False)
                )
                for family in ATTACK_FAMILIES:
                    row[f"{prefix}recall_{family}"] = detection.per_family.loc[family, "detection_recall"]
                table = classification.per_family.reset_index()
                for name in (*REPORT_IDENTITY, "fold", "fold_id", "task", "classifier", "method"):
                    table[name] = row[name]
                table["decision_rule"] = rule
                family_parts.append(table)
            family_table = pd.concat(family_parts, ignore_index=True)
            row.update({
                "status": "ok", "error": "", "threshold": detector.threshold,
                "threshold_validation_macro_f1": detector.threshold_validation_macro_f1,
                "calibration_train_rows": detector.calibration_train_rows,
                "calibration_validation_rows": detector.calibration_validation_rows,
                "warnings": detector.fit_warnings,
                "unavailable_metrics": evaluations["tuned"][1].metadata["unavailable_metrics"],
                "resampled_class_counts": detector.pipeline.named_steps["coverage"].class_counts_,
                "preprocessor_fit_rows": detector.pipeline.named_steps["prepare"].n_fit_rows_,
            })
            sampler = detector.pipeline.named_steps["sample_encoded"]
            row["adasyn_generated"] = getattr(sampler, "n_generated_", None)
            row["adasyn_zero_allocation"] = getattr(sampler, "no_synthetic_samples_", False)
            prediction_table = pd.DataFrame({
                "source_row": validation_indices, "fold": fold,
                "label": validation[BINARY_LABEL].to_numpy(),
                "attack_cat": validation[FAMILY_LABEL].to_numpy(),
                "attack_score": prediction.attack_score, "prediction": prediction.binary,
                "default_prediction": prediction.default_binary,
            })
            if prediction.families is not None:
                prediction_table["family_prediction"] = prediction.families
                prediction_table["default_family_prediction"] = prediction.default_families
            for index, name in enumerate(prediction.score_labels):
                prediction_table[f"probability_{name}"] = prediction.probabilities[:, index]
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
        row["fit_and_score_seconds"] = time.perf_counter() - start
        output = FoldOutput(row, family_table, prediction_table)
        if on_fold is not None:
            on_fold(output)
        records.append(row)
        if not family_table.empty:
            family_tables.append(family_table)
        if not prediction_table.empty:
            prediction_tables.append(prediction_table)
    return CVResult(
        pd.DataFrame(records),
        pd.concat(family_tables, ignore_index=True) if family_tables else pd.DataFrame(),
        pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame(),
    )


# ============================================================================
# Result summaries and figures
# ============================================================================


METRICS = [
    "macro_f1", "pr_auc", "average_precision", "roc_auc",
    "detection_macro_f1", "detection_pr_auc", "detection_average_precision", "detection_roc_auc",
    "rare_family_macro_recall", "false_positives_per_100k_normal",
    "default_macro_f1", "default_rare_family_macro_recall", "default_false_positives_per_100k_normal",
]


KEY = ["task", "classifier", "method"]
REPORT_IDENTITY = ["dataset_id", "seed", "expected_folds", "inner_splits"]


def _report_identity(table):
    """Require one identified dataset/split protocol per report, including E6 inputs."""
    missing = set(REPORT_IDENTITY) - set(table.columns)
    if missing:
        raise ValueError(f"Summary inputs need identity columns: {sorted(missing)}.")
    identity = []
    for column in REPORT_IDENTITY:
        if table[column].isna().any() or table[column].nunique() != 1:
            raise ValueError(f"Summary inputs must use one dataset/protocol; mixed or missing {column}.")
        identity.append(table[column].iloc[0])
    return tuple(identity)


def summarize(folds):
    """Summarize one dataset/protocol; incomplete groups receive no estimates."""
    if folds.empty:
        return pd.DataFrame()
    identity = _report_identity(folds)
    rows = []
    for keys, group in folds.groupby(KEY, sort=True):
        row = {**dict(zip(REPORT_IDENTITY, identity)), **dict(zip(KEY, keys))}
        successful = group[group["status"] == "ok"]
        expected = int(group["expected_folds"].iloc[0])
        complete = len(successful) == expected and successful["fold_id"].nunique() == expected
        row.update(completed_folds=len(successful), expected_folds=expected,
                   status="complete" if complete else "incomplete")
        for metric in METRICS:
            values = pd.to_numeric(successful.get(metric, pd.Series(dtype=float)), errors="coerce")
            defined = complete and len(values) == expected and values.notna().all()
            row[metric + "_mean"] = float(values.mean()) if defined else np.nan
            row[metric + "_std"] = float(values.std(ddof=1)) if defined else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Before/after differences use identical fold IDs, never unrelated averages.
    for metric in ("macro_f1", "rare_family_macro_recall", "false_positives_per_100k_normal"):
        summary["delta_" + metric] = np.nan
    for index, row in summary.iterrows():
        if row["status"] != "complete":
            continue
        subset = folds[(folds["task"] == row["task"]) & (folds["classifier"] == row["classifier"]) & (folds["status"] == "ok")]
        baseline = subset[subset["method"] == "none"].set_index("fold_id")
        treatment = subset[subset["method"] == row["method"]].set_index("fold_id")
        if len(baseline) != row["expected_folds"] or set(baseline.index) != set(treatment.index):
            continue
        for metric in ("macro_f1", "rare_family_macro_recall", "false_positives_per_100k_normal"):
            if metric in treatment and metric in baseline:
                difference = treatment[metric] - baseline[metric]
                if difference.notna().all():
                    summary.loc[index, "delta_" + metric] = float(difference.mean())
    return summary


def summarize_families(families, folds):
    """Per-family fold estimates and paired detection-recall changes vs none."""
    if families.empty:
        return pd.DataFrame()
    identity = _report_identity(folds)
    if _report_identity(families) != identity:
        raise ValueError("Family and fold tables must use the same dataset/protocol.")
    rows = []
    for keys, group in families.groupby([*KEY, "decision_rule", "family"], sort=True):
        row = {**dict(zip(REPORT_IDENTITY, identity)),
               **dict(zip([*KEY, "decision_rule", "family"], keys))}
        corresponding = folds[
            (folds.task == row["task"]) & (folds.classifier == row["classifier"])
            & (folds.method == row["method"])
        ]
        expected = int(corresponding.expected_folds.iloc[0])
        successful = corresponding[corresponding.status == "ok"]
        complete = (len(group) == expected and group.fold_id.nunique() == expected
                    and len(successful) == expected
                    and set(group.fold_id) == set(successful.fold_id))
        row.update(status="complete" if complete else "incomplete", completed_folds=len(group),
                   is_rare=row["family"] in RARE_FAMILIES)
        row["support_total"] = int(group.support.sum()) if complete else np.nan
        row["detected_total"] = int(group.detected_as_attack.sum()) if complete else np.nan
        row["pooled_detection_recall"] = (
            row["detected_total"] / row["support_total"] if complete and row["support_total"] else np.nan
        )
        for metric in ("detection_recall", "classification_precision", "classification_recall",
                       "classification_f1", "pr_auc", "average_precision", "roc_auc"):
            values = group.get(metric, pd.Series(dtype=float))
            defined = complete and len(values) == expected and values.notna().all()
            row[metric + "_mean"] = float(values.mean()) if defined else np.nan
            row[metric + "_std"] = float(values.std(ddof=1)) if defined else np.nan
        baseline = families[
            (families.task == row["task"]) & (families.classifier == row["classifier"])
            & (families.method == "none") & (families.decision_rule == row["decision_rule"])
            & (families.family == row["family"])
        ]
        baseline_folds = folds[
            (folds.task == row["task"]) & (folds.classifier == row["classifier"])
            & (folds.method == "none") & (folds.status == "ok")
        ]
        row["delta_detection_recall"] = np.nan
        if (complete and len(baseline) == expected and len(baseline_folds) == expected
                and set(baseline.fold_id) == set(group.fold_id) == set(baseline_folds.fold_id)):
            differences = group.set_index("fold_id").detection_recall - baseline.set_index("fold_id").detection_recall
            if differences.notna().all():
                row["delta_detection_recall"] = float(differences.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, records, *, plots=True):
    """Consolidate checkpoint rows; plot only complete successful comparisons."""
    rows = []
    families = []
    for record in records:
        row = {k: v for k, v in record.items() if k != "artifact_sha256"}
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, sort_keys=True)
        rows.append(row)
        if record["status"] == "ok":
            path = out_dir / "checkpoints" / (record["checkpoint"] + ".families.csv")
            if path.exists():
                families.append(pd.read_csv(path))
    folds = pd.DataFrame(rows)
    if folds.empty:
        for name in ("fold_results.csv", "summary.csv", "family_results.csv", "family_summary.csv"):
            (out_dir / name).unlink(missing_ok=True)
        _clear_report_figures(out_dir)
        return folds
    # Validate before replacing any tables in an existing output directory.
    summary = summarize(folds)
    family_table = pd.concat(families, ignore_index=True) if families else pd.DataFrame()
    family_summary = summarize_families(family_table, folds)
    folds = folds.sort_values([*KEY, "fold"])
    folds.to_csv(out_dir / "fold_results.csv", index=False)
    if families:
        family_table.to_csv(out_dir / "family_results.csv", index=False)
        family_summary.to_csv(out_dir / "family_summary.csv", index=False)
    else:
        for name in ("family_results.csv", "family_summary.csv"):
            (out_dir / name).unlink(missing_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    if plots:
        _plots(out_dir, records, summary)
    else:
        _clear_report_figures(out_dir)
    return summary


def _clear_report_figures(out_dir):
    """Remove only figures owned by E1 so old runs cannot masquerade as current."""
    for pattern in ("*_curves.png", "*_rare_recall_tradeoff.png"):
        for path in (out_dir / "figures").glob(pattern):
            path.unlink()


def _plots(out_dir, records, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    # These names are owned by this report. An incomplete rerun must not leave
    # an older successful figure looking like a plot of the current results.
    _clear_report_figures(out_dir)
    complete = summary[summary["status"] == "complete"]
    for (task, classifier), group in complete.groupby(["task", "classifier"]):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for method in group["method"]:
            matching = [r for r in records if r["task"] == task and r["classifier"] == classifier
                        and r["method"] == method and r["status"] == "ok"]
            predictions = pd.concat([
                pd.read_csv(out_dir / "checkpoints" / (r["checkpoint"] + ".predictions.csv.gz"))
                for r in matching
            ], ignore_index=True)
            truth, scores = predictions["label"], predictions["attack_score"]
            precision, recall, _ = precision_recall_curve(truth, scores)
            fpr, tpr, _ = roc_curve(truth, scores)
            axes[0].plot(recall, precision, label=method)
            axes[1].plot(fpr, tpr, label=method)
        axes[0].set(xlabel="Attack recall", ylabel="Attack precision", title="Precision-recall")
        axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{classifier}, {task} training: pooled out-of-fold detection scores")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task}_{classifier}_curves.png", dpi=160)
        plt.close(fig)
    for task, group in complete.groupby("task"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in group.iterrows():
            x = row["false_positives_per_100k_normal_mean"]
            y = row["rare_family_macro_recall_mean"]
            if pd.notna(x) and pd.notna(y):
                ax.scatter(x, y)
                ax.annotate(f"{row['classifier']} / {row['method']}", (x, y),
                            xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set(xlabel="False positives per 100,000 normal connections",
               ylabel="Mean recall across the four rare families",
               title=f"{task} training: tuned detection operating points")
        ax.set_xlim(0, 100_000)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task}_rare_recall_tradeoff.png", dpi=160)
        plt.close(fig)


# ============================================================================
# Command-line arguments and checkpoint support
# ============================================================================


def _json_values(value):
    if isinstance(value, dict):
        return {str(k): _json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_values(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(train, *, n_splits, inner_splits, seed, model_params):
    import imblearn
    import scipy
    import sklearn
    import xgboost

    source_names = ["e1.py", "run_e1.py", "config.py", "preprocess.py", "evaluate.py"]
    return {
        "schema_version": 1, "dataset_id": dataset_id(train), "rows": len(train),
        "family_counts": train[FAMILY_LABEL].value_counts().to_dict(),
        "n_splits": n_splits, "inner_splits": inner_splits, "seed": seed,
        "model_params": model_params,
        "split_strategy": "StratifiedGroupKFold by family, grouped on identical raw features",
        "threshold_objective": "binary macro-F1 on an inner validation fold; ties prefer fewer alerts",
        "mlp_stopping": "internal validation disabled before all E1 MLP fits; max_iter is the budget",
        "sources": {name: _file_hash(Path(__file__).with_name(name)) for name in source_names},
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "imblearn": imblearn.__version__,
            "xgboost": xgboost.__version__, "scipy": scipy.__version__,
        },
    }


def _save_fold(out_dir, output):
    row = output.row.copy()
    key = f"{row['task']}__{row['classifier']}__{row['method']}__fold{row['fold']}"
    row["checkpoint"] = key
    directory = out_dir / "checkpoints"
    checksums = {}
    if row["status"] == "ok":
        for suffix, table, compression in [
            (".families.csv", output.family_metrics, None),
            (".predictions.csv.gz", output.predictions, "gzip"),
        ]:
            path = directory / (key + suffix)
            temporary = path.with_suffix(path.suffix + ".tmp")
            table.to_csv(temporary, index=False, compression=compression)
            temporary.replace(path)
            checksums[path.name] = _file_hash(path)
    row["artifact_sha256"] = checksums
    _write_json(directory / (key + ".json"), row)


def _valid_checkpoint(out_dir, row):
    if row.get("status") != "ok" or len(row.get("artifact_sha256", {})) != 2:
        return False
    return all(
        (out_dir / "checkpoints" / name).exists()
        and _file_hash(out_dir / "checkpoints" / name) == digest
        for name, digest in row["artifact_sha256"].items()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--out-dir", default="results/e1")
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS)
    parser.add_argument("--methods", nargs="+", choices=IMBALANCE_METHODS, default=IMBALANCE_METHODS)
    parser.add_argument("--tasks", nargs="+", choices=["binary", "multiclass"], default=["binary", "multiclass"])
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-size", type=int, help="optional stratified subset size, recorded in the manifest")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume only if data, code, versions and settings match")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--model-params", type=Path, help="optional JSON mapping classifier names to parameter overrides")
    args = parser.parse_args(argv)
    path = Path(args.train_path)
    if not path.exists():
        parser.error(
            f"Raw training CSV not found: {path}. Download UNSW_NB15_training-set.csv from "
            "https://research.unsw.edu.au/projects/unsw-nb15-dataset and place it in data/, "
            "or supply --train-path. preprocess.py does not download it and is not needed before E1."
        )
    if path.name.lower() == "unsw_nb15_testing-set.csv":
        parser.error("The official testing partition must not be used as E1 training data.")
    train = training_frame(load_raw(path))
    raw_rows = len(train)
    if not args.keep_duplicates:
        train = train.drop_duplicates().reset_index(drop=True)
    deduplicated_rows = len(train)
    if args.sample_size is not None:
        if not 0 < args.sample_size <= len(train):
            parser.error("--sample-size must be positive and no larger than the available training data.")
        if args.sample_size < len(train):
            selected, _ = train_test_split(
                np.arange(len(train)), train_size=args.sample_size,
                stratify=train[FAMILY_LABEL], random_state=args.seed,
            )
            train = train.iloc[np.sort(selected)].reset_index(drop=True)
    model_params = json.loads(args.model_params.read_text()) if args.model_params else {}
    if not isinstance(model_params, dict) or set(model_params) - set(CLASSIFIERS):
        parser.error("--model-params must map known classifier names to parameter dictionaries.")
    if not all(isinstance(v, dict) for v in model_params.values()):
        parser.error("Every model-params value must be a parameter dictionary.")
    pairs = [(c, m) for c in dict.fromkeys(args.classifiers) for m in dict.fromkeys(args.methods)
             if m != "focal_loss" or c in ("mlp", "xgboost")]
    if not pairs:
        parser.error("No valid classifier/treatment pairs: focal loss applies only to MLP and XGBoost.")
    tasks = list(dict.fromkeys(args.tasks))
    manifest = _manifest(train, n_splits=args.n_folds, inner_splits=args.inner_folds,
                         seed=args.seed, model_params=model_params)
    manifest.update(raw_rows=raw_rows, deduplicated_rows=deduplicated_rows,
                    sample_size=args.sample_size, keep_duplicates=args.keep_duplicates)
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            parser.error("Output already exists; use --resume or choose a different --out-dir.")
        if json.loads(manifest_path.read_text()) != _json_values(manifest):
            parser.error("Cannot resume: data, code, versions or settings changed. Choose a new --out-dir.")
    elif out_dir.exists() and any(out_dir.iterdir()):
        parser.error("Nonempty output directory has no E1 manifest; choose a different --out-dir.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    _write_json(manifest_path, manifest)
    splits = make_folds(train, args.n_folds, args.seed)
    # Save exact outer/inner memberships, independent of the selected treatment.
    split_plan = []
    for fold, (fit, validation) in enumerate(splits, 1):
        inner_train, inner_validation = make_folds(train.iloc[fit], args.inner_folds, args.seed)[0]
        split_plan.append({"fold": fold, "train_rows": fit.tolist(), "validation_rows": validation.tolist(),
                           "threshold_train_rows": fit[inner_train].tolist(),
                           "threshold_validation_rows": fit[inner_validation].tolist()})
    _write_json(out_dir / "split_plan.json", split_plan)
    saved = [json.loads(p.read_text()) for p in (out_dir / "checkpoints").glob("*.json")]
    completed = {(r["task"], r["classifier"], r["method"], r["fold"])
                 for r in saved if _valid_checkpoint(out_dir, r)}
    total = sum((task, c, m, fold) not in completed for task in tasks for c, m in pairs for fold in range(1, args.n_folds + 1))
    print(f"Training rows: {raw_rows:,} raw -> {deduplicated_rows:,} deduplicated -> {len(train):,} selected", flush=True)
    print(f"{len(pairs)} valid classifier/treatment pairs; {total} new outer folds; two fits per fold.", flush=True)
    print("The official test partition is not opened. Runtime estimate updates after the first fold.", flush=True)
    finished, elapsed = 0, 0.0

    def checkpoint(output):
        nonlocal finished, elapsed
        _save_fold(out_dir, output)
        finished += 1
        elapsed += output.row["fit_and_score_seconds"]
        remaining = elapsed / finished * (total - finished)
        r = output.row
        detail = f"macro-F1={r.get('macro_f1', float('nan')):.3f}" if r["status"] == "ok" else r["error"]
        print(f"[{finished}/{total}] {r['task']} {r['classifier']} / {r['method']} fold {r['fold']}: "
              f"{r['status']} {detail}; estimated remaining {remaining / 60:.1f} min", flush=True)

    for task in tasks:
        for classifier, method in pairs:
            skip = [fold for fold in range(1, args.n_folds + 1) if (task, classifier, method, fold) in completed]
            run_cv(train, classifier=classifier, method=method, task=task,
                   n_splits=args.n_folds, inner_splits=args.inner_folds, seed=args.seed,
                   model_params=model_params.get(classifier), on_fold=checkpoint, skip_folds=skip)
    records = [json.loads(p.read_text()) for p in sorted((out_dir / "checkpoints").glob("*.json"))]
    for record in records:
        if record["status"] == "ok" and not _valid_checkpoint(out_dir, record):
            record.update(status="error", error="Checkpoint artifacts are missing or corrupt; rerun this selection with --resume.")
    summary = write_report(out_dir, records, plots=not args.no_plots)
    print(f"Wrote fold metrics, family metrics, summary and checkpoints to {out_dir.resolve()}", flush=True)
    if not summary.empty:
        print(summary[["task", "classifier", "method", "status", "completed_folds", "macro_f1_mean"]].to_string(index=False))
    return int(any(r["status"] != "ok" for r in records))
