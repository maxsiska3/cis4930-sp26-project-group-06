"""Leakage, objective, pipeline and end-to-end tests for E1. No real data needed."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

from config import ATTACK_FAMILIES, CLASSIFIERS, IMBALANCE_METHODS
from e1 import (
    ClassCoverageGuard, FocalMLPClassifier, FocalXGBClassifier, FoldPreprocessor,
    ProjectedADASYN, feature_groups, features, fit_detector, focal_objective,
    main, make_folds, make_training_pipeline, run_cv, select_threshold,
    summarize, summarize_families, training_frame, write_report,
)


SUMMARY_IDENTITY = dict(dataset_id="synthetic", seed=42, expected_folds=5, inner_splits=5)


def raw_example(count=40):
    rng = np.random.default_rng(42)
    families = np.repeat(["Normal", *ATTACK_FAMILIES], count)
    n = len(families)
    frame = pd.DataFrame({
        "dur": rng.uniform(0, 20, n), "f1": rng.normal(size=n),
        "f2": rng.normal(size=n), "f3": rng.normal(size=n),
        "proto": rng.choice(["tcp", "udp"], n),
        "service": rng.choice(["http", "-"], n),
        "state": rng.choice(["fin", "int"], n),
        "attack_cat": families, "label": (families != "Normal").astype(int),
    })
    return frame


class FocalLossTests(unittest.TestCase):
    def test_gradient_and_diagonal_hessian_against_finite_differences(self):
        for logits, y in [
            (np.array([-0.4, 0.7]), np.array([0, 1])),
            (np.array([[0.2, -0.3, 0.1], [0.5, 0.3, -0.2]]), np.array([1, 0])),
        ]:
            for gamma in (0, 1, 2):
                with self.subTest(shape=logits.shape, gamma=gamma):
                    _, gradient, hessian = focal_objective(y, logits, gamma=gamma)
                    epsilon = 1e-5
                    for coordinate in np.ndindex(logits.shape):
                        above, below = logits.copy(), logits.copy()
                        above[coordinate] += epsilon
                        below[coordinate] -= epsilon
                        la, ga, _ = focal_objective(y, above, gamma=gamma)
                        lb, gb, _ = focal_objective(y, below, gamma=gamma)
                        numeric_gradient = (la.sum() - lb.sum()) / (2 * epsilon)
                        numeric_hessian = (ga[coordinate] - gb[coordinate]) / (2 * epsilon)
                        self.assertAlmostEqual(gradient[coordinate], numeric_gradient, places=7)
                        self.assertAlmostEqual(hessian[coordinate], numeric_hessian, places=7)

    def test_gamma_zero_is_cross_entropy(self):
        logits = np.array([-0.4, 0.7])
        y = np.array([0, 1])
        weights = np.array([0.5, 2.0])
        loss, gradient, hessian = focal_objective(y, logits, gamma=0, class_weights=weights)
        p = expit(logits)
        expected_loss = -weights[y] * (y * np.log(p) + (1 - y) * np.log(1 - p))
        np.testing.assert_allclose(loss, expected_loss)
        np.testing.assert_allclose(gradient, weights[y] * (p - y))
        np.testing.assert_allclose(hessian, weights[y] * p * (1 - p))

    def test_focal_models_learn_a_separable_binary_problem(self):
        rng = np.random.default_rng(42)
        X = np.r_[rng.normal(-2, 0.2, (30, 3)), rng.normal(2, 0.2, (30, 3))]
        y = np.repeat([0, 1], 30)
        models = [
            FocalXGBClassifier(n_estimators=10, max_depth=2, n_jobs=1),
            FocalMLPClassifier(hidden_layer_sizes=(8,), max_iter=60,
                               learning_rate_init=0.01, early_stopping=False),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__), warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X, y)
                self.assertGreater((model.predict(X) == y).mean(), 0.95)
                np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)


class PipelineTests(unittest.TestCase):
    def test_e1_mlp_never_splits_resampled_data_for_validation(self):
        frame = raw_example(30)
        frame = pd.concat([frame[frame.attack_cat == "Normal"],
                           frame[frame.attack_cat == "Worms"].iloc[:12]], ignore_index=True)
        y = (frame.attack_cat != "Normal").astype(int).to_numpy()
        for method in IMBALANCE_METHODS:
            pipeline = make_training_pipeline(
                "mlp", method, 2, model_params={"max_iter": 2, "hidden_layer_sizes": (4,)},
            )
            self.assertFalse(pipeline.named_steps["clf"].early_stopping)
            if method not in ("none", "smote", "focal_loss"):
                continue
            with self.subTest(method=method), warnings.catch_warnings(), \
                    patch("e1.train_test_split", side_effect=AssertionError("internal split")), \
                    patch("sklearn.neural_network._multilayer_perceptron.train_test_split",
                          side_effect=AssertionError("internal split")):
                warnings.simplefilter("ignore", ConvergenceWarning)
                pipeline.fit(features(frame), y)
                self.assertEqual(pipeline.predict_proba(features(frame)).shape, (len(frame), 2))

    def test_e1_rejects_reenabling_internal_mlp_validation(self):
        for method in ("none", "focal_loss"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "after preprocessing and resampling"):
                make_training_pipeline("mlp", method, 2, model_params={"early_stopping": True})

    def test_validation_does_not_change_fitted_preprocessing(self):
        X = pd.DataFrame({"dur": [1.0, 3.0, np.nan], "proto": ["tcp", "tcp", "udp"],
                          "service": ["-", "http", "http"], "state": ["fin"] * 3})
        pre = FoldPreprocessor(min_proto_count=2).fit(X)
        original_mean = pre.scaler_.mean_.copy()
        validation = X.iloc[[0]].copy()
        validation["dur"] = 10000
        validation["proto"] = "unseen"
        transformed = pre.transform(validation)
        self.assertEqual(pre.medians_["dur"], 2)
        self.assertEqual(pre.n_fit_rows_, 3)
        self.assertEqual(pre.keep_protocols_, {"tcp"})
        self.assertEqual(transformed.iloc[0]["proto"], "other")
        np.testing.assert_array_equal(pre.scaler_.mean_, original_mean)

    def test_target_columns_cannot_leak_into_features(self):
        with self.assertRaisesRegex(ValueError, "Target columns"):
            FoldPreprocessor().fit(raw_example())

    def test_all_26_valid_pipeline_combinations_fit(self):
        frame = raw_example(16)
        # Three overlapping classes with meaningful imbalance for both samplers.
        frame = pd.concat([frame[frame.attack_cat == "Normal"],
                           frame[frame.attack_cat == "Analysis"].iloc[:12],
                           frame[frame.attack_cat == "Worms"].iloc[:8]], ignore_index=True)
        X = features(frame)
        _, y = np.unique(frame.attack_cat, return_inverse=True)
        # Mostly coherent class clusters, plus one ambiguous point per class.
        # Pure noise can legitimately make ENN delete an entire class; perfectly
        # separated data can make ADASYN undefined. This fixture exercises both.
        X = X.copy()
        rng = np.random.default_rng(7)
        X["dur"] = 1.0
        X["service"], X["state"], X["proto"] = "http", "fin", "tcp"
        for column in ["f1", "f2", "f3"]:
            X[column] = y * 10 + rng.normal(0, 0.05, len(y))
            for label in range(3):
                X.loc[np.flatnonzero(y == label)[0], column] = ((label + 1) % 3) * 10
        checked = 0
        for classifier in CLASSIFIERS:
            params = {"n_estimators": 3, "n_jobs": 1} if classifier in ("random_forest", "xgboost") else {}
            if classifier == "mlp":
                params = {"max_iter": 5, "hidden_layer_sizes": (8,), "early_stopping": False}
            for method in IMBALANCE_METHODS:
                if method == "focal_loss" and classifier not in ("mlp", "xgboost"):
                    with self.assertRaises(ValueError):
                        make_training_pipeline(classifier, method, 3)
                    continue
                with self.subTest(classifier=classifier, method=method), warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pipeline = make_training_pipeline(classifier, method, 3, model_params=params)
                    from config import get_fit_params
                    pipeline.fit(X, y, **get_fit_params(method, y))
                    probabilities = pipeline.predict_proba(X.iloc[:4])
                    self.assertEqual(probabilities.shape, (4, 3))
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                    self.assertEqual(pipeline.named_steps["prepare"].n_fit_rows_, len(X))
                    checked += 1
        self.assertEqual(checked, 26)

    def test_class_removal_is_reported_instead_of_hidden(self):
        with self.assertRaisesRegex(ValueError, "removed a target class"):
            ClassCoverageGuard(3).fit_resample(np.ones((4, 2)), [0, 0, 1, 1])

    def test_categorical_synthetic_samples_are_valid(self):
        frame = raw_example(20)
        frame = pd.concat([frame[frame.attack_cat == "Normal"],
                           frame[frame.attack_cat == "Worms"].iloc[:8]], ignore_index=True)
        y = (frame.attack_cat != "Normal").astype(int)
        pipeline = make_training_pipeline("decision_tree", "smote", 2)
        prepared = pipeline.named_steps["prepare"].fit_transform(features(frame))
        sampled, _ = pipeline.named_steps["sample_raw"].fit_resample(prepared, y)
        for column in ["proto", "service", "state"]:
            self.assertLessEqual(set(sampled[column]), set(prepared[column]))
        encoded = pipeline.named_steps["encode"].fit_transform(prepared)
        projected, _ = ProjectedADASYN(n_neighbors=3).fit_resample(encoded, y)
        for column in ["proto", "service", "state"]:
            block = projected.filter(regex=f"^cat__{column}_").to_numpy()
            np.testing.assert_array_equal(block.sum(axis=1), np.ones(len(block)))
            self.assertTrue(np.isin(block, [0, 1]).all())


class CrossValidationTests(unittest.TestCase):
    def test_runner_help_does_not_start_training(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("run_e1.py")), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_e1.py", result.stdout)
        self.assertIn("--classifiers", result.stdout)
        self.assertNotIn("Training rows:", result.stdout)

    def test_feature_duplicates_with_different_labels_stay_in_one_fold(self):
        frame = training_frame(raw_example())
        duplicate = frame.iloc[[0]].copy()
        duplicate["attack_cat"], duplicate["label"] = "Worms", 1
        frame = pd.concat([frame, duplicate], ignore_index=True)
        groups = feature_groups(frame)
        folds = make_folds(frame)
        visited = []
        for fit, validation in folds:
            self.assertFalse(set(groups[fit]) & set(groups[validation]))
            visited.extend(validation)
        self.assertEqual(sorted(visited), list(range(len(frame))))
        second = make_folds(frame)
        for first, again in zip(folds, second):
            np.testing.assert_array_equal(first[1], again[1])

    def test_threshold_matches_an_exhaustive_independent_check(self):
        truth = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.8, 0.4, 0.9, 0.4, 0.8])
        threshold, value = select_threshold(truth, scores)
        candidates = [np.nextafter(1.0, np.inf), *np.unique(scores)[::-1]]
        expected = [f1_score(truth, scores >= t, labels=[0, 1], average="macro", zero_division=0) for t in candidates]
        self.assertAlmostEqual(value, max(expected))
        self.assertEqual(threshold, candidates[np.argmax(expected)])
        tied, _ = select_threshold([0, 0, 1, 1], [0.2] * 4)
        self.assertGreater(tied, 1)

    def test_single_method_cv_is_paired_and_preprocessing_fits_only_training(self):
        frame = raw_example(24)
        first = run_cv(frame, classifier="decision_tree", method="none", n_splits=3, inner_splits=2)
        second = run_cv(frame, classifier="decision_tree", method="class_weights", n_splits=3, inner_splits=2)
        self.assertTrue((first.folds.status == "ok").all(), first.folds.get("error"))
        self.assertTrue((second.folds.status == "ok").all(), second.folds.get("error"))
        self.assertEqual(first.folds.fold_id.tolist(), second.folds.fold_id.tolist())
        self.assertEqual(len(first.predictions), len(frame))
        self.assertEqual(first.predictions.source_row.nunique(), len(frame))
        self.assertTrue((first.folds.preprocessor_fit_rows == first.folds.train_rows).all())
        self.assertTrue((first.folds.calibration_train_rows < first.folds.train_rows).all())
        self.assertEqual(first.folds.dataset_id.nunique(), 1)
        self.assertEqual(len(first.families), 3 * 9 * 2)

    def test_binary_cv_and_reusable_single_fit(self):
        frame = raw_example(24)
        result = run_cv(frame, classifier="decision_tree", method="none", task="binary", n_splits=3, inner_splits=2)
        self.assertTrue((result.folds.status == "ok").all(), result.folds.get("error"))
        held_out = frame[frame.attack_cat != "Worms"]
        detector = fit_detector(held_out, classifier="decision_tree", method="none", inner_splits=2)
        prediction = detector.predict(frame[frame.attack_cat == "Worms"])
        self.assertNotIn("Worms", prediction.score_labels)
        self.assertEqual(len(prediction.binary), 24)

    def test_incomplete_results_never_receive_comparable_means(self):
        rows = pd.DataFrame([
            dict(task="binary", classifier="decision_tree", method="none", fold_id=str(i),
                 **SUMMARY_IDENTITY, status="ok", macro_f1=0.8) for i in range(4)
        ])
        summary = summarize(rows)
        self.assertEqual(summary.iloc[0]["status"], "incomplete")
        self.assertTrue(pd.isna(summary.iloc[0]["macro_f1_mean"]))

    def test_paired_changes_require_the_same_fold_ids(self):
        rows = []
        for method, offset in [("none", 0), ("smote", 0.1)]:
            for fold in range(5):
                rows.append(dict(task="binary", classifier="decision_tree", method=method,
                                 fold_id=str(fold), **SUMMARY_IDENTITY, status="ok",
                                 macro_f1=0.5 + fold * 0.02 + offset))
        frame = pd.DataFrame(rows)
        summary = summarize(frame).set_index("method")
        self.assertAlmostEqual(summary.loc["smote", "delta_macro_f1"], 0.1)
        frame.loc[frame.method == "smote", "fold_id"] += "different"
        summary = summarize(frame).set_index("method")
        self.assertTrue(pd.isna(summary.loc["smote", "delta_macro_f1"]))

    def test_summaries_reject_mixed_or_missing_experiment_identity(self):
        rows = pd.DataFrame([
            dict(**SUMMARY_IDENTITY, task="binary", classifier="decision_tree", method="none",
                 fold_id=str(i), status="ok", macro_f1=0.8) for i in range(5)
        ])
        for column, changed in [("dataset_id", "another_dataset"), ("seed", 7),
                                ("expected_folds", 3), ("inner_splits", 2)]:
            mixed = rows.copy()
            mixed.loc[3:, column] = changed
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "one dataset/protocol"):
                summarize(mixed)
        with self.assertRaisesRegex(ValueError, "identity columns"):
            summarize(rows.drop(columns="dataset_id"))

    def test_family_summary_requires_matching_identity_and_successful_folds(self):
        folds = pd.DataFrame([
            dict(**SUMMARY_IDENTITY, task="multiclass", classifier="decision_tree", method="none",
                 fold_id=str(i), status="ok") for i in range(5)
        ])
        families = folds.drop(columns="status").assign(
            family="Worms", decision_rule="tuned", support=10,
            detected_as_attack=8, detection_recall=0.8,
        )
        self.assertEqual(summarize_families(families, folds).iloc[0].status, "complete")
        other = families.assign(dataset_id="another_dataset")
        with self.assertRaisesRegex(ValueError, "same dataset/protocol"):
            summarize_families(other, folds)
        folds.loc[0, "status"] = "error"
        summary = summarize_families(families, folds).iloc[0]
        self.assertEqual(summary.status, "incomplete")
        self.assertTrue(pd.isna(summary.detection_recall_mean))
        treatment = families.assign(method="smote", detection_recall=0.9, detected_as_attack=9)
        treatment_folds = folds.assign(method="smote", status="ok")
        combined = summarize_families(pd.concat([families, treatment]),
                                      pd.concat([folds, treatment_folds])).set_index("method")
        self.assertEqual(combined.loc["smote", "status"], "complete")
        self.assertTrue(pd.isna(combined.loc["smote", "delta_detection_recall"]))

    def test_failed_or_empty_reports_remove_stale_owned_outputs(self):
        for records in [[], [dict(**SUMMARY_IDENTITY, task="multiclass", classifier="decision_tree",
                                  method="none", fold=1, fold_id="1", status="error")]]:
            with self.subTest(records=len(records)), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                (out / "figures").mkdir()
                for name in ("fold_results.csv", "summary.csv", "family_results.csv", "family_summary.csv"):
                    (out / name).write_text("old complete results")
                stale_plot = out / "figures" / "multiclass_rare_recall_tradeoff.png"
                stale_plot.write_bytes(b"old plot")
                unrelated = out / "figures" / "other_experiment.png"
                unrelated.write_bytes(b"keep")
                summary = write_report(out, records, plots=False)
                self.assertFalse((out / "family_results.csv").exists())
                self.assertFalse((out / "family_summary.csv").exists())
                self.assertFalse(stale_plot.exists())
                self.assertTrue(unrelated.exists())
                if records:
                    self.assertEqual(summary.iloc[0].status, "incomplete")
                else:
                    self.assertFalse((out / "fold_results.csv").exists())
                    self.assertFalse((out / "summary.csv").exists())

    def test_mlp_cv_records_stopping_policy_and_focal_comparison_limit(self):
        for method in ("none", "focal_loss"):
            result = run_cv(raw_example(12), classifier="mlp", method=method, task="binary",
                            n_splits=3, inner_splits=2,
                            model_params={"max_iter": 2, "hidden_layer_sizes": (4,)})
            with self.subTest(method=method):
                self.assertTrue((result.folds.status == "ok").all(), result.folds.get("error"))
                self.assertTrue(result.folds.mlp_stopping.str.contains("no internal validation split").all())
                self.assertEqual(set(result.families.dataset_id), set(result.folds.dataset_id))
                if method == "focal_loss":
                    self.assertTrue(result.folds.comparison_limitations.str.contains("does not isolate").all())

    def test_cli_checkpoint_resume_and_changed_data_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            train = path / "train.csv"
            raw_example(24).to_csv(train, index=False)
            out = path / "out"
            args = ["--train-path", str(train), "--out-dir", str(out), "--classifiers", "decision_tree",
                    "--methods", "none", "--tasks", "multiclass", "--n-folds", "3", "--inner-folds", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(len(list((out / "checkpoints").glob("*.json"))), 3)
            self.assertTrue((out / "figures" / "multiclass_decision_tree_curves.png").exists())
            self.assertTrue((out / "summary.csv").exists())
            family_summary = pd.read_csv(out / "family_summary.csv")
            self.assertEqual(len(family_summary), 18)
            self.assertTrue((family_summary.status == "complete").all())
            np.testing.assert_allclose(family_summary.delta_detection_recall, 0)
            before = {p.name: p.stat().st_mtime_ns for p in (out / "checkpoints").glob("*.json")}
            with patch("e1.fit_detector", side_effect=AssertionError("A completed fit was rerun")):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([*args, "--resume"]), 0)
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in (out / "checkpoints").glob("*.json")})
            damaged = next((out / "checkpoints").glob("*.predictions.csv.gz"))
            damaged.write_bytes(b"corrupted checkpoint")
            with patch("e1.fit_detector", wraps=fit_detector) as fit_spy:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([*args, "--resume"]), 0)
                self.assertEqual(fit_spy.call_count, 1)
            self.assertGreater(len(pd.read_csv(damaged)), 0)
            frame = pd.read_csv(train)
            frame.loc[0, "dur"] += 1
            frame.to_csv(train, index=False)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main([*args, "--resume"])
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
