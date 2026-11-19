"""Hand-calculated statistics, pairing/provenance checks and E1-to-E6 integration."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from e6 import analyze, holm_adjust, load_e1_results, main, signed_rank, validate_folds


def example_folds():
    rows = []
    for method in ("none", "smote"):
        for fold in range(1, 6):
            validation = np.arange((fold - 1) * 10, fold * 10, dtype=np.int64)
            rows.append(dict(
                dataset_id="synthetic", seed=42, expected_folds=5, inner_splits=5,
                task="multiclass", classifier="decision_tree", method=method,
                fold=fold, fold_id=hashlib.sha256(validation.tobytes()).hexdigest()[:16],
                status="ok", train_rows=40, validation_rows=10,
                rare_family_macro_recall=0.1 * fold + (0.01 * fold if method == "smote" else 0),
                macro_f1=0.5 + (0.01 * fold if method == "smote" else 0),
                false_positives_per_100k_normal=100 + (fold if method == "smote" else 0),
            ))
    return pd.DataFrame(rows)


def export_example(directory, *, baseline_only=False, legacy=False):
    """Synthetic E1-format exports, explicitly not measured project results."""
    directory.mkdir()
    (directory / "checkpoints").mkdir()
    folds = example_folds()
    if baseline_only:
        folds = folds[folds.method == "none"].copy()
    if legacy:
        folds = folds.drop(columns="inner_splits")
    folds["checkpoint"] = [f"{r.task}__{r.classifier}__{r.method}__fold{r.fold}" for r in folds.itertuples()]
    folds.to_csv(directory / "fold_results.csv", index=False)
    manifest = dict(dataset_id="synthetic", seed=42, n_splits=5, inner_splits=5, rows=50,
                    sources={"e1.py": "synthetic-fixture"})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    plan = []
    for fold in range(1, 6):
        validation = list(range((fold - 1) * 10, fold * 10))
        fit = sorted(set(range(50)) - set(validation))
        plan.append(dict(fold=fold, train_rows=fit, validation_rows=validation))
    (directory / "split_plan.json").write_text(json.dumps(plan))
    for row in folds.to_dict("records"):
        (directory / "checkpoints" / (row["checkpoint"] + ".json")).write_text(json.dumps(row))
    return folds


def compare(folds, **kwargs):
    options = dict(tasks=["multiclass"], classifiers=["decision_tree"], methods=["smote"],
                   metrics=["rare_family_macro_recall"])
    options.update(kwargs)
    return analyze(folds, **options)


class StatisticalTests(unittest.TestCase):
    def test_hand_calculated_signed_rank(self):
        result = signed_rank([1, 2, 3, -4, 5])
        self.assertEqual(result["statistic"], 4)
        self.assertEqual(result["p_value"], 0.4375)
        self.assertAlmostEqual(result["rank_biserial"], 7 / 15)
        reference = stats.wilcoxon([1, 2, 3, -4, 5], alternative="two-sided", method="exact")
        self.assertEqual(result["p_value"], reference.pvalue)

    def test_five_positive_pairs_cannot_reach_point_zero_five(self):
        for differences in ([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]):
            with self.subTest(differences=differences):
                result = signed_rank(differences)
                self.assertEqual(result["p_value"], 0.0625)
                self.assertEqual(result["rank_biserial"], 1)

    def test_ties_and_zeros_have_a_correct_permutation_distribution(self):
        result = signed_rank([0, 1, -1, 2, 2])
        self.assertEqual(result["statistic"], 1.5)
        self.assertEqual(result["p_value"], 0.375)
        self.assertEqual(result["nonzero_pairs"], 4)
        self.assertAlmostEqual(result["rank_biserial"], 0.7)
        self.assertEqual(signed_rank([0, 0, 0, 0, 1])["p_value"], 1)

    def test_all_zero_and_roundoff_handling(self):
        result = signed_rank([0, 0, 0, 0, 0])
        self.assertEqual(result["p_value"], 1)
        self.assertEqual(result["nonzero_pairs"], 0)
        self.assertTrue(np.isnan(result["rank_biserial"]))
        rounded = signed_rank([0.1 + 0.2 - 0.3, 1, -1, 2, 2])
        self.assertEqual(rounded["p_value"], 0.375)
        for bad in ([1], [0, np.nan], [[1, 2]]):
            with self.assertRaises(ValueError):
                signed_rank(bad)

    def test_monte_carlo_is_reproducible(self):
        differences = np.arange(1, 18) * np.where(np.arange(17) % 3, 1, -1)
        first, second = signed_rank(differences), signed_rank(differences)
        self.assertEqual(first["p_value"], second["p_value"])
        self.assertIn("Monte Carlo", first["p_method"])

    def test_holm_matches_hand_calculation_and_keeps_unavailable_hypotheses(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.01, 0.03], 3), [0.06, 0.03, 0.06])
        result = holm_adjust([0.01, np.nan, 0.04], 3)
        np.testing.assert_allclose(result[[0, 2]], [0.03, 0.08])
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(holm_adjust([np.nan], 1)[0]))
        for values, count in [([np.inf], 1), ([0.1], 0), ([0.1], 1.5), ([-0.1], 1)]:
            with self.assertRaises(ValueError):
                holm_adjust(values, count)


class PairingTests(unittest.TestCase):
    def test_pairing_uses_ids_not_input_order(self):
        first, pairs = compare(example_folds())
        shuffled, shuffled_pairs = compare(example_folds().sample(frac=1, random_state=2))
        pd.testing.assert_frame_equal(first, shuffled)
        pd.testing.assert_frame_equal(pairs, shuffled_pairs)
        self.assertAlmostEqual(first.iloc[0].mean_delta, 0.03)
        self.assertEqual(pairs.fold.tolist(), [1, 2, 3, 4, 5])
        self.assertFalse(first.iloc[0].significant_holm)

    def test_failed_missing_mismatched_and_undefined_folds_are_not_dropped(self):
        frame = example_folds()
        changes = [
            frame.drop(index=9),
            frame.assign(status=np.where(frame.index == 9, "error", "ok")),
            frame.assign(fold_id=np.where(frame.index == 9, "different", frame.fold_id)),
            frame.assign(rare_family_macro_recall=np.where(frame.index == 9, np.nan, frame.rare_family_macro_recall)),
            frame.assign(validation_rows=np.where(frame.index == 9, 11, frame.validation_rows)),
            frame.drop(columns="rare_family_macro_recall"),
        ]
        for changed in changes:
            with self.subTest(columns=changed.columns.tolist(), rows=len(changed)):
                results, pairs = compare(changed)
                self.assertEqual(results.iloc[0].status, "unavailable")
                self.assertTrue(pd.isna(results.iloc[0].p_value))
                self.assertEqual(results.iloc[0].n_pairs, 0)
                self.assertTrue(pairs.empty)

    def test_mixed_protocols_duplicates_and_wrong_units_are_rejected(self):
        frame = example_folds()
        for name, value in [("dataset_id", "other"), ("seed", 7), ("expected_folds", 3), ("inner_splits", 2)]:
            changed = frame.copy()
            changed.loc[9, name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "one dataset/split protocol"):
                compare(changed)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare(pd.concat([frame, frame.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_folds(frame.drop(columns="fold_id"))
        frame.loc[9, "rare_family_macro_recall"] = 95
        results, _ = compare(frame)
        self.assertEqual(results.iloc[0].status, "unavailable")
        self.assertIn("units/range", results.iloc[0].reason)

    def test_false_alarm_direction_and_units(self):
        results, pairs = compare(example_folds(), metrics=["false_positives_per_100k_normal"])
        self.assertEqual(results.iloc[0].better, "lower")
        self.assertEqual(results.iloc[0].mean_delta, 3)
        self.assertEqual(results.iloc[0].mean_improvement, -3)
        self.assertEqual(results.iloc[0].folds_improved, 0)
        self.assertEqual(pairs.delta.tolist(), [1, 2, 3, 4, 5])

    def test_missing_methods_remain_in_holm_family(self):
        results, _ = compare(example_folds(), methods=["smote", "adasyn"])
        available = results[results.status == "ok"].iloc[0]
        self.assertEqual(available.planned_hypotheses, 2)
        self.assertEqual(available.p_value, 0.0625)
        self.assertEqual(available.p_holm, 0.125)
        self.assertEqual(len(results), 2)

    def test_focal_plan_uses_current_e1_support(self):
        results, _ = analyze(example_folds(), tasks=["multiclass"], methods=["focal_loss"],
                             metrics=["macro_f1"])
        self.assertEqual(set(results.classifier), {"mlp", "xgboost"})
        with self.assertRaisesRegex(ValueError, "other than none"):
            compare(example_folds(), methods=["none"])


class LoadingAndCliTests(unittest.TestCase):
    def test_export_loading_and_legacy_metadata(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "e1"
                export_example(path, legacy=legacy)
                folds, provenance = load_e1_results(path)
                self.assertEqual(len(folds), 10)
                self.assertTrue(folds.inner_splits.eq(5).all())
                self.assertEqual(provenance["filled_from_manifest"], ["inner_splits"] if legacy else [])

    def test_tampered_manifest_split_and_metrics_are_rejected(self):
        for artifact in ("manifest", "split", "noninteger_split", "metrics"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "e1"
                export_example(path)
                if artifact == "manifest":
                    manifest = json.loads((path / "manifest.json").read_text())
                    manifest["seed"] = 7
                    (path / "manifest.json").write_text(json.dumps(manifest))
                elif artifact in ("split", "noninteger_split"):
                    plan = json.loads((path / "split_plan.json").read_text())
                    if artifact == "split":
                        plan[0]["validation_rows"][0] = 49
                    else:
                        plan[0]["train_rows"][0] += 0.5
                    (path / "split_plan.json").write_text(json.dumps(plan))
                else:
                    folds = pd.read_csv(path / "fold_results.csv")
                    folds.loc[0, "macro_f1"] = 0.99
                    folds.to_csv(path / "fold_results.csv", index=False)
                with self.assertRaises(ValueError):
                    load_e1_results(path)

    def test_cli_writes_auditable_results_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "e1", root / "e6"
            export_example(source)
            args = ["--input-dir", str(source), "--out-dir", str(out), "--tasks", "multiclass",
                    "--classifiers", "decision_tree", "--methods", "smote"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            comparisons = pd.read_csv(out / "comparisons.csv")
            self.assertEqual(len(comparisons), 3)
            self.assertTrue(comparisons.status.eq("ok").all())
            self.assertEqual(len(pd.read_csv(out / "paired_differences.csv")), 15)
            self.assertIn("0.0625", (out / "E6_SUMMARY.md").read_text())
            metadata = json.loads((out / "manifest.json").read_text())
            self.assertEqual(metadata["alternative"], "two-sided")
            self.assertEqual(metadata["input_sha256"]["fold_results.csv"],
                             hashlib.sha256((source / "fold_results.csv").read_bytes()).hexdigest())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)

    def test_baseline_only_is_explicitly_unavailable_and_keeps_csv_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "e1", root / "e6"
            export_example(source, baseline_only=True)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--input-dir", str(source), "--out-dir", str(out)])
            self.assertEqual(code, 1)
            comparisons = pd.read_csv(out / "comparisons.csv")
            self.assertEqual(len(comparisons), 132)
            self.assertTrue(comparisons.status.eq("unavailable").all())
            self.assertFalse(comparisons.significant_holm.any())
            self.assertTrue(pd.read_csv(out / "paired_differences.csv").empty)
            self.assertIn("No significance conclusions", (out / "E6_SUMMARY.md").read_text())

    def test_runner_help_does_not_start_analysis(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_e6.py")), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input-dir", result.stdout)
        self.assertNotIn("Reading saved", result.stdout)

    def test_native_e1_outputs_feed_e6_without_changes(self):
        from e1 import main as run_e1
        from test_e1 import raw_example

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, source, out = root / "synthetic_train.csv", root / "e1", root / "e6"
            raw_example(24).to_csv(train, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_e1([
                    "--train-path", str(train), "--out-dir", str(source), "--tasks", "multiclass",
                    "--classifiers", "decision_tree", "--methods", "none", "class_weights",
                    "--n-folds", "3", "--inner-folds", "2", "--no-plots",
                ]), 0)
                self.assertEqual(main([
                    "--input-dir", str(source), "--out-dir", str(out), "--tasks", "multiclass",
                    "--classifiers", "decision_tree", "--methods", "class_weights",
                ]), 0)
            results = pd.read_csv(out / "comparisons.csv")
            self.assertTrue(results.n_pairs.eq(3).all())


if __name__ == "__main__":
    unittest.main()
