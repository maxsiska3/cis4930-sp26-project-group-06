"""E2 invariants and real-pipeline integration on small synthetic family data."""
import unittest
import tempfile
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd

from e2_rare_family import choose_threshold, fit_scores, metric_row, reduce_family, run, recall_curve


class E2Tests(unittest.TestCase):
    def test_rounded_levels_do_not_duplicate_seed_observations(self):
        rows = pd.DataFrame({"retained": [1] * 9, "seed": [0, 1, 2] * 3,
                             "family_detection_recall": [.2, .5, .8] * 3})
        curve = recall_curve(rows)
        self.assertEqual(curve.loc[1, "count"], 3)
        self.assertAlmostEqual(curve.loc[1, "std"], .3)

    def test_argmax_detection_is_separate_from_tuned_alerts(self):
        target = pd.DataFrame({"label": [0, 1], "attack_cat": ["Normal", "Worms"]})
        metrics = metric_row(target, np.array([.1, .4]), np.array(["Normal", "Shellcode"]), .5, "Worms")
        self.assertEqual(metrics["family_detection_recall"], 0)
        self.assertEqual(metrics["family_argmax_detection_recall"], 1)

    def setUp(self):
        rng = np.random.default_rng(10)
        self.frame = pd.DataFrame({"x": rng.normal(size=180),
                                   "attack_cat": np.repeat(["Normal", "Worms", "Shellcode"], [100, 40, 40]),
                                   "label": np.repeat([0, 1], [100, 80])})

    def test_subsampling_preserves_other_families_and_nested_samples(self):
        full = self.frame.copy()
        small = reduce_family(full, "Worms", .05, 1)
        larger = reduce_family(full, "Worms", .5, 1)
        pd.testing.assert_frame_equal(full, self.frame)
        pd.testing.assert_frame_equal(small[small.attack_cat != "Worms"].reset_index(drop=True),
                                      full[full.attack_cat != "Worms"].reset_index(drop=True))
        self.assertEqual((small.attack_cat == "Worms").sum(), 2)
        self.assertTrue(set(small[small.attack_cat == "Worms"].x).issubset(larger[larger.attack_cat == "Worms"].x))
        self.assertEqual((reduce_family(full, "Worms", .001, 1).attack_cat == "Worms").sum(), 1)

    def test_threshold_optimizes_validation_attack_f1(self):
        self.assertAlmostEqual(choose_threshold([0, 0, 1, 1], [.1, .2, .3, .9]), .3)

    def test_tiny_sample_smote_is_unavailable_not_silently_clamped(self):
        tiny = reduce_family(self.frame, "Worms", .05, 0)
        with self.assertRaisesRegex(ValueError, "unsupported:"):
            fit_scores(tiny, self.frame, "decision_tree", "smote", False)

    def test_real_family_pipeline_and_detection_scoring(self):
        train = self.frame.iloc[np.r_[0:80, 100:130, 140:170]]
        target = self.frame.iloc[np.r_[80:100, 130:140, 170:180]]
        for method in ["none", "class_weights", "undersample", "smote", "adasyn", "smote_enn"]:
            with self.subTest(method=method):
                scores, predicted = fit_scores(train, target, "decision_tree", method, False)
                self.assertEqual(len(scores), len(target))
                self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
                metrics = metric_row(target, scores, predicted, .5, "Worms")
                self.assertEqual(metrics["family_support"], 10)
                self.assertTrue(0 <= metrics["family_detection_recall"] <= 1)

    def test_test_labels_do_not_affect_model_predictions(self):
        scores, predicted = fit_scores(self.frame, self.frame, "decision_tree", "none", False)
        relabeled = self.frame.assign(label=0, attack_cat="Normal")
        other_scores, other_predicted = fit_scores(self.frame, relabeled, "decision_tree", "none", False)
        np.testing.assert_array_equal(scores, other_scores)
        np.testing.assert_array_equal(predicted, other_predicted)

    def test_other_shared_classifiers_accept_family_codes(self):
        for classifier in ["random_forest", "xgboost", "mlp"]:
            with self.subTest(classifier=classifier):
                scores, predicted = fit_scores(self.frame, self.frame, classifier, "none", False)
                self.assertEqual(len(scores), len(self.frame))
                self.assertTrue(set(predicted).issubset(set(self.frame.attack_cat)))

    def test_raw_preprocessing_handles_unseen_test_categories(self):
        raw = self.frame.assign(proto="tcp", service="no_service", state="con")
        target = raw.assign(proto="unseen", service="new_service", state="new_state")
        scores, predicted = fit_scores(raw, target, "decision_tree", "none", True)
        self.assertEqual(len(scores), len(target))
        self.assertEqual(len(predicted), len(target))

    def test_csv_pipeline_reports_unsupported_cases_and_resumes(self):
        extra = self.frame[self.frame.attack_cat == "Shellcode"]
        full = pd.concat([self.frame, extra.assign(attack_cat="Analysis"),
                          extra.assign(attack_cat="Backdoor")], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ["train", "test"]:
                full.to_csv(root / f"{split}_processed.csv", index=False)
            args = Namespace(out=str(root / "runs.csv"), raw=False, processed_dir=directory,
                             families=["Worms"], classifiers=["decision_tree"],
                             methods=["none", "smote"], levels=[.001], seeds=[0], breakdown_drop=.1)
            run(args)
            rows = pd.read_csv(args.out)
            final = rows[rows.phase == "test"]
            self.assertEqual(len(final), 4)
            self.assertEqual(final.status.value_counts().to_dict(), {"ok": 3, "unsupported": 1})
            self.assertFalse(rows.duplicated(["family", "classifier", "method", "fraction", "seed", "phase", "fold"]).any())
            before = Path(args.out).read_bytes()
            run(args)
            self.assertEqual(Path(args.out).read_bytes(), before)
            self.assertTrue((root / "runs_floors.csv").exists())
            coverage = pd.read_csv(root / "runs_coverage.csv")
            self.assertEqual(len(coverage), 4)
            self.assertEqual(coverage.status.value_counts().to_dict(), {"ok": 3, "unsupported": 1})
            self.assertTrue((root / "runs_decision_tree.png").exists())


if __name__ == "__main__":
    unittest.main()
