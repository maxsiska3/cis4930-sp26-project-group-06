"""E4 checks on small synthetic data (no dataset needed).

Run with:  python -m unittest -v test_e4
"""
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from config import ATTACK_FAMILIES
from e4_transfer import coverage, fit_scores, matrix, run, run_source, source_subset


def synthetic(seed=0, per_family=30, normal=200):
    """Normal around 0; each attack family shifted along one of two directions."""
    rng = np.random.default_rng(seed)
    frames = [pd.DataFrame({"x1": rng.normal(0, 1, normal), "x2": rng.normal(0, 1, normal),
                            "label": 0, "attack_cat": "Normal"})]
    for i, family in enumerate(ATTACK_FAMILIES):
        # First four families share a direction, so they should transfer to each other.
        shift = (4, 0) if i < 4 else (0, 4)
        frames.append(pd.DataFrame({"x1": rng.normal(shift[0], 1, per_family),
                                    "x2": rng.normal(shift[1], 1, per_family),
                                    "label": 1, "attack_cat": family}))
    return pd.concat(frames, ignore_index=True)


class E4Tests(unittest.TestCase):
    def setUp(self):
        self.train, self.test = synthetic(0), synthetic(1)

    def test_subset_has_only_normal_and_source(self):
        subset = source_subset(self.train, "Worms")
        self.assertEqual(set(subset.attack_cat), {"Normal", "Worms"})
        self.assertEqual((subset.attack_cat == "Normal").sum(), 200)

    def test_test_labels_do_not_change_scores(self):
        subset = source_subset(self.train, "DoS")
        scores = fit_scores(subset, self.test, "decision_tree", "none", False)
        relabeled = self.test.assign(label=0, attack_cat="Normal")
        np.testing.assert_array_equal(scores, fit_scores(subset, relabeled, "decision_tree", "none", False))

    def test_one_row_per_target_and_similar_families_transfer(self):
        rows = pd.DataFrame(run_source(self.train, self.test, ATTACK_FAMILIES[0],
                                       "decision_tree", "none", False))
        self.assertEqual(list(rows.target), ATTACK_FAMILIES)
        self.assertEqual(rows.seen.sum(), 1)
        recall = rows.set_index("target").detection_recall
        # Family 1 shares a direction with family 0; family 8 does not.
        self.assertGreater(recall[ATTACK_FAMILIES[1]], 0.8)
        self.assertLess(recall[ATTACK_FAMILIES[8]], 0.3)

    def test_raw_mode_preprocesses_inside_training_subset(self):
        raw = self.train.assign(proto="tcp", service="no_service", state="con")
        target = self.test.assign(proto="unseen", service="new_service", state="new_state")
        scores = fit_scores(source_subset(raw, "DoS"), target, "decision_tree", "none", True)
        self.assertEqual(len(scores), len(target))

    def test_every_method_runs(self):
        for method in ["none", "class_weights", "undersample", "smote", "adasyn", "smote_enn"]:
            with self.subTest(method=method):
                rows = run_source(self.train, self.test, "Worms", "decision_tree", method, False)
                self.assertEqual(len(rows), 9)

    def test_full_run_matrix_coverage_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.train.to_csv(root / "train_processed.csv", index=False)
            self.test.to_csv(root / "test_processed.csv", index=False)
            args = Namespace(out=str(root / "e4.csv"), raw=False, processed_dir=directory,
                             sources=ATTACK_FAMILIES, classifiers=["decision_tree"], methods=["none"])
            run(args)
            rows = pd.read_csv(args.out)
            self.assertEqual(len(rows), 81)
            self.assertEqual(matrix(rows, "decision_tree", "none").shape, (9, 9))
            self.assertEqual(len(coverage(rows)), 9)
            self.assertTrue((root / "e4_decision_tree_none.png").exists())
            before = Path(args.out).read_bytes()
            run(args)  # second run should skip everything already saved
            self.assertEqual(Path(args.out).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()