"""E5 checks on small synthetic data (no dataset needed).

Run with:  python -m unittest -v test_e5
"""
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from config import ATTACK_FAMILIES
from e5_rules import (FAMILIES, METHODS, Model, boundary_stats, describe_rule, importance_shift,
                      leaf_table, original_value, path_conditions, run, run_config)


def synthetic(seed=0, per_family=40, normal=300, spread=0.3):
    """Normal near 0 on x1; each attack family sits at its own x1 value.

    sbytes is a log column (config.LOG_COLUMNS) so threshold conversion is exercised;
    `proto` is one-hot encoded in raw mode.
    """
    rng = np.random.default_rng(seed)
    frames = [pd.DataFrame({"x1": rng.normal(0, 0.3, normal), "sbytes": rng.normal(2, 0.3, normal),
                            "label": 0, "attack_cat": "Normal"})]
    for i, family in enumerate(ATTACK_FAMILIES, 1):
        frames.append(pd.DataFrame({"x1": rng.normal(3 * i, spread, per_family),
                                    "sbytes": rng.normal(6, 0.3, per_family),
                                    "label": 1, "attack_cat": family}))
    return pd.concat(frames, ignore_index=True)


def with_categoricals(df):
    proto = np.where(df.label == 1, "udp", "tcp")
    return df.assign(proto=proto, service="no_service", state="con")


class E5Tests(unittest.TestCase):
    def setUp(self):
        self.train, self.test = synthetic(0), synthetic(1)

    def test_thresholds_convert_back_to_original_units(self):
        model = Model(self.train, "decision_tree", "none", raw=False)
        i = model.features.index("x1")
        scaled = (4.2 - model.scaler.mean_[i]) / model.scaler.scale_[i]
        self.assertAlmostEqual(original_value(model, "x1", scaled), 4.2, places=5)
        j = model.features.index("sbytes")
        scaled = (np.log1p(100) - model.scaler.mean_[j]) / model.scaler.scale_[j]
        self.assertAlmostEqual(original_value(model, "sbytes", scaled), 100, places=3)

    def test_rules_reproduce_the_tree(self):
        """Every training row satisfying a leaf's readable rule lands in that leaf."""
        model = Model(self.train, "decision_tree", "none", raw=False)
        leaves = model.clf.apply(model.X_scaled)
        for leaf in np.unique(leaves):
            steps = path_conditions(model.clf, leaf)
            mask = np.ones(len(leaves), bool)
            for feature, right, threshold in steps:
                column = model.X_scaled[:, feature]
                mask &= column > threshold if right else column <= threshold
            np.testing.assert_array_equal(mask, leaves == leaf)
            self.assertTrue(describe_rule(model, steps))

    def test_one_hot_rules_name_categories(self):
        raw = with_categoricals(self.train)
        model = Model(raw, "decision_tree", "none", raw=True)
        normal_leaf = model.clf.apply(model.X_scaled[:1])[0]
        rule = describe_rule(model, path_conditions(model.clf, normal_leaf))
        self.assertNotIn("proto_", rule)

    def test_no_synthetic_leaves_without_resampling(self):
        model = Model(self.train, "decision_tree", "none", raw=False)
        table = leaf_table(model.clf, model.X_scaled, model.codes)
        self.assertTrue((table.real_support > 0).all())
        stats = boundary_stats(model, model.scaled(self.test),
                               pd.Categorical(self.test.attack_cat, categories=FAMILIES).codes)
        self.assertEqual(stats.synthetic_only_leaves.sum(), 0)

    def test_leaf_with_only_synthetic_records_is_flagged(self):
        # Fit on real rows plus fake "synthetic" Worms in an empty region (x1 = 1.5);
        # only the real rows are passed to leaf_table, so that leaf has no real support.
        from sklearn.tree import DecisionTreeClassifier
        real_X = self.train[["x1", "sbytes"]].to_numpy()
        real_codes = pd.Categorical(self.train.attack_cat, categories=FAMILIES).codes
        fake_X = np.column_stack([np.full(20, 1.5), np.full(20, 6.0)])   # between Normal and Fuzzers
        worms = FAMILIES.index("Worms")
        tree = DecisionTreeClassifier(random_state=0).fit(
            np.vstack([real_X, fake_X]), np.concatenate([real_codes, np.full(20, worms)]))
        table = leaf_table(tree, real_X, real_codes)
        only = table[table.real_support == 0]
        self.assertEqual(len(only), 1)
        self.assertEqual(only.predicted.iloc[0], worms)
        self.assertEqual(only.fit_samples.iloc[0], 20)
        self.assertEqual(only.leaf.iloc[0], tree.apply(fake_X[:1])[0])

    def test_smote_fits_on_more_rows_than_real_data(self):
        model = Model(self.train, "decision_tree", "smote", raw=False)
        table = leaf_table(model.clf, model.X_scaled, model.codes)
        self.assertEqual(table.real_total.sum(), len(self.train))
        self.assertEqual(table.fit_samples.sum(), 300 * len(FAMILIES))   # every class raised to Normal's 300

    def test_test_labels_do_not_change_results(self):
        relabeled = self.test.assign(label=0, attack_cat="Normal")
        model = Model(self.train, "decision_tree", "none", raw=False)
        np.testing.assert_array_equal(model.predict(self.test)[0], model.predict(relabeled)[0])

    def test_config_outputs_for_both_classifiers(self):
        for classifier in ["decision_tree", "random_forest"]:
            with self.subTest(classifier=classifier):
                summary, families, imp, rules, text = run_config(self.train, self.test, classifier,
                                                                 "smote", raw=False)
                self.assertEqual(list(families.family), FAMILIES)
                self.assertAlmostEqual(imp.importance.sum(), 1.0, places=6)
                self.assertGreaterEqual(summary["surrogate_fidelity"], 0.9)
                self.assertGreater(summary["rare_detection_recall"], 0.9)
                self.assertIn("Worms: top leaf rules", text)
                self.assertTrue(all(r["family"] in FAMILIES for r in rules))

    def test_every_method_runs(self):
        # Overlapping families: ADASYN refuses perfectly separated classes.
        train, test = synthetic(0, spread=2.0), synthetic(1, spread=2.0)
        for method in METHODS:
            with self.subTest(method=method):
                summary = run_config(train, test, "decision_tree", method, raw=False)[0]
                self.assertTrue(0 <= summary["threshold"] <= 1)

    def test_importance_shift_is_zero_against_itself(self):
        model = Model(self.train, "decision_tree", "none", raw=False)
        imp = pd.DataFrame({"classifier": "decision_tree", "method": "none",
                            "feature": model.features, "importance": model.clf.feature_importances_,
                            "rank": [1, 2]})
        shift = importance_shift(imp, top=2)
        self.assertEqual(shift.loc[0, "top2_overlap_vs_none"], 2)
        self.assertEqual(shift.loc[0, "largest_gain_change"], 0)

    def test_full_run_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.train.to_csv(root / "train_processed.csv", index=False)
            self.test.to_csv(root / "test_processed.csv", index=False)
            args = Namespace(raw=False, processed_dir=directory, out_dir=directory,
                             fig_dir=str(root / "figures"), classifiers=["decision_tree"],
                             methods=["none", "smote"])
            run(args)
            summary = pd.read_csv(root / "e5_summary.csv")
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(pd.read_csv(root / "e5_families.csv")), 2 * len(FAMILIES))
            self.assertEqual(len(pd.read_csv(root / "e5_importance_shift.csv")), 2)
            self.assertTrue((root / "e5_rules" / "decision_tree_smote.txt").exists())
            self.assertTrue((root / "figures" / "e5_importance_decision_tree.png").exists())
            self.assertTrue((root / "figures" / "e5_synthetic_leaves.png").exists())
            before = (root / "e5_summary.csv").read_bytes()
            run(args)   # second run skips everything already saved
            self.assertEqual((root / "e5_summary.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
