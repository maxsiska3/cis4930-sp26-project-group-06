"""Hand-checkable scoring cases. Run: python -m unittest -v test_evaluate"""

import unittest

import numpy as np
import pandas as pd

from config import ATTACK_FAMILIES
from evaluate import evaluate_predictions


class BinaryEvaluationTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = evaluate_predictions(
            [0, 0, 1, 1], [0, 0, 1, 1],
            y_score=[0.1, 0.2, 0.8, 0.9],
            family_true=["Normal", "Normal", "Worms", "Shellcode"],
        )
        for metric in ["macro_f1", "pr_auc", "average_precision", "roc_auc"]:
            self.assertEqual(result.summary[metric], 1.0)
        self.assertEqual(result.summary["false_positives_per_100k_normal"], 0.0)
        self.assertEqual(result.per_family.loc["Worms", "detection_recall"], 1.0)
        np.testing.assert_array_equal(result.confusion_matrix.to_numpy(), [[2, 0], [0, 2]])

    def test_known_misses_false_alarms_and_ranking(self):
        result = evaluate_predictions(
            [0, 0, 1, 1], [0, 1, 0, 1],
            y_score=[0.1, 0.8, 0.4, 0.9],
            family_true=["Normal", "Normal", "Worms", "Shellcode"],
        )
        self.assertEqual(result.summary["macro_f1"], 0.5)
        self.assertEqual(result.summary["false_positives_per_100k_normal"], 50_000.0)
        self.assertAlmostEqual(result.summary["roc_auc"], 3 / 4)
        self.assertAlmostEqual(result.summary["pr_auc"], 19 / 24)
        self.assertAlmostEqual(result.summary["average_precision"], 5 / 6)
        for metric in ["precision", "recall", "f1"]:
            np.testing.assert_allclose(result.per_class[metric], [0.5, 0.5])
        for name in ["true_negatives", "false_positives", "false_negatives", "true_positives"]:
            self.assertEqual(result.summary[name], 1)
        self.assertEqual(result.per_family.loc["Worms", "detection_recall"], 0.0)
        self.assertEqual(result.per_family.loc["Shellcode", "detection_recall"], 1.0)

    def test_auc_uses_probabilities_instead_of_thresholded_predictions(self):
        result = evaluate_predictions(
            [0, 0, 1, 1], [0, 0, 0, 0], y_score=[0.1, 0.2, 0.8, 0.9]
        )
        self.assertEqual(result.summary["roc_auc"], 1.0)
        self.assertEqual(result.summary["pr_auc"], 1.0)
        self.assertEqual(result.per_class.loc[1, "recall"], 0.0)
        self.assertEqual(result.per_class.loc[1, "precision"], 0.0)
        self.assertAlmostEqual(result.summary["macro_f1"], 1 / 3)

    def test_absent_family_has_zero_support_and_undefined_recall(self):
        result = evaluate_predictions(
            [0, 1], [0, 1], family_true=["Normal", "Worms"]
        )
        self.assertEqual(list(result.per_family.index), ATTACK_FAMILIES)
        self.assertEqual(result.per_family.loc["Analysis", "support"], 0)
        self.assertEqual(result.per_family.loc["Analysis", "detected_as_attack"], 0)
        self.assertTrue(np.isnan(result.per_family.loc["Analysis", "detection_recall"]))
        self.assertIn("per_family.Analysis.detection_recall", result.metadata["unavailable_metrics"])
        self.assertNotIn("precision", result.per_family.columns)

    def test_missing_optional_inputs_are_explicit(self):
        result = evaluate_predictions([0, 1], [0, 1])
        for metric in ["pr_auc", "average_precision", "roc_auc"]:
            self.assertTrue(np.isnan(result.summary[metric]))
            self.assertIn(metric, result.metadata["unavailable_metrics"])
        self.assertTrue(result.per_family["support"].isna().all())
        self.assertTrue(result.per_family["detection_recall"].isna().all())
        self.assertIn("per_family", result.metadata["unavailable_metrics"])

    def test_attack_only_evaluation(self):
        result = evaluate_predictions(
            [1, 1], [0, 1], y_score=[0.2, 0.9], family_true=["Worms", "Worms"]
        )
        self.assertEqual(result.summary["n_normal"], 0)
        self.assertEqual(result.summary["false_positives"], 0)
        self.assertTrue(np.isnan(result.summary["false_positives_per_100k_normal"]))
        self.assertTrue(np.isnan(result.summary["roc_auc"]))
        self.assertEqual(result.per_family.loc["Worms", "detection_recall"], 0.5)

    def test_normal_only_evaluation(self):
        result = evaluate_predictions([0, 0], [0, 1], y_score=[0.1, 0.8])
        self.assertEqual(result.summary["false_positives_per_100k_normal"], 50_000.0)
        for metric in ["pr_auc", "average_precision", "roc_auc"]:
            self.assertTrue(np.isnan(result.summary[metric]))
        self.assertAlmostEqual(result.summary["macro_f1"], 1 / 3)

    def test_matching_series_indices(self):
        result = evaluate_predictions(
            pd.Series([0, 1], index=[20, 10]),
            pd.Series([0, 1], index=[20, 10]),
            family_true=pd.Series(["Normal", "Worms"], index=[20, 10]),
        )
        self.assertEqual(result.summary["macro_f1"], 1.0)

    def test_reordered_series_are_rejected(self):
        for argument in ["y_pred", "y_score", "family_true"]:
            with self.subTest(argument=argument):
                inputs = {
                    "y_true": pd.Series([0, 1], index=[20, 10]),
                    "y_pred": [0, 1],
                }
                values = ["Normal", "Worms"] if argument == "family_true" else [0, 1]
                inputs[argument] = pd.Series(values, index=[10, 20])
                with self.assertRaisesRegex(ValueError, "identical pandas row indices"):
                    evaluate_predictions(**inputs)

    def test_family_alias_and_whitespace(self):
        result = evaluate_predictions(
            [0, 1], [0, 1], family_true=[" Normal ", " Backdoors "]
        )
        self.assertEqual(result.per_family.loc["Backdoor", "support"], 1)

    def test_invalid_labels(self):
        for values in [[], [[0, 1]], [0, 2], [0, np.nan], ["0", "1"], [0, None]]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    evaluate_predictions(values, [0, 1])
                with self.assertRaises(ValueError):
                    evaluate_predictions([0, 1], values)

    def test_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "must have 2 rows"):
            evaluate_predictions([0, 1], [1])
        for argument, values in [("y_score", [0.1]), ("family_true", ["Normal"])]:
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(ValueError, "must have 2 rows"):
                    evaluate_predictions([0, 1], [0, 1], **{argument: values})

    def test_invalid_probabilities(self):
        for scores in [[-0.1, 0.9], [0.1, 1.1], [np.nan, 0.9], [0.1, np.inf],
                       [[0.9, 0.1], [0.1, 0.9]], ["low", "high"]]:
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    evaluate_predictions([0, 1], [0, 1], y_score=scores)

    def test_invalid_family_labels(self):
        for families in [["Normal", "Unknown"], ["Normal", None],
                         ["Normal", "Normal"], ["Worms", "Normal"]]:
            with self.subTest(families=families):
                with self.assertRaises(ValueError):
                    evaluate_predictions([0, 1], [0, 1], family_true=families)

    def test_multiclass_is_not_silently_scored_as_binary(self):
        with self.assertRaisesRegex(ValueError, "unknown families"):
            evaluate_predictions([0, 1], [0, 1], task="multiclass")


class MulticlassEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["Normal", "Worms", "Shellcode"]

    def test_perfect_predictions_for_all_project_classes(self):
        labels = ["Normal", *ATTACK_FAMILIES]
        result = evaluate_predictions(
            labels, labels, task="multiclass",
            y_score=np.eye(len(labels)), score_labels=labels,
        )
        for metric in ["macro_f1", "pr_auc", "average_precision", "roc_auc"]:
            self.assertEqual(result.summary[metric], 1.0)
        self.assertEqual(result.summary["false_positives_per_100k_normal"], 0.0)
        np.testing.assert_array_equal(result.confusion_matrix.to_numpy(), np.eye(len(labels)))
        self.assertEqual(list(result.confusion_matrix.index), labels)
        self.assertEqual(result.metadata["auc_labels"], labels)
        self.assertTrue((result.per_family["classification_precision"] == 1).all())
        self.assertTrue((result.per_family["classification_recall"] == 1).all())

    def test_family_classification_and_attack_detection_are_distinct(self):
        result = evaluate_predictions(
            ["Normal", "Normal", "Worms", "Worms", "Shellcode", "Shellcode"],
            ["Normal", "Worms", "Shellcode", "Normal", "Shellcode", "Worms"],
            task="multiclass", labels=self.labels,
        )
        self.assertAlmostEqual(result.summary["macro_f1"], 1 / 3)
        self.assertEqual(result.summary["false_positives_per_100k_normal"], 50_000.0)
        self.assertEqual(result.summary["true_positives"], 3)
        self.assertEqual(result.summary["false_negatives"], 1)
        self.assertEqual(result.summary["true_negatives"], 1)
        self.assertEqual(result.summary["false_positives"], 1)
        self.assertEqual(result.per_family.loc["Worms", "detection_recall"], 0.5)
        self.assertEqual(result.per_family.loc["Worms", "classification_recall"], 0.0)
        self.assertEqual(result.per_family.loc["Shellcode", "detection_recall"], 1.0)
        self.assertEqual(result.per_family.loc["Shellcode", "classification_recall"], 0.5)
        self.assertEqual(result.per_family.loc["Shellcode", "classification_precision"], 0.5)
        self.assertEqual(result.per_family.loc["Worms", "predicted_count"], 2)
        np.testing.assert_array_equal(
            result.confusion_matrix.to_numpy(), [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        )

    def test_known_one_vs_rest_auc_values(self):
        # Worms has one tied negative and one higher-scoring negative. Shellcode
        # has one lower and one higher negative. Values are hand-calculated.
        result = evaluate_predictions(
            self.labels, ["Normal", "Shellcode", "Worms"],
            task="multiclass", labels=self.labels,
            y_score=[[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.2, 0.4, 0.4]],
            score_labels=self.labels,
        )
        np.testing.assert_allclose(result.per_class["roc_auc"], [1, 1 / 4, 1 / 2])
        np.testing.assert_allclose(result.per_class["pr_auc"], [1, 1 / 6, 1 / 4])
        np.testing.assert_allclose(result.per_class["average_precision"], [1, 1 / 3, 1 / 2])
        self.assertAlmostEqual(result.summary["roc_auc"], 7 / 12)
        self.assertAlmostEqual(result.summary["pr_auc"], 17 / 36)
        self.assertAlmostEqual(result.summary["average_precision"], 11 / 18)
        self.assertEqual(result.per_family.loc["Worms", "roc_auc"], 1 / 4)

    def test_reordered_score_columns_produce_identical_results(self):
        probabilities = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.2, 0.4, 0.4]])
        first = evaluate_predictions(
            self.labels, self.labels, task="multiclass", labels=self.labels,
            y_score=probabilities, score_labels=self.labels,
        )
        order = [2, 0, 1]
        reordered = evaluate_predictions(
            self.labels, self.labels, task="multiclass", labels=self.labels,
            y_score=probabilities[:, order], score_labels=[self.labels[i] for i in order],
        )
        self.assertEqual(first.summary, reordered.summary)
        pd.testing.assert_frame_equal(first.per_class, reordered.per_class)
        pd.testing.assert_frame_equal(first.per_family, reordered.per_family)

    def test_macro_f1_uses_fixed_default_universe(self):
        result = evaluate_predictions(
            self.labels, self.labels, task="multiclass",
            y_score=np.eye(3), score_labels=self.labels,
        )
        self.assertAlmostEqual(result.summary["macro_f1"], 3 / 10)
        self.assertEqual(result.confusion_matrix.shape, (10, 10))
        self.assertTrue(np.isnan(result.summary["roc_auc"]))
        self.assertEqual(result.per_class.loc["Worms", "roc_auc"], 1.0)
        self.assertEqual(result.per_class.loc["Analysis", "support"], 0)

    def test_explicit_subset_defines_macro_average(self):
        result = evaluate_predictions(
            self.labels, self.labels, task="multiclass", labels=self.labels,
            y_score=np.eye(3), score_labels=self.labels,
        )
        self.assertEqual(result.summary["macro_f1"], 1.0)
        self.assertEqual(result.summary["roc_auc"], 1.0)
        self.assertEqual(result.metadata["macro_f1_labels"], self.labels)
        self.assertTrue(np.isnan(result.per_family.loc["Analysis", "classification_recall"]))

    def test_class_absent_from_truth_is_not_silently_dropped_from_auc(self):
        result = evaluate_predictions(
            ["Normal", "Worms"], ["Normal", "Worms"],
            task="multiclass", labels=self.labels,
            y_score=[[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], score_labels=self.labels,
        )
        self.assertAlmostEqual(result.summary["macro_f1"], 2 / 3)
        for metric in ["pr_auc", "average_precision", "roc_auc"]:
            self.assertTrue(np.isnan(result.summary[metric]))
            self.assertTrue(np.isnan(result.per_class.loc["Shellcode", metric]))
            self.assertEqual(result.per_class.loc["Worms", metric], 1.0)
            self.assertIn(f"per_class.Shellcode.{metric}", result.metadata["unavailable_metrics"])
        self.assertTrue(np.isnan(result.per_family.loc["Shellcode", "detection_recall"]))

    def test_missing_model_class_has_no_invented_probabilities(self):
        result = evaluate_predictions(
            self.labels, ["Normal", "Normal", "Shellcode"],
            task="multiclass", labels=self.labels,
            y_score=[[0.9, 0.1], [0.7, 0.3], [0.1, 0.9]],
            score_labels=["Normal", "Shellcode"],
        )
        self.assertEqual(result.per_class.loc["Worms", "recall"], 0.0)
        self.assertEqual(result.per_class.loc["Normal", "roc_auc"], 1.0)
        self.assertTrue(np.isnan(result.per_class.loc["Worms", "roc_auc"]))
        self.assertTrue(np.isnan(result.summary["roc_auc"]))
        reason = result.metadata["unavailable_metrics"]["per_class.Worms.roc_auc"]
        self.assertIn("No probability column", reason)

    def test_without_probabilities_still_scores_classification(self):
        result = evaluate_predictions(self.labels, self.labels, task="multiclass", labels=self.labels)
        self.assertEqual(result.summary["macro_f1"], 1.0)
        self.assertTrue(np.isnan(result.summary["roc_auc"]))
        self.assertTrue(result.per_class["roc_auc"].isna().all())
        self.assertEqual(result.per_family.loc["Worms", "classification_precision"], 1.0)

    def test_only_one_true_family(self):
        result = evaluate_predictions(
            ["Worms", "Worms"], ["Worms", "Worms"], task="multiclass",
            labels=["Worms"], y_score=[[1.0], [1.0]], score_labels=["Worms"],
        )
        self.assertEqual(result.summary["macro_f1"], 1.0)
        self.assertTrue(np.isnan(result.summary["roc_auc"]))
        self.assertTrue(np.isnan(result.summary["false_positives_per_100k_normal"]))
        self.assertEqual(result.summary["true_positives"], 2)

    def test_normal_only_still_counts_false_alarms(self):
        result = evaluate_predictions(
            ["Normal", "Normal"], ["Normal", "Worms"], task="multiclass",
            labels=["Normal", "Worms"],
        )
        self.assertEqual(result.summary["false_positives_per_100k_normal"], 50_000.0)
        self.assertAlmostEqual(result.summary["macro_f1"], 1 / 3)

    def test_aliases_are_normalized_in_all_family_arguments(self):
        result = evaluate_predictions(
            [" Normal ", "Backdoors"], ["Normal", " Backdoor "], task="multiclass",
            labels=["Normal", "Backdoors"], score_labels=[" Backdoors ", "Normal"],
            y_score=[[0.1, 0.9], [0.9, 0.1]], family_true=["Normal", "Backdoor"],
        )
        self.assertEqual(result.summary["macro_f1"], 1.0)
        self.assertEqual(result.summary["roc_auc"], 1.0)
        self.assertEqual(result.per_family.loc["Backdoor", "classification_recall"], 1.0)

    def test_supplied_family_truth_must_match_exactly(self):
        with self.assertRaisesRegex(ValueError, "must match multiclass y_true"):
            evaluate_predictions(
                ["Normal", "Worms"], ["Normal", "Worms"], task="multiclass",
                family_true=["Normal", "Shellcode"],
            )

    def test_labeled_probability_dataframe(self):
        probabilities = pd.DataFrame(np.eye(3), columns=self.labels, index=[20, 10, 30])
        truth = pd.Series(self.labels, index=probabilities.index)
        result = evaluate_predictions(
            truth, truth, task="multiclass", labels=self.labels,
            y_score=probabilities, score_labels=self.labels,
        )
        self.assertEqual(result.summary["roc_auc"], 1.0)
        with self.assertRaisesRegex(ValueError, "identical pandas row indices"):
            evaluate_predictions(
                truth, truth, task="multiclass", labels=self.labels,
                y_score=probabilities.iloc[::-1], score_labels=self.labels,
            )
        with self.assertRaisesRegex(ValueError, "columns must match score_labels"):
            evaluate_predictions(
                truth, truth, task="multiclass", labels=self.labels,
                y_score=probabilities, score_labels=self.labels[::-1],
            )

    def test_invalid_target_labels(self):
        for values in [[], ["Normal"], ["Normal", "Unknown"], ["Normal", None],
                       [["Normal", "Worms"]], [0, 1]]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    evaluate_predictions(["Normal", "Worms"], values, task="multiclass")
                with self.assertRaises(ValueError):
                    evaluate_predictions(values, ["Normal", "Worms"], task="multiclass")

    def test_declared_labels_cannot_drop_observations(self):
        for labels in [["Normal", "Worms"], [], ["Normal", "Backdoor", "Backdoors"]]:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    evaluate_predictions(self.labels, self.labels, task="multiclass", labels=labels)

    def test_score_labels_are_required_unique_and_match_matrix(self):
        for score_labels in [None, [], ["Normal", "Worms"],
                             ["Normal", "Worms", "Worms"], ["Normal", "Worms", "Unknown"]]:
            with self.subTest(score_labels=score_labels):
                with self.assertRaises(ValueError):
                    evaluate_predictions(
                        self.labels, self.labels, task="multiclass", labels=self.labels,
                        y_score=np.eye(3), score_labels=score_labels,
                    )
        with self.assertRaisesRegex(ValueError, "outside labels"):
            evaluate_predictions(
                self.labels, self.labels, task="multiclass", labels=self.labels,
                y_score=np.eye(3), score_labels=["Normal", "Worms", "Generic"],
            )
        with self.assertRaisesRegex(ValueError, "requires y_score"):
            evaluate_predictions(self.labels, self.labels, task="multiclass", score_labels=self.labels)

    def test_invalid_probability_matrices(self):
        for scores in [
            [0.1, 0.2, 0.3], [[1, 0, 0]],
            [[-0.1, 0.1, 1.0]] * 3, [[1.1, 0, 0]] * 3,
            [[np.nan, 0, 1]] * 3, [[np.inf, 0, 0]] * 3,
            [[0.1, 0.1, 0.1]] * 3, [["x", "y", "z"]] * 3,
        ]:
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    evaluate_predictions(
                        self.labels, self.labels, task="multiclass", labels=self.labels,
                        y_score=scores, score_labels=self.labels,
                    )

    def test_unknown_task_and_binary_only_arguments(self):
        with self.assertRaisesRegex(ValueError, "task must be"):
            evaluate_predictions([0, 1], [0, 1], task="other")
        for argument in ["labels", "score_labels"]:
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(ValueError, "only used with task='multiclass'"):
                    evaluate_predictions([0, 1], [0, 1], **{argument: self.labels})


if __name__ == "__main__":
    unittest.main()
