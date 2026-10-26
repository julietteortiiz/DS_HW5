"""Public examples for HW05. Run: python3 tests.py.

These tests exercise small independent cases, not the complete grading suite.
"""
from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

from NaiveBayes import NaiveBayes
from Partition import Example, Partition
from run_NB import accuracy, balanced_error_rate, confusion_matrix, parse_args, read_arff, read_csv


class TestNaiveBayes(unittest.TestCase):
    def setUp(self) -> None:
        self.train = Partition(
            [Example({'color': 'red'}, 0), Example({'color': 'red'}, 0),
             Example({'color': 'blue'}, 1)],
            {'color': ['red', 'blue', 'green']}, 3)

    def test_smoothed_probabilities_and_absent_class(self) -> None:
        """Zero-count values and an absent declared class still get probability."""
        model = NaiveBayes(self.train)
        self.assertEqual(len(model.log_class), 3)
        for actual, expected in zip(model.log_class, [3/6, 2/6, 1/6]):
            self.assertAlmostEqual(math.exp(actual), expected)
        self.assertAlmostEqual(math.exp(model.log_feature['color']['green'][0]), 1/5)
        self.assertAlmostEqual(math.exp(model.log_feature['color']['red'][2]), 1/3)
        for k in range(3):
            self.assertAlmostEqual(sum(math.exp(model.log_feature['color'][v][k])
                                       for v in self.train.F['color']), 1.0)

    def test_scores_and_multiclass_predictions(self) -> None:
        model = NaiveBayes(self.train)
        expected = [math.log(3/6 * 3/5), math.log(2/6 * 1/4), math.log(1/6 * 1/3)]
        for actual, value in zip(model.predict_log_scores({'color': 'red'}), expected):
            self.assertAlmostEqual(actual, value)
        self.assertEqual(model.predict([{'color': 'red'}, {'color': 'blue'}]), [0, 1])
        self.assertEqual(model.predict([]), [])
        three = Partition([Example({'x': str(k)}, k) for k in range(3)],
                          {'x': ['0', '1', '2']}, 3)
        self.assertEqual(NaiveBayes(three).classify({'x': '2'}), 2)

    def test_tie_and_unknown_values(self) -> None:
        model = NaiveBayes(Partition([Example({'x': 'a'}, 0), Example({'x': 'a'}, 1)],
                                     {'x': ['a']}, 2))
        self.assertEqual(model.classify({'x': 'a'}), 0)
        with self.assertRaises(KeyError):
            model.classify({'x': 'new'})
        with self.assertRaises(KeyError):
            model.classify({})

    def test_helpers_are_quiet_and_predictions_use_cached_probabilities(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            model = NaiveBayes(self.train)
            before = model.predict_log_scores({'color': 'red'})
            # Classification must not recount the original examples.
            self.train.data.clear()
            self.assertEqual(model.predict_log_scores({'color': 'red'}), before)
            model.classify({'color': 'red'})
            confusion_matrix([0, 1], [0, 0], 2)
        self.assertEqual(output.getvalue(), '')


class TestDataAndMetrics(unittest.TestCase):
    def test_csv_label_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / 'tiny.csv'
            filename.write_text('color,sex,income\nred,Female,low\nblue,Male,high\nred,Female,high\n')
            partition = read_csv(str(filename))
        self.assertEqual((partition.n, partition.K), (3, 2))
        self.assertEqual(partition.F, {'color': ['red', 'blue'], 'income': ['low', 'high']})
        self.assertEqual([e.label for e in partition.data], [1, 0, 1])
        self.assertNotIn('sex', partition.data[0].features)

    def test_arff_preserves_declared_domains(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / 'tiny.arff'
            filename.write_text("% comment\n@relation tiny\n@ATTRIBUTE 'color' {red, blue, green}\n"
                                "@attribute class {dog, cat, bird}\n@DATA\nblue,bird\nred,dog\n")
            partition = read_arff(str(filename))
        self.assertEqual((partition.n, partition.K), (2, 3))
        self.assertEqual(partition.F, {'color': ['red', 'blue', 'green']})
        self.assertEqual([e.label for e in partition.data], [2, 0])

    def test_metrics_with_unequal_class_sizes(self) -> None:
        actual = [0, 0, 0, 0, 1, 1]
        predicted = [0, 0, 0, 1, 0, 1]
        matrix = confusion_matrix(actual, predicted, 2)
        self.assertEqual(matrix, [[3, 1], [1, 1]])
        self.assertAlmostEqual(accuracy(actual, predicted), 4/6)
        self.assertAlmostEqual(balanced_error_rate(matrix), 0.375)
        self.assertEqual(confusion_matrix([], [], 3), [[0, 0, 0]] * 3)
        with self.assertRaises(ValueError):
            accuracy([], [])
        with self.assertRaises(ValueError):
            confusion_matrix([0], [], 2)
        with self.assertRaises(ValueError):
            confusion_matrix([-1], [0], 2)
        with self.assertRaises(ValueError):
            balanced_error_rate([[1, 0], [0, 0]])

    def test_arguments_without_modifying_sys_argv(self) -> None:
        args = parse_args(['-r', 'train.csv', '-e', 'test.csv'])
        self.assertEqual(args.train_filename, 'train.csv')
        self.assertEqual(args.test_filename, 'test.csv')


if __name__ == '__main__':
    unittest.main(verbosity=2)
