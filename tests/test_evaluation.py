"""Run from root as: python -m unittest discover -s tests -v"""
import unittest
from src.evaluation import accuracy_score, confusion_matrix, class_metrics


class TestEvaluation(unittest.TestCase):
    def test_perfect_predictions(self):
        self.assertEqual(accuracy_score([0, 1, 0], [0, 1, 0]), 1.0)
        self.assertEqual(confusion_matrix([0, 1, 0], [0, 1, 0]), [[2, 0], [0, 1]])

    def test_mixed_predictions(self):
        truth = [0, 0, 1, 1]
        prediction = [0, 1, 0, 1]
        self.assertEqual(accuracy_score(truth, prediction), 0.5)
        cm = confusion_matrix(truth, prediction)
        self.assertEqual(cm, [[1, 1], [1, 1]])
        self.assertEqual(class_metrics(cm, positive_label=1)['recall'], 0.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            accuracy_score([0], [])
        with self.assertRaises(ValueError):
            confusion_matrix([0], [])


if __name__ == '__main__':
    unittest.main()
