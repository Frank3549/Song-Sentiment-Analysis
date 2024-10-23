import os, sys, unittest
import numpy as np
import sentiment as snt
from sklearn import metrics


class BaseModelTest(unittest.TestCase):
    def setUp(self):
        self.model = snt.Sentiment([0, 1])
        self.model.add_example("good", 1)
        self.model.add_example("bad", 0)

    def test_simple_prediction(self):
        """Base model accurately predicts labels for observed words, with pseudo-counts"""
        prediction = self.model.predict("good", 0.0001)
        self.assertEqual(len(prediction), 2)
        self.assertAlmostEqual(np.sum(prediction), 1.0, "Predicted probabilities should sum to 1")
        self.assertEqual(np.argmax(prediction), 1)
        np.testing.assert_array_equal(
            np.greater(prediction, 0) & np.less(prediction, 1),
            [True] * 2,
            err_msg="With pseudo counts probabilities should not be 0",
        )
        np.testing.assert_allclose(
            prediction, [9.9980004e-05, 9.9990002e-01], rtol=1e-4
        )

    def test_missing_word(self):
        """Base model predicts correct probabilities for unobserved words"""
        prediction = self.model.predict("missing", 0.0001)
        np.testing.assert_allclose(prediction, [0.5, 0.5])

    def test_splitting_string(self):
        """Preprocessing splits strings into individual words"""
        self.model.add_example("good ok", 1)
        prediction = self.model.predict("ok", 0.0001)
        np.testing.assert_allclose(
            prediction, [1.49957512e-04, 9.99850042e-01], rtol=1e-4
        )


class BaseModelFullDatasetTest(unittest.TestCase):
    def setUp(self):
        self.model = snt.Sentiment(labels=[0, 1])

    def test_entire_dataset(self):
        """Base model achieves reasonable accuracy for provided test data"""
        for id, example, y_true in snt.process_zipfile(
            os.path.join(os.path.dirname(__file__), "data", "train.zip")
        ):
            self.model.add_example(example, y_true, id=id)

        predictions = []
        for id, example, y_true in snt.process_zipfile(
            os.path.join(os.path.dirname(__file__), "data", "test.zip")
        ):
            # Determine the most likely class from predicted probabilities
            predictions.append((y_true, np.argmax(self.model.predict(example, id=id))))

        y_test, y_true = zip(*predictions)
        accuracy = metrics.accuracy_score(y_test, y_true)
        self.assertGreater(accuracy, 0.7, msg="Accuracy should be greater than 0.70")


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
