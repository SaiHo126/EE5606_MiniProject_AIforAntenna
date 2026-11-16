"""Checks for batch training, validation selection and model persistence.

Run all NN checks from the project root:
    python -B -m unittest discover -s tests -p 'test_neural_network*.py' -v
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.NeuralNetworkSkeleton import NeuralNetwork


PARAMETERS = ("W_hidden", "B_hidden", "W_output", "B_output")


def example_data():
    """A simple known regression problem with physical-scale inputs/targets."""
    rng = np.random.default_rng(17)
    X = rng.uniform(-1.0, 1.0, size=(34, 3))
    Y_GHz = 2.4 + X @ np.array([[0.12], [-0.09], [0.04]])
    X_mm = X * np.array([[4.0, 3.0, 0.5]]) + np.array([[40.0, 29.0, 2.0]])
    return X_mm[:25], Y_GHz[:25], X_mm[25:], Y_GHz[25:]


def make_model(hidden_size=8, seed=7):
    return NeuralNetwork(hidden_size=hidden_size, seed=seed)


class NeuralNetworkTrainingTests(unittest.TestCase):
    def test_known_forward_values_shapes_and_mse(self):
        model = NeuralNetwork(hidden_size=2)
        model.W_hidden[:] = [[1, 0, 0], [0, 2, 0]]
        model.B_hidden[:] = [[0.5], [-0.5]]
        model.W_output[:] = [[2, -1]]
        model.B_output[:] = 0.25
        X = np.array([[0.5, 0.75, 9], [-0.5, 0.25, 0]])
        predicted, cache = model.forward(X)
        np.testing.assert_allclose(predicted, [[np.tanh(1.0) + 0.25], [0.25]])
        self.assertEqual(predicted.shape, (2, 1))
        np.testing.assert_array_equal(cache["X"], X.T)
        self.assertEqual(cache["A_hidden"].shape, (2, 2))
        self.assertEqual(model.loss(np.array([[1], [3]]), np.array([[0], [1]])), 2.5)

    def test_partial_batches_keep_raw_values_and_use_only_training_samples(self):
        X, Y, X_validation, Y_validation = example_data()
        model = make_model()
        with patch.object(model, "backward", wraps=model.backward) as backward:
            model.fit(
                {"X_mm": X[:5], "Y_GHz": Y[:5]},
                {"X_mm": X_validation, "Y_GHz": Y_validation},
                epochs=2, batch_size=2,
            )
        sizes = [call.args[2]["X"].shape[1] for call in backward.call_args_list]
        self.assertEqual(sizes, [2, 2, 1, 2, 2, 1])
        # Recover the first epoch's pairs from backward calls. Their values
        # must still be the original mm and GHz, even after shuffling.
        observed = np.concatenate([
            np.column_stack((call.args[2]["X"].T, call.args[1]))
            for call in backward.call_args_list[:3]
        ])
        expected = np.column_stack((X[:5], Y[:5]))
        np.testing.assert_array_equal(
            observed[np.argsort(observed[:, 0])],
            expected[np.argsort(expected[:, 0])],
        )

    def test_training_learns_known_function_and_is_reproducible(self):
        X, Y, X_validation, Y_validation = example_data()
        models = [make_model() for _ in range(2)]
        initial_predicted, _ = models[0].forward(X)
        initial_loss = models[0].loss(initial_predicted, Y)
        constant_baseline_loss = float(np.mean((Y - Y.mean())**2))
        for model in models:
            history = model.fit(
                {"X_mm": X, "Y_GHz": Y},
                {"X_mm": X_validation, "Y_GHz": Y_validation},
                epochs=1000, learning_rate=0.0001, batch_size=7,
            )
            self.assertEqual(len(history["training_loss"]), 1000)
            self.assertEqual(len(history["validation_loss"]), 1000)
            final_predicted, _ = model.forward(X)
            final_loss = model.loss(final_predicted, Y)
            self.assertLess(final_loss, initial_loss * 0.1)
            self.assertLess(final_loss, constant_baseline_loss)
            validation_predicted, _ = model.forward(X_validation)
            restored_loss = model.loss(validation_predicted, Y_validation)
            self.assertAlmostEqual(restored_loss, min(history["validation_loss"]), places=14)
            self.assertEqual(model.best_epoch, int(np.argmin(history["validation_loss"])) + 1)
        for name in PARAMETERS:
            np.testing.assert_array_equal(getattr(models[0], name), getattr(models[1], name))
        self.assertEqual(models[0].history, models[1].history)

    def test_restores_earlier_validation_checkpoint(self):
        # Training targets pull predictions upward while validation prefers -1.
        # The best validation parameters must therefore come from an earlier epoch.
        model = NeuralNetwork(seed=5)
        X = np.zeros((5, 3))
        history = model.fit(
            {"X_mm": X, "Y_GHz": np.ones((5, 1))},
            {"X_mm": X[:3], "Y_GHz": -np.ones((3, 1))},
            epochs=8, learning_rate=0.03, batch_size=2,
        )
        self.assertEqual(model.best_epoch, 1)
        self.assertGreater(history["validation_loss"][-1], history["validation_loss"][0])
        prediction, _ = model.forward(X[:3])
        self.assertAlmostEqual(
            model.loss(prediction, -np.ones((3, 1))), history["validation_loss"][0], places=14
        )

    def test_saved_model_preserves_raw_predictions_settings_and_history(self):
        X, Y, X_validation, Y_validation = example_data()
        model = make_model(hidden_size=3, seed=29)
        model.fit(
            {"X_mm": X, "Y_GHz": Y},
            {"X_mm": X_validation, "Y_GHz": Y_validation},
            epochs=20, learning_rate=0.0001, batch_size=7,
        )
        predictions = [model.predict(design) for design in X_validation]
        expected_GHz, _ = model.forward(X_validation)
        np.testing.assert_allclose(np.array(predictions).reshape(-1, 1), expected_GHz)
        self.assertIsInstance(predictions[0], float)

        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(prefix="nn_test_", dir=project_root) as directory:
            model_path = Path(directory) / "model.npz"
            model.save(model_path)
            with np.load(model_path) as saved:
                self.assertEqual(saved["data_representation"].item(), "raw_mm_GHz")
                self.assertTrue(set(saved.files).isdisjoint({"X_mean", "X_std", "Y_mean", "Y_std"}))
            restored = NeuralNetwork.load(model_path)
        for name in PARAMETERS:
            np.testing.assert_array_equal(getattr(restored, name), getattr(model, name))
        self.assertEqual(restored.hidden_size, 3)
        self.assertEqual(restored.seed, 29)
        self.assertEqual(restored.best_epoch, model.best_epoch)
        self.assertEqual(restored.history, model.history)
        self.assertEqual(restored.training_settings, model.training_settings)
        np.testing.assert_array_equal(
            [restored.predict(design) for design in X_validation], predictions
        )

    def test_loaded_npz_inputs_match_dictionary_training(self):
        X, Y, X_validation, Y_validation = example_data()
        dataTrain = {"X_mm": X, "Y_GHz": Y}
        dataValidation = {"X_mm": X_validation, "Y_GHz": Y_validation}
        dictionary_model = make_model()
        archive_model = make_model()
        settings = {"epochs": 5, "learning_rate": 0.0001, "batch_size": 7}
        dictionary_model.fit(dataTrain, dataValidation, **settings)

        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(prefix="nn_data_test_", dir=project_root) as directory:
            train_path = Path(directory) / "train.npz"
            validation_path = Path(directory) / "validation.npz"
            np.savez(train_path, **dataTrain)
            np.savez(validation_path, **dataValidation)
            # Pass loaded data objects directly, keeping the archives open
            # until fit() has finished reading their arrays.
            with np.load(train_path) as train_data, np.load(validation_path) as validation_data:
                archive_model.fit(train_data, validation_data, **settings)

        self.assertEqual(dictionary_model.history, archive_model.history)
        for name in PARAMETERS:
            np.testing.assert_array_equal(
                getattr(dictionary_model, name), getattr(archive_model, name)
            )

    def test_load_rejects_previous_standardized_archive(self):
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(prefix="nn_old_format_", dir=project_root) as directory:
            model_path = Path(directory) / "old_model.npz"
            np.savez(model_path, X_mean=[[40, 30, 2]], X_std=[[5, 5, 1]])
            with self.assertRaisesRegex(ValueError, "raw mm/GHz"):
                NeuralNetwork.load(model_path)


if __name__ == "__main__":
    unittest.main()
