"""Numerical checks for the experimental multilayer, handwritten NN."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.neuralnetworktest import NeuralNetwork, audit_designs, rmse_khz


class MultilayerTests(unittest.TestCase):
    def test_derivatives_with_one_two_and_three_hidden_layers(self):
        rng = np.random.default_rng(42)
        for sizes in [(3,), (3, 2), (3, 2, 2)]:
            with self.subTest(sizes=sizes):
                model = NeuralNetwork(sizes, seed=7, use_gpu=False)
                X, Y = rng.normal(size=(5, 3)), rng.normal(size=(5, 1))
                prediction, cache = model.forward(X)
                J = model.jacobian(cache)
                original = model.parameters().copy()
                numerical = np.empty_like(J)
                epsilon = 1e-6
                for index in range(len(original)):
                    plus = original.copy()
                    plus[index] += epsilon
                    model.set_parameters(plus)
                    upper = model.forward(X)[0]
                    minus = original.copy()
                    minus[index] -= epsilon
                    model.set_parameters(minus)
                    lower = model.forward(X)[0]
                    numerical[:, index] = ((upper - lower) / (2 * epsilon)).ravel()
                model.set_parameters(original)
                np.testing.assert_allclose(J, numerical, rtol=1e-5, atol=1e-8)
                gradients = np.concatenate([a.ravel() for pair in
                    model.backward(prediction, Y, cache) for a in pair])
                np.testing.assert_allclose(gradients,
                    2 / len(Y) * numerical.T @ (prediction - Y).ravel(),
                    rtol=1e-5, atol=1e-8)

    def test_training_scaling_best_parameters_and_portable_save(self):
        X = audit_designs(40, seed=17)
        Y = 2.4 + (X - [40, 30, 2]) @ np.array([[0.01], [-0.02], [0.03]])
        train = {"X_mm": X[:30], "Y_GHz": Y[:30]}
        validation = {"X_mm": X[30:], "Y_GHz": Y[30:]}
        model = NeuralNetwork((4, 3), seed=2, use_gpu=False)
        history = model.fit(train, validation, epochs=40, target_khz=0)
        np.testing.assert_allclose(model.X_mean, X[:30].mean(axis=0))
        np.testing.assert_allclose(model.X_std, X[:30].std(axis=0))
        np.testing.assert_allclose(model.Y_mean, Y[:30].mean(axis=0))
        self.assertLess(history["training_loss"][-1], history["training_loss"][0])
        mse = np.mean((model.predict(X[30:]) - Y[30:]) ** 2)
        self.assertAlmostEqual(mse, min(history["validation_loss"]), places=12)
        self.assertIsInstance(model.predict(X[0]), float)
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=root / "tmp") as directory:
            filename = Path(directory) / "model.npz"
            model.save(filename)
            restored = NeuralNetwork.load(filename, use_gpu=False)
        np.testing.assert_array_equal(model.predict(X), restored.predict(X))
        self.assertEqual(model.history, restored.history)
        self.assertEqual(model.training_settings, restored.training_settings)

    def test_units_and_audit_constraints(self):
        self.assertAlmostEqual(rmse_khz(np.array([[2.40001]]), np.array([[2.4]])), 10)
        X = audit_designs(100, seed=3)
        self.assertEqual(X.shape, (100, 3))
        self.assertTrue(np.all(X >= [25, 25, 1]))
        self.assertTrue(np.all(X <= [50, 45, 5]))
        self.assertTrue(np.all(X[:, 0] / X[:, 1] >= 1.2))


if __name__ == "__main__":
    unittest.main()
