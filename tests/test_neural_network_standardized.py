"""Checks for scaling and stopping in the original single-hidden-layer NN."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src import NerualNetwork as nn
from src.PSOskeleton import load_neural_network


class StandardizedNetworkTests(unittest.TestCase):
    def setUp(self):
        self.backend = patch.object(nn, "xp", np)
        self.backend.start()
        self.addCleanup(self.backend.stop)
        rng = np.random.default_rng(17)
        X = rng.uniform([30, 25, 1], [50, 40, 5], size=(20, 3))
        Y = 2.4 + (X - [40, 30, 2]) @ np.array([[0.01], [-0.02], [0.03]])
        self.train = {"X_mm": X[:15], "Y_GHz": Y[:15]}
        self.validation = {"X_mm": X[15:], "Y_GHz": Y[15:]}

    def test_scaling_roundtrip_physical_loss_and_pso_loader(self):
        model = nn.NerualNetwork(32, 1)
        history = model.fit(self.train, self.validation, epochs=20, target_rmse_khz=0)
        for prefix, key in (("X", "X_mm"), ("Y", "Y_GHz")):
            np.testing.assert_allclose(getattr(model, prefix + "_mean"), self.train[key].mean(axis=0, keepdims=True))
            np.testing.assert_allclose(getattr(model, prefix + "_std"), self.train[key].std(axis=0, keepdims=True))
        predictions = np.array([model.predict(x) for x in self.validation["X_mm"]]).reshape(-1, 1)
        mse = np.mean((predictions - self.validation["Y_GHz"]) ** 2)
        self.assertAlmostEqual(mse, min(history["validation_loss"]), places=12)
        self.assertEqual(model.W_hidden.shape, (32, 3))
        self.assertEqual(model.activationHidden, "tanh")
        self.assertEqual(model.training_settings["learning_rate"], 1.0)
        self.assertTrue(model.training_settings["line_search"])
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=root / "tmp") as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            with np.load(path) as data:
                self.assertEqual(data["data_representation"].item(), "scaled_mm_GHz")
            for restored in (nn.NerualNetwork.load(path), load_neural_network(path)):
                np.testing.assert_array_equal(predictions.ravel(), [restored.predict(x) for x in self.validation["X_mm"]])
                self.assertEqual(model.history, restored.history)
                self.assertEqual(model.training_settings, restored.training_settings)
                self.assertEqual(model.step_sizes, restored.step_sizes)
                self.assertEqual(model.stop_reason, restored.stop_reason)

    def test_epoch_cap_target_and_patience(self):
        for kwargs, expected in [
            ({"epochs": 1, "target_rmse_khz": 0}, 1),
            ({"epochs": 20, "target_rmse_khz": 1e9}, 1),
            ({"epochs": 20, "learning_rate": 0, "line_search": False, "target_rmse_khz": 0, "patience": 2}, 3),
        ]:
            with self.subTest(kwargs=kwargs):
                model = nn.NerualNetwork(3, 1)
                history = model.fit(self.train, self.validation, **kwargs)
                self.assertEqual(len(history["training_loss"]), expected)

    def test_constant_data_and_legacy_archive(self):
        model = nn.NerualNetwork(3, 1)
        constant = {"X_mm": np.tile([38., 30., 1.6], (4, 1)), "Y_GHz": np.full((4, 1), 2.4)}
        model.fit(constant, constant, epochs=2)
        self.assertTrue(np.all(model.X_std == 1))
        self.assertTrue(np.all(model.Y_std == 1))
        self.assertAlmostEqual(model.predict([38., 30., 1.6]), 2.4)
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=root / "tmp") as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(path, W_hidden=model.W_hidden, B_hidden=model.B_hidden,
                     W_output=model.W_output, B_output=model.B_output,
                     data_representation="mm_GHz", hidden_size=3, seed=1,
                     best_epoch=1, epochs=2, learning_rate=1e-5, batch_size=32,
                     training_loss=[1.], validation_loss=[1.])
            restored = nn.NerualNetwork.load(path)
        X = np.array([[38., 30., 1.6]])
        expected = (model.W_output @ np.tanh(model.W_hidden @ X.T + model.B_hidden) + model.B_output)[0, 0]
        self.assertAlmostEqual(restored.predict(X[0]), expected)

    def test_scaled_backpropagation_matches_finite_differences(self):
        model = nn.NerualNetwork(3, 1)
        model.fit(self.train, self.validation, epochs=1, target_rmse_khz=0)
        X = (self.train["X_mm"][:4] - model.X_mean) / model.X_std
        Y = (self.train["Y_GHz"][:4] - model.Y_mean) / model.Y_std
        prediction, cache = model.forward(X)
        gradients = model.backward(prediction, Y, cache)
        for name in ("W_hidden", "B_hidden", "W_output", "B_output"):
            parameter = getattr(model, name)
            numerical = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + 1e-6
                plus = model.loss(model.forward(X)[0], Y)
                parameter[index] = original - 1e-6
                minus = model.loss(model.forward(X)[0], Y)
                parameter[index] = original
                numerical[index] = (plus - minus) / 2e-6
            np.testing.assert_allclose(gradients["d" + name], numerical, rtol=1e-5, atol=1e-8)

    def test_network_step_obeys_both_wolfe_conditions_and_is_not_cumulative(self):
        model = nn.NerualNetwork(32, 1)
        X = np.array([[-1., 0.5, 0.3], [0.2, -0.3, 0.1]])
        Y = np.array([[0.7], [-0.2]])
        prediction, cache = model.forward(X)
        gradients = model.backward(prediction, Y, cache)
        names = ("W_hidden", "B_hidden", "W_output", "B_output")
        before = {name: getattr(model, name).copy() for name in names}
        y0 = model.loss(prediction, Y)
        g0 = -sum(np.sum(g*g) for g in gradients.values())
        alpha = model.line_search_step(X, Y, gradients, 10., 1e-4, 0.1, 40)
        prediction, cache = model.forward(X)
        after_gradients = model.backward(prediction, Y, cache)
        g = -sum(np.sum(after_gradients[k]*gradients[k]) for k in gradients)
        self.assertLessEqual(model.loss(prediction, Y), y0 + 1e-4*alpha*g0)
        self.assertLessEqual(abs(g), -0.1*g0)
        for name in names:
            np.testing.assert_allclose(getattr(model, name), before[name]-alpha*gradients['d'+name])

    def test_search_failure_restores_parameters(self):
        model = nn.NerualNetwork(3, 1)
        X, Y = np.zeros((2, 3)), np.ones((2, 1))
        prediction, cache = model.forward(X)
        gradients = model.backward(prediction, Y, cache)
        before = {name: getattr(model, name).copy() for name in ("W_hidden", "B_hidden", "W_output", "B_output")}
        with self.assertRaises(RuntimeError):
            model.line_search_step(X, Y, gradients, 1e-10, 1e-4, 0.1, 1)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(model, name), value)


class LineSearchTests(unittest.TestCase):
    def test_slides_worked_example(self):
        # Slides 49-59: f(x)=x1^2+x1*x2+x2^2, x=[1,2], d=[-1,-1].
        def evaluate(alpha):
            x, y = 1-alpha, 2-alpha
            return x*x+x*y+y*y, -(2*x+y)-(x+2*y)
        alpha = nn.approximate_line_search(evaluate, initial_step=10, sigma=0.1)
        self.assertEqual(alpha, 1.5625)
        self.assertEqual(evaluate(alpha), (0.26171875, 0.375))

    def test_bracket_expands_a_too_small_initial_step(self):
        phi = lambda a: ((a-2)**2, 2*(a-2))
        alpha = nn.approximate_line_search(phi, initial_step=0.01)
        self.assertGreater(alpha, 0.01)
        self.assertLessEqual(phi(alpha)[0], 4 + 1e-4*alpha*(-4))
        self.assertLessEqual(abs(phi(alpha)[1]), 0.4)

    def test_nonfinite_trial_is_rejected(self):
        def phi(a):
            return (np.inf, np.nan) if a >= 1 else ((a-0.5)**2, 2*(a-0.5))
        self.assertEqual(nn.approximate_line_search(phi, initial_step=2), 0.5)

    def test_limits_and_descent_direction_are_checked(self):
        with self.assertRaises(ValueError):
            nn.approximate_line_search(lambda a: (a*a, 0))
        with self.assertRaises(ValueError):
            nn.approximate_line_search(lambda a: (-a, -1), beta=0.9, sigma=0.1)
        with self.assertRaises(RuntimeError):
            nn.approximate_line_search(lambda a: (-a, -1), max_iterations=3)


if __name__ == "__main__":
    unittest.main()
