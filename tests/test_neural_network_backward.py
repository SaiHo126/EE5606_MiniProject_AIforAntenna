"""Numerical checks of the skeleton's implemented backward() method.

Run from the project root:
    python -B -m unittest discover -s tests -p test_neural_network_backward.py -v

The independent forward/MSE calculations below provide a numerical reference
for checking the implemented forward, backward and parameter update methods.
"""

import unittest

import numpy as np

from src.NeuralNetworkSkeleton import NeuralNetwork


PARAMETERS = ("W_hidden", "B_hidden", "W_output", "B_output")


def make_model(hidden_size=8):
    model = NeuralNetwork(hidden_size=hidden_size, seed=11)
    # Check gradients away from zero biases as well as random weights.
    model.B_hidden[:] = np.linspace(-0.2, 0.3, hidden_size).reshape(-1, 1)
    model.B_output[:] = 0.15
    return model


def reference_forward(model, X_mm):
    X = X_mm.T
    Z_hidden = model.W_hidden @ X + model.B_hidden
    A_hidden = np.tanh(Z_hidden)
    Y_pred = (model.W_output @ A_hidden + model.B_output).T
    return Y_pred, {"X": X, "Z_hidden": Z_hidden, "A_hidden": A_hidden}


def reference_mse(model, X_mm, Y_actual):
    Y_pred, _ = reference_forward(model, X_mm)
    return float(np.mean((Y_pred - Y_actual)**2))


class BackwardTests(unittest.TestCase):
    def test_all_gradients_match_centred_finite_differences(self):
        # Include a tiny network, one sample, and batches of different sizes.
        for hidden_size, batch_size in [(2, 3), (8, 1), (8, 4), (8, 8)]:
            with self.subTest(hidden_size=hidden_size, batch_size=batch_size):
                model = make_model(hidden_size)
                rng = np.random.default_rng(23)
                X_mm = rng.uniform([25, 25, 1], [50, 45, 5], size=(batch_size, 3))
                Y_actual = rng.uniform(2.0, 3.0, size=(batch_size, 1))
                Y_pred, cache = model.forward(X_mm)
                before = {name: getattr(model, name).copy() for name in PARAMETERS}

                gradients = model.backward(Y_pred, Y_actual, cache)
                self.assertEqual(set(gradients), set(PARAMETERS))
                epsilon = 1e-6
                max_error = 0.0
                for name in PARAMETERS:
                    parameter = getattr(model, name)
                    # backward must not update parameters before propagation ends.
                    np.testing.assert_array_equal(parameter, before[name])
                    self.assertEqual(gradients[name].shape, parameter.shape)
                    numerical = np.zeros_like(parameter)
                    for index in np.ndindex(parameter.shape):
                        original = parameter[index]
                        parameter[index] = original + epsilon
                        loss_plus = reference_mse(model, X_mm, Y_actual)
                        parameter[index] = original - epsilon
                        loss_minus = reference_mse(model, X_mm, Y_actual)
                        parameter[index] = original
                        numerical[index] = (loss_plus - loss_minus) / (2 * epsilon)

                    np.testing.assert_allclose(
                        gradients[name], numerical, rtol=1e-5, atol=1e-8,
                        err_msg=name,
                    )
                    max_error = max(max_error, float(np.max(abs(gradients[name] - numerical))))
                print(
                    f"hidden={hidden_size}, samples={batch_size}: "
                    f"maximum gradient error={max_error:.3e}"
                )

    def test_repeating_samples_does_not_change_mean_loss_gradients(self):
        model = make_model()
        rng = np.random.default_rng(31)
        X_mm = rng.uniform([25, 25, 1], [50, 45, 5], size=(3, 3))
        Y_actual = rng.normal(size=(3, 1))
        Y_pred, cache = model.forward(X_mm)
        original = model.backward(Y_pred, Y_actual, cache)

        repeated_X = np.repeat(X_mm, 4, axis=0)
        repeated_Y = np.repeat(Y_actual, 4, axis=0)
        Y_pred, cache = model.forward(repeated_X)
        repeated = model.backward(Y_pred, repeated_Y, cache)
        for name in PARAMETERS:
            np.testing.assert_allclose(original[name], repeated[name], atol=1e-12)

    def test_small_gradient_descent_step_reduces_mse(self):
        model = make_model()
        rng = np.random.default_rng(41)
        X_mm = rng.uniform([25, 25, 1], [50, 45, 5], size=(4, 3))
        Y_actual = rng.uniform(2.0, 3.0, size=(4, 1))
        loss_before = reference_mse(model, X_mm, Y_actual)
        Y_pred, cache = model.forward(X_mm)
        gradients = model.backward(Y_pred, Y_actual, cache)

        model.update_parameters(gradients, learning_rate=1e-5)
        self.assertLess(reference_mse(model, X_mm, Y_actual), loss_before)


if __name__ == "__main__":
    unittest.main()
