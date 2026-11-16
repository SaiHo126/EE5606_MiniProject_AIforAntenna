"""Check the ReLU derivative, finite-difference gradients, training and storage.

Run: python -B -m unittest discover -s tests -p test_neural_network_relu.py -v
"""

from io import BytesIO
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.NeuralNetworkReLU import NeuralNetworkReLU


PARAMETERS = ("W_hidden", "B_hidden", "W_output", "B_output")
X = np.array([[38., 30., 1.6], [42., 28., 2.5], [34., 26., 4.], [45., 35., 3.1]])
Y = np.array([[2.4], [2.6], [2.7], [2.1]])


class ReLUTests(unittest.TestCase):
    def test_prediction_jacobian_matches_finite_differences(self):
        model = NeuralNetworkReLU(hiddenSize=3, seed=17)
        model.B_hidden[:] = -(model.W_hidden @ X.mean(axis=0).reshape(3, 1))
        _, cache = model.forward(X)
        jacobian = model.prediction_jacobian(cache)
        parameters = model.parameter_vector().copy()
        numerical = np.empty_like(jacobian)
        for column in range(len(parameters)):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[column] += 1e-6
            minus[column] -= 1e-6
            model.set_parameter_vector(plus)
            y_plus = model.forward(X)[0].ravel()
            model.set_parameter_vector(minus)
            y_minus = model.forward(X)[0].ravel()
            numerical[:, column] = (y_plus - y_minus) / 2e-6
        model.set_parameter_vector(parameters)
        np.testing.assert_allclose(jacobian, numerical, rtol=1e-5, atol=1e-7)
        # The loss gradient must also equal 2 J.T @ residual / N.
        prediction, cache = model.forward(X)
        gradients = model.backward(prediction, Y, cache)
        actual = np.concatenate([gradients['d' + name].ravel() for name in PARAMETERS])
        np.testing.assert_allclose(actual, 2 * jacobian.T @ (prediction - Y).ravel() / len(X))

    def test_lm_fit_raw_inputs_and_checkpoint_roundtrip(self):
        model = NeuralNetworkReLU(hiddenSize=8, seed=1)
        target = 2 + 0.01 * X[:, :1] - 0.02 * X[:, 1:2] + 0.1 * X[:, 2:3]
        data = {'X_mm': X.copy(), 'Y_GHz': target.copy()}
        model.fit_lm(data, data, epochs=50, target_rmse_kHz=10, progress_every=0)
        self.assertLess(model.loss(model.forward(X)[0], target), 1e-10)
        self.assertEqual(model.stop_reason, 'validation_target')
        np.testing.assert_array_equal(data['X_mm'], X)
        np.testing.assert_array_equal(data['Y_GHz'], target)
        archive = BytesIO()
        model.save(archive)
        archive.seek(0)
        restored = NeuralNetworkReLU.load(archive)
        self.assertEqual(restored.training_settings, model.training_settings)
        np.testing.assert_array_equal(restored.forward(X)[0], model.forward(X)[0])

    def test_refinement_preserves_predictions_before_output_fit(self):
        model = NeuralNetworkReLU(hiddenSize=8, seed=1)
        target = 2 + 0.01 * X[:, :1] - 0.02 * X[:, 1:2] + 0.1 * X[:, 2:3]
        data = {'X_mm': X, 'Y_GHz': target}
        model.fit_lm(data, data, epochs=50, target_rmse_kHz=10, progress_every=0)
        original = model.parameter_vector().copy()
        refined = model.refine_hidden_layer(factor=4)
        self.assertGreater(refined.hiddenSize, model.hiddenSize)
        np.testing.assert_allclose(refined.forward(X)[0], model.forward(X)[0], atol=1e-13)
        refined.fit_output(data, data)
        self.assertLess(refined.loss(refined.forward(X)[0], target), 1e-10)
        np.testing.assert_array_equal(model.parameter_vector(), original)
        self.assertEqual(refined.activationHidden, 'relu')
        archive = BytesIO()
        refined.save(archive)
        archive.seek(0)
        restored = NeuralNetworkReLU.load(archive)
        np.testing.assert_array_equal(restored.forward(X)[0], refined.forward(X)[0])
        self.assertEqual(restored.training_settings, refined.training_settings)

    def test_relu_forward_keeps_output_linear(self):
        model = NeuralNetworkReLU(hiddenSize=1, seed=1)
        model.W_hidden[:] = [[1., 0., 0.]]
        model.B_hidden[:] = 0
        model.W_output[:] = -2
        model.B_output[:] = 0.5
        prediction, cache = model.forward(np.array([[2., 0., 0.], [-2., 0., 0.]]))
        np.testing.assert_array_equal(cache['A_hidden'], [[2., 0.]])
        np.testing.assert_array_equal(prediction, [[-3.5], [0.5]])

    def test_derivative_is_zero_for_negative_and_zero_hidden_inputs(self):
        model = NeuralNetworkReLU(hiddenSize=3, seed=1)
        model.W_hidden[:] = 0
        model.B_hidden[:] = [[1.], [-1.], [0.]]
        model.W_output[:] = 1
        model.B_output[:] = 0
        prediction, cache = model.forward(X[:1])
        gradients = model.backward(prediction, np.zeros((1, 1)), cache)
        np.testing.assert_array_equal(gradients['dB_hidden'], [[2.], [0.], [0.]])
        np.testing.assert_array_equal(gradients['dW_output'], [[2., 0., 0.]])

    def test_gradients_match_centred_finite_differences_away_from_relu_corner(self):
        for hidden_size in [2, 8, 16]:
            with self.subTest(hidden_size=hidden_size):
                model = NeuralNetworkReLU(hiddenSize=hidden_size, seed=17)
                # Centre pre-activations for the test only, exercising both
                # active and inactive branches for each neuron.
                model.B_hidden[:] = -(model.W_hidden @ X.mean(axis=0).reshape(3, 1))
                prediction, cache = model.forward(X)
                epsilon = 1e-6
                # Perturbations must not cross the nondifferentiable point.
                self.assertGreater(float(np.min(abs(cache['Z_hidden']))), 2 * epsilon * X.max())
                self.assertTrue(np.all(np.any(cache['Z_hidden'] > 0, axis=1)))
                self.assertTrue(np.all(np.any(cache['Z_hidden'] < 0, axis=1)))
                gradients = model.backward(prediction, Y, cache)
                max_error = 0.0
                for name in PARAMETERS:
                    parameter = getattr(model, name)
                    numerical = np.zeros_like(parameter)
                    self.assertEqual(gradients['d' + name].shape, parameter.shape)
                    for index in np.ndindex(parameter.shape):
                        original = parameter[index]
                        parameter[index] = original + epsilon
                        plus = model.loss(model.forward(X)[0], Y)
                        parameter[index] = original - epsilon
                        minus = model.loss(model.forward(X)[0], Y)
                        parameter[index] = original
                        numerical[index] = (plus - minus) / (2 * epsilon)
                    np.testing.assert_allclose(gradients['d' + name], numerical, rtol=1e-5, atol=1e-7)
                    max_error = max(max_error, float(np.max(abs(gradients['d' + name] - numerical))))
                print(f'ReLU hidden={hidden_size}: maximum gradient difference={max_error:.3e}')

    def test_training_checkpoint_and_save_load_roundtrip(self):
        model = NeuralNetworkReLU(hiddenSize=8, seed=1)
        train = {'X_mm': X, 'Y_GHz': Y}
        validation = {'X_mm': X[:2], 'Y_GHz': Y[:2]}
        initial_loss = model.loss(model.forward(X)[0], Y)
        model.fit(train, validation, epochs=5, learning_rate=1e-5, batch_size=3)
        self.assertEqual(len(model.history['training_loss']), 5)
        self.assertTrue(np.isfinite(model.history['validation_loss']).all())
        self.assertLess(model.loss(model.forward(X)[0], Y), initial_loss)
        self.assertAlmostEqual(
            model.loss(model.forward(X[:2])[0], Y[:2]),
            min(model.history['validation_loss']), places=14,
        )
        archive = BytesIO()
        model.save(archive)
        archive.seek(0)
        restored = NeuralNetworkReLU.load(archive)
        self.assertEqual(restored.activationHidden, 'relu')
        self.assertEqual(restored.history, model.history)
        self.assertEqual(restored.training_settings, model.training_settings)
        np.testing.assert_array_equal(restored.forward(X)[0], model.forward(X)[0])
        self.assertEqual(restored.predict(X[0]), model.predict(X[0]))

    def test_rejects_tanh_or_unlabelled_model_archives(self):
        for activation in [None, 'tanh']:
            archive = BytesIO()
            fields = {'data_representation': 'mm_GHz'}
            if activation is not None:
                fields['activation_hidden'] = activation
            np.savez(archive, **fields)
            archive.seek(0)
            with self.assertRaisesRegex(ValueError, 'NeuralNetworkReLU'):
                NeuralNetworkReLU.load(archive)

    def test_epoch_limit_and_optional_percentage_stopping(self):
        data = {'X_mm': X, 'Y_GHz': Y}
        model = NeuralNetworkReLU(hiddenSize=8, seed=1)
        model.fit(data, data, epochs=1, learning_rate=0)
        self.assertEqual(len(model.history['validation_loss']), 1)
        self.assertEqual(model.stop_reason, 'epoch_limit')
        # No updates -> exactly unchanged loss. Three small changes require
        # four observations, so patience=3 must stop at epoch 4.
        model.fit(data, data, epochs=10, learning_rate=0,
                  tolerance_percent=0.1, patience=3)
        self.assertEqual(len(model.history['validation_loss']), 4)
        self.assertEqual(model.stop_reason, 'validation_change')

    def test_plot_contains_both_loss_histories(self):
        import matplotlib
        matplotlib.use('Agg')
        model = NeuralNetworkReLU(hiddenSize=8, seed=1)
        data = {'X_mm': X, 'Y_GHz': Y}
        model.fit(data, data, epochs=3, learning_rate=1e-5)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'curve.png'
            figure = model.plot_history(path)
            self.assertGreater(path.stat().st_size, 1000)
            lines = figure.axes[0].lines
            np.testing.assert_array_equal(lines[0].get_ydata(), model.history['training_loss'])
            np.testing.assert_array_equal(lines[1].get_ydata(), model.history['validation_loss'])


@unittest.skipUnless(find_spec('cupy') is not None, 'Optional CuPy backend is not installed')
class GPUReLUTests(unittest.TestCase):
    def test_gpu_lm_and_output_refinement(self):
        import cupy as cp
        model = NeuralNetworkReLU(hiddenSize=8, seed=1, device='gpu')
        target = 2 + 0.01 * X[:, :1] - 0.02 * X[:, 1:2] + 0.1 * X[:, 2:3]
        data = {'X_mm': X, 'Y_GHz': target}
        model.fit_lm(data, data, epochs=50, target_rmse_kHz=10, progress_every=0)
        refined = model.refine_hidden_layer(factor=4)
        refined.fit_output(data, data)
        self.assertIsInstance(refined.W_hidden, cp.ndarray)
        self.assertLess(refined.loss(refined.forward(X)[0], target), 1e-10)

    def test_cpu_gpu_forward_gradients_training_and_portable_checkpoint(self):
        import cupy as cp
        # float64 makes differences easier to distinguish from float32 rounding.
        cpu = NeuralNetworkReLU(hiddenSize=16, seed=17)
        gpu = NeuralNetworkReLU(hiddenSize=16, seed=17, device='gpu')
        for model in (cpu, gpu):
            model.B_hidden[:] = -(model.W_hidden @ model.xp.asarray(X.mean(axis=0).reshape(3, 1)))
        cpu_prediction, cpu_cache = cpu.forward(X)
        gpu_prediction, gpu_cache = gpu.forward(X)
        self.assertIsInstance(gpu_prediction, cp.ndarray)
        np.testing.assert_allclose(gpu.to_numpy(gpu_prediction), cpu_prediction, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(gpu.to_numpy(gpu.prediction_jacobian(gpu_cache)),
                                   cpu.prediction_jacobian(cpu_cache), rtol=1e-11, atol=1e-12)
        cpu_gradients = cpu.backward(cpu_prediction, Y, cpu_cache)
        gpu_gradients = gpu.backward(gpu_prediction, Y, gpu_cache)
        for name in cpu_gradients:
            self.assertIsInstance(gpu_gradients[name], cp.ndarray)
            np.testing.assert_allclose(gpu.to_numpy(gpu_gradients[name]), cpu_gradients[name],
                                       rtol=1e-11, atol=1e-12)
        data = {'X_mm': X, 'Y_GHz': Y}
        for model in (cpu, gpu):
            model.fit(data, data, epochs=5, learning_rate=1e-5, batch_size=3)
        np.testing.assert_allclose(gpu.history['training_loss'], cpu.history['training_loss'],
                                   rtol=1e-11, atol=1e-12)
        for name in PARAMETERS:
            self.assertIsInstance(getattr(gpu, name), cp.ndarray)
            np.testing.assert_allclose(gpu.to_numpy(getattr(gpu, name)), getattr(cpu, name),
                                       rtol=1e-11, atol=1e-12)
        archive = BytesIO()
        gpu.save(archive)
        for device in ('cpu', 'gpu'):
            archive.seek(0)
            restored = NeuralNetworkReLU.load(archive, device=device)
            self.assertEqual(restored.device, device)
            self.assertAlmostEqual(restored.predict(X[0]), gpu.predict(X[0]), places=12)
            self.assertEqual(restored.history, gpu.history)


if __name__ == '__main__':
    unittest.main()
