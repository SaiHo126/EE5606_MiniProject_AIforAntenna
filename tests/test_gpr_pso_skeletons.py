"""Numerical and integration checks: python -B -m unittest discover -s tests -p test_gpr_pso_skeletons.py -v"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.GPRskeleton import GaussianProcessRegressor, load_prepared_split
from src.PSOskeleton import (PSOConfig, ParticleSwarmOptimizer, geometry_is_feasible,
                             repair_antenna_design, make_antenna_objective, verify_design)
from src.PatchFormula import RMA, AntennaBounds_mm


class GPTests(unittest.TestCase):
    def setUp(self):
        self.train = {"X_mm": np.array([[30., 25., 1.], [50., 35., 3.]]),
                      "Y_GHz": np.array([[2.], [4.]])}
        self.validation = {"X_mm": np.array([[40., 30., 2.]]), "Y_GHz": np.array([[3.]])}

    def fitted(self):
        model = GaussianProcessRegressor()
        model.fit(self.train, self.validation, length_scales=[1.], noise_stds=[0.])
        return model

    def test_kernel_symmetry_and_known_values(self):
        X = np.array([[0., 0., 0.], [1., 0., 0.]])
        K = GaussianProcessRegressor.kernel(X, X, [1., 2., 3.], 2.)
        np.testing.assert_allclose(K, [[4., 4*np.exp(-0.5)], [4*np.exp(-0.5), 4.]])
        np.testing.assert_array_equal(K, K.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(K) > 0))

    def test_posterior_matches_two_point_closed_form(self):
        model = self.fitted()
        mean, std = model.predict([40., 30., 2.], return_std=True)
        # Standardized training rows are [-1,-1,-1], [1,1,1].
        expected_var = 1 - 2*np.exp(-3.)/(1 + model.jitter_used + np.exp(-6.))
        self.assertAlmostEqual(mean, 3., places=12)
        self.assertAlmostEqual(std**2, expected_var, places=12)
        np.testing.assert_allclose(model.predict(self.train['X_mm']), self.train['Y_GHz'], atol=1e-8)
        covariance = model.kernel(model.X_train, model.X_train) + model.jitter_used*np.eye(2)
        np.testing.assert_allclose(model.L @ model.L.T, covariance)
        np.testing.assert_allclose(covariance @ model.alpha, model.Y_train)

    def test_scaling_uses_training_only_and_selection_uses_validation(self):
        model = GaussianProcessRegressor()
        validation = {"X_mm": np.array([[100., 90., 10.]]), "Y_GHz": np.array([[2.2]])}
        history = model.fit(self.train, validation, length_scales=[0.5, 1., 4.])
        np.testing.assert_allclose(model.X_mean, [40., 30., 2.])
        np.testing.assert_allclose(model.Y_mean, [3.])
        self.assertEqual(model.best_validation_rmse_MHz, min(row['validation_rmse_MHz'] for row in history))
        self.assertEqual(len(model.X_train), 2)

    def test_duplicate_inputs_and_constant_target_remain_stable(self):
        data = {"X_mm": np.tile([38., 30., 1.6], (4, 1)), "Y_GHz": np.full((4, 1), 2.4)}
        model = GaussianProcessRegressor()
        model.fit(data, data, length_scales=[2.], noise_stds=[0.])
        mean, std = model.predict(data['X_mm'], return_std=True)
        np.testing.assert_allclose(mean, 2.4)
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertTrue(np.all(std >= 0))

    def test_roundtrip_preserves_mean_std_and_metadata(self):
        model = self.fitted()
        model.metadata = {"split_seed": 1, "train_indices": [0, 2]}
        with TemporaryDirectory() as directory:
            filename = Path(directory) / 'gp.npz'
            model.save(filename)
            loaded = GaussianProcessRegressor.load(filename)
        for actual, expected in zip(loaded.predict(self.train['X_mm'], True), model.predict(self.train['X_mm'], True)):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(loaded.metadata, model.metadata)
        self.assertEqual(loaded.history, model.history)

    def test_invalid_shapes_and_unfitted_prediction(self):
        with self.assertRaises(RuntimeError):
            GaussianProcessRegressor().predict([38., 30., 1.6])
        model = self.fitted()
        for invalid in ([1, 2], [[1, 2, np.nan]], np.zeros((3, 1))):
            with self.assertRaises(ValueError):
                model.predict(invalid)

    def test_overlapping_split_is_rejected(self):
        with TemporaryDirectory() as directory:
            filename = Path(directory) / 'data.npz'
            np.savez(filename, X_mm=np.ones((3, 3)), Y_GHz=np.ones((3, 1)),
                     train_indices=[0], validation_indices=[0], test_indices=[2])
            with self.assertRaises(ValueError):
                load_prepared_split(filename)


class PSOTests(unittest.TestCase):
    def test_rma_geometry_fix_refreshes_all_derived_values(self):
        antenna = RMA(25., 45., 1.)
        old_volume = antenna.volume_mm3
        old_frequency = antenna.ResonanceF_GHz
        antenna.geometryFix()
        self.assertTrue(antenna.geometryCheck())
        self.assertNotEqual(antenna.volume_mm3, old_volume)
        self.assertNotEqual(antenna.ResonanceF_GHz, old_frequency)
        antenna.thickness_mm = 2.
        # Refresh even when manually changed dimensions already satisfy bounds.
        antenna.geometryFix()
        reference = RMA(antenna.width_mm, antenna.length_mm, antenna.thickness_mm)
        for name in ('volume_mm3', 'RelativePermittivityEff', 'lengthDelta_mm',
                     'lengthEff_mm', 'ResonanceF_GHz'):
            self.assertEqual(getattr(antenna, name), getattr(reference, name))

    def test_known_sphere_convergence_and_reproducibility(self):
        objective = lambda x: float(np.sum((x - [0.3, -0.2])**2))
        config = PSOConfig(swarm_size=30, iterations=120, seed=7)
        optimizer = ParticleSwarmOptimizer(objective, [[-2., 2.], [-2., 2.]], config)
        result, repeat = optimizer.run(), optimizer.run()
        self.assertLess(result.best_score, 1e-10)
        np.testing.assert_array_equal(result.best_score_history, repeat.best_score_history)
        self.assertEqual(len(result.best_score_history), 121)
        self.assertEqual(result.objective_evaluations, 30 * 121)
        self.assertTrue(np.all(np.diff(result.best_score_history) <= 0))
        np.testing.assert_allclose([objective(x) for x in result.best_position_history], result.best_score_history)
        np.testing.assert_array_equal(result.best_position, result.best_position_history[-1])
        self.assertEqual(result.best_score, result.best_score_history[-1])

    def test_zero_iterations_includes_initial_best(self):
        result = ParticleSwarmOptimizer(lambda x: float(x @ x), [[-1, 1]], PSOConfig(iterations=0)).run()
        self.assertEqual(len(result.best_score_history), 1)
        self.assertEqual(result.objective_evaluations, 40)

    def test_boundary_and_random_repairs(self):
        examples = [[25, 45, 1], [50, 45, 5], [30, 25, 1], [50, 50/1.2, 5], [-1, -1, -1]]
        examples.extend(np.random.default_rng(0).uniform(-100, 100, (1000, 3)))
        self.assertFalse(geometry_is_feasible([25, 45, 1]))  # Inside box but invalid ratio.
        for example in examples:
            original = np.array(example, dtype=float)
            repaired = repair_antenna_design(original)
            self.assertTrue(geometry_is_feasible(repaired))
            np.testing.assert_array_equal(original, example)
            np.testing.assert_array_equal(repair_antenna_design(repaired), repaired)

    def test_every_objective_evaluation_has_feasible_geometry(self):
        visited = []
        def objective(x):
            self.assertTrue(geometry_is_feasible(x))
            visited.append(x.copy())
            return float(np.sum((x - [38., 30., 1.6])**2))
        result = ParticleSwarmOptimizer(objective, AntennaBounds_mm,
                                       PSOConfig(iterations=80, seed=2), repair_antenna_design).run()
        self.assertEqual(len(visited), result.objective_evaluations)
        self.assertLess(result.best_score, 1e-5)

    def test_objective_matches_existing_fitness(self):
        x = np.array([38., 30., 1.6])
        self.assertAlmostEqual(RMA(*x).ResonanceF_GHz, 2.357548542, places=9)
        self.assertEqual(RMA(*x).volume_mm3, 1824.)
        for penalty in [0., 1., 10.]:
            objective = make_antenna_objective(lambda design: RMA(*design).ResonanceF_GHz, penalty)
            self.assertAlmostEqual(objective(x), RMA(*x).fitness(penalty))

    def test_search_never_calls_analytical_formula_and_verification_separates_errors(self):
        surrogate = lambda x: 2.4
        objective = make_antenna_objective(surrogate)
        with patch('src.PSOskeleton.RMA', side_effect=AssertionError('Physics called during search')):
            result = ParticleSwarmOptimizer(objective, AntennaBounds_mm,
                                           PSOConfig(iterations=5), repair_antenna_design).run()
        checked = verify_design([38., 30., 1.6], surrogate, tolerance_GHz=0.01)
        self.assertEqual(checked['surrogate_target_deviation_MHz'], 0.)
        self.assertGreater(checked['analytical_target_deviation_MHz'], 42.)
        self.assertFalse(checked['success'])
        self.assertTrue(geometry_is_feasible(result.best_position))

    def test_invalid_objective_and_config_fail_clearly(self):
        for value in [np.nan, -np.inf, np.inf]:
            with self.assertRaises(ValueError):
                ParticleSwarmOptimizer(lambda x: value, [[0, 1]]).run()
        with self.assertRaises(ValueError):
            PSOConfig(iterations=-1)


if __name__ == '__main__':
    unittest.main()
