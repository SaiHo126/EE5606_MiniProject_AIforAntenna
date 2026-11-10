"""Numerical and integration checks for the hand-written GA.

Run from the project root: python -B -m unittest discover -s tests -v
"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.GASkeleton import (
    ANTENNA_BOUNDS_MM,
    GAConfig,
    GeneticAlgorithm,
    antenna_objective,
    geometry_is_feasible,
    main,
    repair_antenna_design,
)
from src.PatchFormula import RMA


class GeneticAlgorithmTests(unittest.TestCase):
    def test_worksheet_reference_values_and_length_unit_scaling(self):
        antenna = RMA(38.0, 30.0, 1.6)
        scaled = RMA(76.0, 60.0, 3.2)
        self.assertAlmostEqual(antenna.ResonanceF / 1e9, 2.3575485421095275, places=10)
        self.assertAlmostEqual(antenna.volume_mm3, 1824.0)
        self.assertAlmostEqual(scaled.ResonanceF / antenna.ResonanceF, 0.5)
        self.assertAlmostEqual(scaled.volume_mm3 / antenna.volume_mm3, 8.0)

    def test_antenna_search_meets_frequency_and_reduces_volume(self):
        result = GeneticAlgorithm(
            antenna_objective, ANTENNA_BOUNDS_MM, repair_antenna_design, GAConfig(seed=42)
        ).run()
        antenna = RMA(*result.best_design_mm)
        self.assertTrue(geometry_is_feasible(result.best_design_mm))
        self.assertLess(abs(antenna.ResonanceF / 1e9 - 2.4) * 1000.0, 10.0)
        self.assertLess(antenna.volume_mm3, 1500.0)

    def make_sphere_optimizer(self, **settings):
        bounds = np.array([[-5.0, 5.0]] * 3)
        return GeneticAlgorithm(
            objective=lambda x: float(x @ x),  # Known global minimum: zero at [0,0,0].
            bounds=bounds,
            repair=lambda x: np.clip(x, bounds[:, 0], bounds[:, 1]),
            config=GAConfig(**settings),
        )

    def test_antenna_repair_enforces_coupled_constraint_without_aliasing(self):
        rng = np.random.default_rng(11)
        points = list(rng.uniform(-100.0, 100.0, size=(200, 3)))
        points += [np.array([25.0, 45.0, 1.0]), np.array([50.0, 50.0 / 1.2, 5.0])]
        for point in points:
            before = point.copy()
            repaired = repair_antenna_design(point)
            self.assertTrue(geometry_is_feasible(repaired), repaired)
            np.testing.assert_array_equal(point, before)
            self.assertFalse(np.shares_memory(point, repaired))
        self.assertFalse(geometry_is_feasible([25.0, 45.0, 1.0]))

    def test_initial_population_is_feasible_and_repeatable(self):
        def make():
            return GeneticAlgorithm(
                lambda x: float(np.sum(x)), ANTENNA_BOUNDS_MM,
                repair_antenna_design, GAConfig(seed=19),
            )
        first = make().initialize_population()
        second = make().initialize_population()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (60, 3))
        self.assertTrue(all(geometry_is_feasible(row) for row in first))

    def test_full_population_tournament_selects_best_and_returns_copy(self):
        optimizer = self.make_sphere_optimizer(population_size=4, tournament_size=4)
        population = np.arange(12.0).reshape(4, 3)
        parent = optimizer.select_parent(population, np.array([3.0, 1.0, 4.0, 2.0]))
        np.testing.assert_array_equal(parent, population[1])
        parent[:] = 0.0
        np.testing.assert_array_equal(population[1], [3.0, 4.0, 5.0])

    def test_crossover_and_mutation_preserve_parents_and_geometry(self):
        optimizer = GeneticAlgorithm(
            lambda x: float(np.sum(x)), ANTENNA_BOUNDS_MM, repair_antenna_design,
            GAConfig(crossover_probability=1.0, mutation_probability=1.0, mutation_scale=2.0),
        )
        parent_a = np.array([38.0, 30.0, 1.6])
        parent_b = np.array([50.0, 40.0, 4.0])
        for _ in range(50):
            children = optimizer.crossover(parent_a, parent_b)
            for child in children:
                before = child.copy()
                mutated = optimizer.mutate(child)
                self.assertTrue(geometry_is_feasible(child))
                self.assertTrue(geometry_is_feasible(mutated))
                np.testing.assert_array_equal(child, before)
                self.assertFalse(np.shares_memory(child, mutated))
        np.testing.assert_array_equal(parent_a, [38.0, 30.0, 1.6])
        np.testing.assert_array_equal(parent_b, [50.0, 40.0, 4.0])

    def test_zero_operator_probabilities_leave_independent_copies(self):
        optimizer = self.make_sphere_optimizer(crossover_probability=0.0, mutation_probability=0.0)
        first, second = np.ones(3), np.full(3, 2.0)
        child_a, child_b = optimizer.crossover(first, second)
        np.testing.assert_array_equal(child_a, first)
        np.testing.assert_array_equal(child_b, second)
        self.assertFalse(np.shares_memory(first, child_a))
        self.assertFalse(np.shares_memory(second, child_b))
        np.testing.assert_array_equal(optimizer.mutate(first), first)

    def test_odd_population_retains_elites_and_exact_size(self):
        optimizer = self.make_sphere_optimizer(population_size=7, elite_count=2)
        population = optimizer.initialize_population()
        scores = optimizer.evaluate_population(population)
        original = population.copy()
        next_population = optimizer.create_next_generation(population, scores)
        self.assertEqual(next_population.shape, (7, 3))
        np.testing.assert_array_equal(next_population[:2], population[np.argsort(scores)[:2]])
        np.testing.assert_array_equal(population, original)
        self.assertFalse(np.shares_memory(next_population, population))

    def test_search_converges_on_known_objective_and_is_reproducible(self):
        settings = dict(population_size=50, generations=100, mutation_probability=0.25, seed=7)
        first = self.make_sphere_optimizer(**settings).run()
        second = self.make_sphere_optimizer(**settings).run()
        self.assertLess(first.best_score, 0.05)
        self.assertLess(first.best_score, first.best_score_history[0] * 0.05)
        self.assertTrue(np.all(np.diff(first.best_score_history) <= 0.0))
        self.assertEqual(len(first.best_score_history), 101)
        self.assertEqual(first.objective_evaluations, 50 * 101)
        np.testing.assert_array_equal(first.best_design_mm, second.best_design_mm)
        self.assertEqual(first.best_score_history, second.best_score_history)

    def test_zero_generations_and_invalid_scores(self):
        optimizer = self.make_sphere_optimizer(population_size=5, generations=0)
        result = optimizer.run()
        self.assertEqual(result.objective_evaluations, 5)
        self.assertEqual(len(result.best_score_history), 1)
        optimizer.objective = lambda x: float("inf")
        with self.assertRaisesRegex(ValueError, "No design"):
            optimizer.run()
        optimizer.objective = lambda x: float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            optimizer.run()

    def test_invalid_settings_are_rejected(self):
        for kwargs in (
            {"population_size": 1}, {"generations": -1}, {"generations": 1.5},
            {"tournament_size": 61}, {"elite_count": 60}, {"seed": -1},
            {"mutation_probability": 1.1}, {"mutation_scale": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GAConfig(**kwargs)

    def test_cli_writes_results_history_and_honest_verification_status(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            exit_code = main([
                "--population-size", "7", "--generations", "3",
                "--output-dir", directory,
            ])
            summary = json.loads((Path(directory) / "ga_seed_42.json").read_text())
            with (Path(directory) / "ga_seed_42_history.csv").open(newline="") as handle:
                history = list(csv.DictReader(handle))
            self.assertEqual(len(history), 4)
            self.assertEqual(summary["objective_evaluations"], 7 * 4)
            self.assertTrue(summary["geometry_feasible"])
            self.assertAlmostEqual(
                summary["frequency_error_mhz"], abs(summary["frequency_ghz"] - 2.4) * 1000.0
            )
            accepted = (
                summary["geometry_feasible"] and summary["model_reference_passed"]
                and summary["frequency_error_mhz"] <= summary["frequency_tolerance_mhz"]
            )
            self.assertEqual(summary["success"], accepted)
            self.assertEqual(exit_code, 0 if accepted else 1)


if __name__ == "__main__":
    unittest.main()
