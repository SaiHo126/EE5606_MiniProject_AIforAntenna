"""Checks for the student's object-based GA operators.

Run from the project root:
    python -B -m unittest discover -s tests -p test_genatic_algorithm.py -v

The short generation loop below is test scaffolding, not a production GA run.
"""

import unittest

import numpy as np

from src.GenaticAlgorithm import AntennaBounds_mm, GeneticAlgorithm
from src.PatchFormula import RMA


def make_config(**changes):
    settings = dict(
        populationSize=9,
        generations=6,
        tournamentSize=3,
        crossoverRate=0.8,
        mutationRate=0.2,
        stepSize=0.1,
        eliteCount=1,
        seed=42,
    )
    settings.update(changes)
    return GeneticAlgorithm.config(**settings)


def dimensions(antenna):
    return np.array(
        [antenna.width_mm, antenna.length_mm, antenna.thickness_mm], dtype=float
    )


def population_dimensions(population):
    return np.array([dimensions(antenna) for antenna in population])


class GeneticAlgorithmOperatorTests(unittest.TestCase):
    def assert_consistent_and_feasible(self, antenna, bounds=AntennaBounds_mm):
        """Use the public formula to catch stale values after dimension repair."""
        self.assertIsInstance(antenna, RMA)
        genes = dimensions(antenna)
        self.assertTrue(np.all(np.isfinite(genes)), genes)
        self.assertTrue(np.all(genes >= bounds[:, 0]), genes)
        self.assertTrue(np.all(genes <= bounds[:, 1]), genes)
        self.assertGreaterEqual(genes[0] / genes[1], 1.2)
        self.assertTrue(antenna.geometryCheck(), genes)
        reference = RMA(*genes)
        for attribute in (
            "volume_mm3", "RelativePermittivityEff", "lengthDelta_mm",
            "lengthEff_mm", "ResonanceF_GHz",
        ):
            self.assertAlmostEqual(
                getattr(antenna, attribute), getattr(reference, attribute), places=11,
                msg=attribute,
            )

    def test_analytical_reference_uses_ghz_and_cubic_mm(self):
        antenna = RMA(38.0, 30.0, 1.6)
        self.assertAlmostEqual(antenna.ResonanceF_GHz, 2.3575485421095275, places=10)
        self.assertAlmostEqual(antenna.volume_mm3, 1824.0)

    def test_initialization_returns_independent_feasible_objects(self):
        config = make_config(populationSize=100)
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=17)
        population = optimizer.initializePopulation(config)
        self.assertIsInstance(population, np.ndarray)
        self.assertEqual(population.shape, (100,))
        self.assertEqual(population.dtype, np.dtype(object))
        self.assertEqual(len({id(antenna) for antenna in population}), len(population))
        for antenna in population:
            self.assert_consistent_and_feasible(antenna)

    def test_repeated_initialization_advances_rng(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=21)
        config = make_config()
        first = population_dimensions(optimizer.initializePopulation(config))
        second = population_dimensions(optimizer.initializePopulation(config))
        self.assertFalse(np.array_equal(first, second))

    def test_selection_advances_rng_instead_of_reseeding(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=21)
        config = make_config(tournamentSize=1)
        population = np.array([RMA(35.0 + i, 25.0, 2.0) for i in range(9)])
        selected_widths = {
            optimizer.select_parent(population, config).width_mm for _ in range(30)
        }
        self.assertGreater(len(selected_widths), 1)

    def test_full_tournament_selects_lowest_fitness_and_returns_copy(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=8)
        config = make_config(populationSize=3, tournamentSize=3)
        population = np.array([RMA(50.0, 40.0, 5.0), RMA(38.0, 30.0, 1.6), RMA(40.0, 25.0, 2.0)])
        before = [vars(antenna).copy() for antenna in population]
        expected = min(population, key=lambda antenna: antenna.fitness(2.5))
        selected = optimizer.select_parent(population, config, frequencyPlenalty=2.5)
        np.testing.assert_array_equal(dimensions(selected), dimensions(expected))
        self.assertTrue(all(selected is not antenna for antenna in population))
        self.assert_consistent_and_feasible(selected)
        selected.width_mm = 49.0
        self.assertEqual([vars(antenna) for antenna in population], before)

    def test_custom_objective_receives_frequency_penalty(self):
        calls = []

        def objective(antenna, penalty):
            calls.append((antenna, penalty))
            return -antenna.width_mm

        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=11)
        optimizer.objective = objective
        population = np.array([RMA(35.0, 25.0, 2.0), RMA(45.0, 30.0, 2.0), RMA(40.0, 30.0, 2.0)])
        parent = optimizer.select_parent(
            population, make_config(populationSize=3, tournamentSize=3),
            frequencyPlenalty=7.0,
        )
        self.assertEqual(parent.width_mm, 45.0)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(penalty == 7.0 for _, penalty in calls))
        self.assertTrue(all(all(candidate is not original for original in population)
                            for candidate, _ in calls))

    def test_selection_refreshes_previously_repaired_antenna_properties(self):
        original = RMA(25.0, 45.0, 1.0)
        original.geometryFix()
        before = vars(original).copy()
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=11)
        selected = optimizer.select_parent(
            np.array([original]), make_config(populationSize=1, tournamentSize=1)
        )
        self.assert_consistent_and_feasible(selected)
        np.testing.assert_array_equal(dimensions(selected), dimensions(original))
        self.assertEqual(vars(original), before)

    def test_zero_crossover_returns_independent_parent_copies(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=13)
        parents = [RMA(38.0, 30.0, 1.6), RMA(50.0, 40.0, 4.0)]
        before = [vars(parent).copy() for parent in parents]
        children = optimizer.crossover(*parents, make_config(crossoverRate=0.0))
        self.assertIsInstance(children, np.ndarray)
        self.assertEqual(children.shape, (2,))
        self.assertEqual(children.dtype, np.dtype(object))
        for child, parent in zip(children, parents):
            self.assertIsNot(child, parent)
            np.testing.assert_array_equal(dimensions(child), dimensions(parent))
            self.assert_consistent_and_feasible(child)
        children[0].width_mm = 49.0
        self.assertEqual([vars(parent) for parent in parents], before)

    def test_full_crossover_blends_all_three_dimensions(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=15)
        parents = [RMA(36.0, 27.0, 1.5), RMA(49.0, 38.0, 4.5)]
        first, second = [dimensions(parent) for parent in parents]
        before = [vars(parent).copy() for parent in parents]
        children = optimizer.crossover(*parents, make_config(crossoverRate=1.0))
        child_genes = population_dimensions(children)
        np.testing.assert_allclose(child_genes.sum(axis=0), first + second)
        alpha = (child_genes[0] - second) / (first - second)
        self.assertGreater(alpha[0], 0.0)
        self.assertLess(alpha[0], 1.0)
        np.testing.assert_allclose(alpha, np.full(3, alpha[0]))
        for child in children:
            self.assertTrue(all(child is not parent for parent in parents))
            self.assert_consistent_and_feasible(child)
        self.assertEqual([vars(parent) for parent in parents], before)

    def test_crossover_preserves_ratio_at_floating_point_boundary(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=0)
        children = optimizer.crossover(
            RMA(36.0, 30.0, 1.6), RMA(48.0, 40.0, 2.0),
            make_config(crossoverRate=1.0),
        )
        for child in children:
            self.assert_consistent_and_feasible(child)

    def test_zero_mutation_returns_independent_copy(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=17)
        child = RMA(38.0, 30.0, 1.6)
        before = vars(child).copy()
        mutated = optimizer.mutate(child, make_config(mutationRate=0.0))
        self.assertIsNot(mutated, child)
        np.testing.assert_array_equal(dimensions(mutated), dimensions(child))
        self.assert_consistent_and_feasible(mutated)
        self.assertEqual(vars(child), before)

    def test_repair_handles_box_boundaries_and_invalid_aspect_ratio(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=19)
        cases = (
            [30.0, 25.0, 1.0], [50.0, 25.0, 5.0], [50.0, 50.0 / 1.2, 5.0],
            [25.0, 45.0, 1.0], [40.0, 40.0, 3.0],
            [10.0, 20.0, 0.2], [70.0, 50.0, 7.0],
        )
        self.assertFalse(RMA(25.0, 45.0, 1.0).geometryCheck())
        for genes in cases:
            with self.subTest(genes=genes):
                child = RMA(*genes)
                before = vars(child).copy()
                repaired = optimizer.mutate(child, make_config(mutationRate=0.0))
                self.assert_consistent_and_feasible(repaired)
                self.assertIsNot(repaired, child)
                self.assertEqual(vars(child), before)

    def test_repair_preserves_ratio_at_floating_point_boundary(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=0)
        child = RMA(25.0, 26.90883455893289, 1.6)
        repaired = optimizer.mutate(child, make_config(mutationRate=0.0))
        self.assert_consistent_and_feasible(repaired)

    def test_full_mutation_with_zero_step_preserves_dimensions(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=23)
        child = RMA(38.0, 30.0, 1.6)
        mutated = optimizer.mutate(child, make_config(mutationRate=1.0, stepSize=0.0))
        np.testing.assert_array_equal(dimensions(mutated), dimensions(child))
        self.assert_consistent_and_feasible(mutated)

    def test_integer_inputs_keep_fractional_mutation(self):
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=29)
        child = RMA(40, 30, 3)
        before = vars(child).copy()
        mutated = optimizer.mutate(child, make_config(mutationRate=1.0, stepSize=0.001))
        genes = dimensions(mutated)
        self.assertTrue(np.all(genes != np.round(genes)), genes)
        self.assert_consistent_and_feasible(mutated)
        self.assertEqual(vars(child), before)

    def test_large_mutations_are_repaired_before_formula_evaluation(self):
        config = make_config(mutationRate=1.0, stepSize=20.0)
        child = RMA(38.0, 30.0, 1.6)
        before = vars(child).copy()
        for seed in (0, 7, 42):
            optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=seed)
            for _ in range(30):
                self.assert_consistent_and_feasible(optimizer.mutate(child, config))
        self.assertEqual(vars(child), before)

    def run_operator_generations(self, seed, config):
        """Exercise existing operators together, with an elite retained by this test."""
        optimizer = GeneticAlgorithm(AntennaBounds_mm, seed=seed)
        population = optimizer.initializePopulation(config)
        history = [population_dimensions(population)]
        for _ in range(config.generations):
            before = [vars(antenna).copy() for antenna in population]
            best = min(population, key=lambda antenna: antenna.fitness(1.0))
            next_population = [RMA(*dimensions(best))]
            while len(next_population) < config.populationSize:
                parents = [optimizer.select_parent(population, config) for _ in range(2)]
                for child in optimizer.crossover(*parents, config):
                    if len(next_population) < config.populationSize:
                        next_population.append(optimizer.mutate(child, config))
            self.assertEqual([vars(antenna) for antenna in population], before)
            population = np.array(next_population, dtype=object)
            self.assertEqual(len(population), config.populationSize)
            self.assertEqual(len({id(antenna) for antenna in population}), len(population))
            for antenna in population:
                self.assert_consistent_and_feasible(antenna)
            history.append(population_dimensions(population))
        return np.array(history)

    def test_operator_workflow_is_reproducible_across_several_seeds(self):
        outcomes = []
        for seed in (0, 7, 42):
            with self.subTest(seed=seed):
                first = self.run_operator_generations(seed, make_config())
                second = self.run_operator_generations(seed, make_config())
                np.testing.assert_array_equal(first, second)
                outcomes.append(first)
        self.assertFalse(np.array_equal(outcomes[0], outcomes[1]))

    def test_config_seed_does_not_reset_constructor_generator(self):
        first = self.run_operator_generations(42, make_config(seed=1))
        second = self.run_operator_generations(42, make_config(seed=999))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
