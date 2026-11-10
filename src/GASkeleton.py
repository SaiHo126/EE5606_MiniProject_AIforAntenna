"""Hand-written real-valued GA using the RMA class in PatchFormula.py.

Representation:
    Individual: [W_mm, L_mm, h_mm], shape (3,).
    Population: a float NumPy array, shape (population_size, 3).
    Scores: a float array, shape (population_size,); lower is better.

Run from the project root: python src/GASkeleton.py --seed 42
Results and a convergence CSV are saved in results/ga by default.
The objective calls the existing RMA class without changing its formulas.
It expects RMA.ResonanceF in Hz and RMA.volume_mm3 in mm^3.
The final status checks geometry, frequency tolerance, and an antenna-model
reference case. Failed checks are reported honestly and return exit code 1.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

if __package__:
    from .PatchFormula import RMA
else:
    from PatchFormula import RMA


# Assignment bounds, in the order W, L, h. Each row is [lower, upper].
ANTENNA_BOUNDS_MM = np.array([[25.0, 50.0], [25.0, 45.0], [1.0, 5.0]])


def geometry_is_feasible(design_mm: np.ndarray) -> bool:
    """Check the bounds and W/L >= 1.2; frequency acceptance is separate."""
    design = np.asarray(design_mm, dtype=float)
    if design.shape != (3,) or not np.all(np.isfinite(design)):
        return False
    return bool(
        np.all(design >= ANTENNA_BOUNDS_MM[:, 0])
        and np.all(design <= ANTENNA_BOUNDS_MM[:, 1])
        and design[0] >= 1.2 * design[1]
    )


def repair_antenna_design(design_mm: np.ndarray) -> np.ndarray:
    """Return a feasible copy after sampling, crossover, or mutation."""
    design = np.array(design_mm, dtype=float, copy=True)
    if design.shape != (3,) or not np.all(np.isfinite(design)):
        raise ValueError("A design must contain three finite values: [W, L, h].")
    # L must be <= 50/1.2. Move its floating-point cap one step inward so
    # computing 1.2 * L cannot round above the maximum allowed width of 50.
    maximum_length = np.nextafter(50.0 / 1.2, -np.inf)
    design[1] = np.clip(design[1], 25.0, maximum_length)
    design[0] = np.clip(design[0], 1.2 * design[1], 50.0)
    design[2] = np.clip(design[2], 1.0, 5.0)
    return design


def antenna_objective(design_mm: np.ndarray, penalty_weight: float = 1.0) -> float:
    """Use PatchFormula's calculated attributes to score one design.

    Minimize J = volume/11250 + penalty_weight * abs(f_GHz - 2.4)/2.4.
    The penalty weight is a starting choice, not a worksheet requirement.
    Recheck final frequency against your chosen tolerance after optimization.
    This adapter assumes the antenna class supplies ResonanceF in Hz.
    """
    if not np.isfinite(penalty_weight) or penalty_weight < 0:
        raise ValueError("penalty_weight must be finite and nonnegative.")
    if not geometry_is_feasible(design_mm):
        return float("inf")
    width, length, thickness = np.asarray(design_mm, dtype=float)
    antenna = RMA(width, length, thickness)
    frequency_ghz = antenna.ResonanceF / 1.0e9
    volume_term = antenna.volume_mm3 / 11250.0
    frequency_term = abs(frequency_ghz - 2.4) / 2.4
    return float(volume_term + penalty_weight * frequency_term)


@dataclass(frozen=True)
class GAConfig:
    """Suggested settings for you to tune after the implementation works."""

    population_size: int = 60
    generations: int = 150
    tournament_size: int = 3
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1  # Per coordinate.
    mutation_scale: float = 0.05  # Noise standard deviation / variable range.
    elite_count: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("population_size", "generations", "tournament_size", "elite_count", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer.")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        if self.population_size < 2 or self.generations < 0:
            raise ValueError("Use population_size >= 2 and generations >= 0.")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be within the population size.")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must leave space for at least one child.")
        for value in (self.crossover_probability, self.mutation_probability):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Crossover and mutation probabilities must be in [0, 1].")
        if not np.isfinite(self.mutation_scale) or self.mutation_scale < 0:
            raise ValueError("mutation_scale must be finite and nonnegative.")


@dataclass
class GAResult:
    """History includes the initial population, then each generation."""

    best_design_mm: np.ndarray
    best_score: float
    best_score_history: list[float]
    objective_evaluations: int
    elapsed_seconds: float


class GeneticAlgorithm:
    """Reusable GA driver; antenna physics is supplied through objective.

    Use a new instance with the same seed to repeat a random-number sequence.
    All random draws use self.rng rather than the global random state.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        repair: Callable[[np.ndarray], np.ndarray],
        config: GAConfig | None = None,
    ) -> None:
        self.config = config if config is not None else GAConfig()
        self.bounds = np.array(bounds, dtype=float, copy=True)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2 or not len(self.bounds):
            raise ValueError("bounds must have shape (number_of_variables, 2).")
        if not np.all(np.isfinite(self.bounds)) or np.any(
            self.bounds[:, 0] >= self.bounds[:, 1]
        ):
            raise ValueError("Each bound must be finite with lower < upper.")
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]
        self.variable_ranges = self.upper_bounds - self.lower_bounds
        self.number_of_variables = len(self.bounds)
        self.objective = objective
        self.repair = repair
        self.rng = np.random.default_rng(self.config.seed)
        self.objective_evaluations = 0

    def initialize_population(self) -> np.ndarray:
        """Sample dimensions uniformly, then repair each candidate's geometry."""
        population = self.rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(self.config.population_size, self.number_of_variables),
        )
        return np.array([self.repair(row) for row in population], dtype=float)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Provided: score every row and count objective evaluations."""
        population = np.asarray(population, dtype=float)
        expected_shape = (self.config.population_size, self.number_of_variables)
        if population.shape != expected_shape or not np.all(np.isfinite(population)):
            raise ValueError(f"population must have shape {expected_shape} and be finite.")
        scores = np.array([self.objective(row) for row in population], dtype=float)
        self.objective_evaluations += len(population)
        if scores.shape != (self.config.population_size,):
            raise ValueError("The objective must return one scalar score per individual.")
        if np.any(np.isnan(scores)) or np.any(np.isneginf(scores)):
            raise ValueError("The objective returned NaN or negative infinity.")
        return scores

    def select_parent(self, population: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Choose the lowest-scoring member of a randomly sampled tournament."""
        competitors = self.rng.choice(
            len(population), size=self.config.tournament_size, replace=False
        )
        winner = competitors[np.argmin(scores[competitors])]
        # Copying avoids changing a parent when its child is mutated.
        return population[winner].copy()

    def crossover(
        self, parent_a: np.ndarray, parent_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mix parent coordinates using arithmetic crossover, then repair."""
        if self.rng.random() < self.config.crossover_probability:
            alpha = self.rng.random(self.number_of_variables)
            child_a = alpha * parent_a + (1.0 - alpha) * parent_b
            child_b = (1.0 - alpha) * parent_a + alpha * parent_b
        else:
            child_a, child_b = parent_a.copy(), parent_b.copy()
        return self.repair(child_a), self.repair(child_b)

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Perturb selected coordinates with Gaussian noise, then repair."""
        child = np.array(individual, dtype=float, copy=True)
        mask = self.rng.random(self.number_of_variables) < self.config.mutation_probability
        noise = self.rng.normal(
            loc=0.0,
            scale=self.config.mutation_scale * self.variable_ranges,
        )
        child[mask] += noise[mask]
        return self.repair(child)

    def create_next_generation(
        self, population: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Preserve elites and fill the remaining slots with new offspring."""
        order = np.argsort(scores)
        next_population = [population[i].copy() for i in order[:self.config.elite_count]]
        while len(next_population) < self.config.population_size:
            parent_a = self.select_parent(population, scores)
            parent_b = self.select_parent(population, scores)
            for child in self.crossover(parent_a, parent_b):
                next_population.append(self.mutate(child))
                # Handles odd population sizes and an odd number of free slots.
                if len(next_population) == self.config.population_size:
                    break
        return np.array(next_population, dtype=float)

    def run(self) -> GAResult:
        """Evolve the population and retain the best design seen in any generation."""
        started = perf_counter()
        self.objective_evaluations = 0
        population = self.initialize_population()
        scores = self.evaluate_population(population)
        best_index = int(np.argmin(scores))
        best_design = population[best_index].copy()
        best_score = float(scores[best_index])
        history = [best_score]

        for _ in range(self.config.generations):
            population = self.create_next_generation(population, scores)
            scores = self.evaluate_population(population)
            best_index = int(np.argmin(scores))
            if scores[best_index] < best_score:
                best_design = population[best_index].copy()
                best_score = float(scores[best_index])
            history.append(best_score)

        if not np.isfinite(best_score):
            raise ValueError("No design with a finite objective was found.")
        return GAResult(
            best_design_mm=best_design,
            best_score=best_score,
            best_score_history=history,
            objective_evaluations=self.objective_evaluations,
            elapsed_seconds=perf_counter() - started,
        )


def main(argv: list[str] | None = None) -> int:
    """Run Part I and save enough settings and history to reproduce it."""
    parser = argparse.ArgumentParser(
        description="Part I: hand-written antenna GA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument(
        "--penalty-weight", type=float, default=1.0,
        help="Weight of normalized frequency error; a tuning choice",
    )
    parser.add_argument(
        "--frequency-tolerance-mhz", type=float, default=10.0,
        help="Chosen final acceptance tolerance, not specified by the worksheet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "ga",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.penalty_weight) or args.penalty_weight < 0:
        parser.error("--penalty-weight must be finite and nonnegative")
    if not np.isfinite(args.frequency_tolerance_mhz) or args.frequency_tolerance_mhz <= 0:
        parser.error("--frequency-tolerance-mhz must be finite and positive")
    config = GAConfig(
        seed=args.seed, population_size=args.population_size, generations=args.generations
    )
    optimizer = GeneticAlgorithm(
        objective=lambda design: antenna_objective(design, args.penalty_weight),
        bounds=ANTENNA_BOUNDS_MM,
        repair=repair_antenna_design,
        config=config,
    )
    # This independent known value checks the model, not GA convergence.
    reference = RMA(38.0, 30.0, 1.6)
    expected_reference_ghz = 2.3575485421095275
    reference_ghz = float(reference.ResonanceF / 1e9)
    model_valid = bool(
        np.isclose(reference_ghz, expected_reference_ghz, rtol=1e-8, atol=0.0)
        and np.isclose(reference.volume_mm3, 1824.0, rtol=1e-12, atol=0.0)
    )
    result = optimizer.run()
    antenna = RMA(*result.best_design_mm)
    frequency_ghz = float(antenna.ResonanceF / 1e9)
    error_mhz = abs(frequency_ghz - 2.4) * 1000.0
    geometry_valid = geometry_is_feasible(result.best_design_mm)
    frequency_valid = bool(error_mhz <= args.frequency_tolerance_mhz)
    success = bool(model_valid and geometry_valid and frequency_valid)

    summary = {
        "settings": asdict(config),
        "bounds_mm": ANTENNA_BOUNDS_MM.tolist(),
        "minimum_width_length_ratio": 1.2,
        "target_frequency_ghz": 2.4,
        "penalty_weight": args.penalty_weight,
        "frequency_tolerance_mhz": args.frequency_tolerance_mhz,
        "best_design_mm": result.best_design_mm.tolist(),
        "volume_mm3": float(antenna.volume_mm3),
        "width_length_ratio": float(antenna.width_mm / antenna.length_mm),
        "frequency_ghz": frequency_ghz,
        "frequency_error_mhz": error_mhz,
        "best_score": result.best_score,
        "objective_evaluations": result.objective_evaluations,
        "elapsed_seconds": result.elapsed_seconds,
        "geometry_feasible": geometry_valid,
        "frequency_feasible": frequency_valid,
        "model_reference_passed": model_valid,
        "model_reference_actual_ghz": reference_ghz,
        "model_reference_expected_ghz": expected_reference_ghz,
        "success": success,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"ga_seed_{config.seed}.json"
    history_path = args.output_dir / f"ga_seed_{config.seed}_history.csv"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "best_objective_so_far"])
        writer.writerows(enumerate(result.best_score_history))

    print(f"Best dimensions [W, L, h]: {result.best_design_mm} mm")
    print(f"Volume: {antenna.volume_mm3:.6f} mm^3")
    print(f"W/L: {antenna.width_mm / antenna.length_mm:.6f}")
    print(f"Resonance frequency: {frequency_ghz:.9f} GHz")
    print(f"Target error: {error_mhz:.6f} MHz (tolerance {args.frequency_tolerance_mhz:g} MHz)")
    print(f"Best objective: {result.best_score:.9f}")
    print(f"Evaluations: {result.objective_evaluations}; search time: {result.elapsed_seconds:.3f} s")
    print(f"Geometry check: {'PASS' if geometry_valid else 'FAIL'}")
    print(f"Frequency check: {'PASS' if frequency_valid else 'FAIL'}")
    print(f"Antenna-model reference check: {'PASS' if model_valid else 'FAIL'}")
    if not model_valid:
        print(
            f"PatchFormula reference case: {reference_ghz:.9f} GHz; "
            f"worksheet expectation: {expected_reference_ghz:.9f} GHz."
        )
    print(f"Assignment design status: {'PASS' if success else 'FAIL'}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved convergence: {history_path}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
