"""Compare the current stopping rule with continued evolution, without editing GA.

Run from the project root: python -B -m tests.diagnose_ga_convergence
The 100 generations are a diagnostic observation period, not a new GA stop rule.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.GenaticAlgorithm import AntennaBounds_mm, GeneticAlgorithm


OBSERVATION_GENERATIONS = 100
WINDOW = 20
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "ga" / "diagnosis"


def inspect_run(size, seed):
    config = GeneticAlgorithm.config(size, 100, 2, 0.8, 0.2, 0.1, 1, seed)
    stopped = GeneticAlgorithm(AntennaBounds_mm, seed=seed).run(config)
    ga = GeneticAlgorithm(AntennaBounds_mm, seed=seed)
    population = ga.initializePopulation(config)
    history = [min(ga.objective(a, config.frequencyPenalty) for a in population)]
    for _ in range(OBSERVATION_GENERATIONS):
        population = ga.create_next_generation(
            population, config, config.eliteCount, config.frequencyPenalty
        )
        assert len(population) == size
        assert all(a.geometryCheck() for a in population)
        history.append(min(ga.objective(a, config.frequencyPenalty) for a in population))

    # Replaying the same seed must reproduce the actual run's entire prefix.
    np.testing.assert_array_equal(stopped.bestScoreHistory, history[:len(stopped.bestScoreHistory)])
    assert stopped.iterativeNumber == len(stopped.bestScoreHistory) - 1
    assert stopped.bestFitness == stopped.bestScoreHistory[-1]
    assert stopped.bestFitness == ga.objective(stopped.bestDesign_mm, config.frequencyPenalty)
    assert all(later <= earlier for earlier, later in zip(history, history[1:]))
    changes = np.abs(np.diff(stopped.bestScoreHistory)) / np.abs(stopped.bestScoreHistory[:-1])
    assert changes[-1] < config.terminationCondition
    assert np.all(changes[:-1] >= config.terminationCondition)

    best = min(population, key=lambda a: ga.objective(a, config.frequencyPenalty))
    window_stop = next((
        i for i in range(WINDOW, len(history))
        if abs(history[i] - history[i - WINDOW]) / abs(history[i - WINDOW]) < config.terminationCondition
    ), None)
    return {
        "population_size": size, "actual_seed": seed,
        "settings": {**vars(config), "terminationCondition": config.terminationCondition,
                     "frequencyPenalty": config.frequencyPenalty},
        "original_stop_generation": stopped.iterativeNumber,
        "original_stop_fitness": float(stopped.bestFitness),
        "original_target_error_MHz": float(abs(stopped.bestDesign_mm.ResonanceF_GHz - 2.4) * 1000),
        "continued_history": [float(value) for value in history],
        "continued_design_mm": [float(best.width_mm), float(best.length_mm), float(best.thickness_mm)],
        "continued_frequency_GHz": float(best.ResonanceF_GHz),
        "continued_target_error_MHz": float(abs(best.ResonanceF_GHz - 2.4) * 1000),
        "window_stop_generation": window_stop,
        "window_stop_fitness": float(history[window_stop]) if window_stop is not None else None,
    }


def main():
    records = [inspect_run(size, seed) for size in (3, 30) for seed in SEEDS]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "measurements.json").write_text(json.dumps({
        "observation_generations": OBSERVATION_GENERATIONS,
        "suggested_window_generations": WINDOW,
        "chosen_frequency_tolerance_MHz": 10.0,
        "note": "Diagnostic choices, not worksheet requirements. Production code unchanged.",
        "runs": records,
    }, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, size in zip(axes, (3, 30)):
        record = next(row for row in records if row["population_size"] == size and row["actual_seed"] == 42)
        history = record["continued_history"]
        axis.plot(range(len(history)), history, label="Continued evolution (diagnostic)")
        stop = record["original_stop_generation"]
        axis.scatter([stop], [history[stop]], color="crimson", zorder=3, label="Current stopping rule")
        stop = record["window_stop_generation"]
        if stop is not None:
            axis.scatter([stop], [history[stop]], marker="s", color="darkorange", zorder=3,
                         label="Proposed 20-generation comparison")
        axis.set(title=f"Population = {size}, seed = 42", xlabel="Generation", ylabel="Best fitness (lower is better)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(OUTPUT / "convergence_comparison.png", dpi=180)
    plt.close(figure)

    for size in (3, 30):
        rows = [row for row in records if row["population_size"] == size]
        print(json.dumps({
            "population_size": size, "seeds_tested": len(rows),
            "stopped_after_one_generation": sum(row["original_stop_generation"] == 1 for row in rows),
            "improved_if_continued": sum(row["continued_history"][-1] < row["original_stop_fitness"] for row in rows),
            "original_within_10_MHz": sum(row["original_target_error_MHz"] <= 10 for row in rows),
            "continued_within_10_MHz": sum(row["continued_target_error_MHz"] <= 10 for row in rows),
        }))
    print(f"All 20 diagnostic runs passed the history, score, and geometry checks. Output: {OUTPUT}")


if __name__ == "__main__":
    main()
