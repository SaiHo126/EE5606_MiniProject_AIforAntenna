import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

if __package__:
    from .GPRskeleton import GaussianProcessRegressor
    from .NeuralNetwork import NeuralNetwork
    from .PatchFormula import RMA, AntennaBounds_mm
else:
    from GPRskeleton import GaussianProcessRegressor
    from NeuralNetwork import NeuralNetwork
    from PatchFormula import RMA, AntennaBounds_mm
from dataclasses import dataclass, asdict
import numpy as np

class PSO:
    
    @dataclass
    class Config:
        swarm_size :int = 40
        iterations_min: int =150
        iterations_max: int = 1000
        target = 0.01
        inertia: float = 0.7
        cognitive: float = 1.6 #c1
        social: float = 1.6 #c2
        velocity_fraction = 0.2 #limits each velocity componenet to n% of that dimension's allowed range
        seed =1
    @dataclass
    class Result:
        best_position: np.ndarray
        best_position_history : np.ndarray
        best_score: float
        best_score_history: np.ndarray
        elapsed_seconds_history: np.ndarray
        objective_evaluations: int
        search_seconds:float
    
    def __init__(self, objective, bounds, Config, repair):
        self.objective = objective
        self.bounds = np.array(bounds, dtype=float, copy=True)
        self.Config = Config
        self.repair = repair

        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

        # Maximum velocity for each coordinate.
        self.vmax = (
            Config.velocity_fraction * (self.upper - self.lower)
        )
    @staticmethod
    def repair_design(design):
        # Repair dimensions without evaluating analytical frequency during PSO.
        design = np.clip(
            np.asarray(design, dtype=float),
            AntennaBounds_mm[:, 0],
            AntennaBounds_mm[:, 1],
        )

        if design[0] / design[1] < 1.2:
            design[1] = min(design[1], np.nextafter(50.0 / 1.2, -np.inf))
            design[0] = max(design[0], 1.2 * design[1])
            if design[0] / design[1] < 1.2:
                design[0] = np.nextafter(design[0], np.inf)
        return design
    def initialize_swarm(self, rng):
        shape = (self.Config.swarm_size, len(self.lower))
        positions = rng.uniform(self.lower, self.upper, size=shape)
        positions = np.array([self.repair(position) for position in positions])
        velocities = rng.uniform(-self.vmax, self.vmax, size = shape)
        
        return positions, velocities
    
    def evaluate(self, positions):
        scores = np.array([self.objective(position) for position in positions], dtype=float)
        return scores
    def update_velocity(self, positions, velocities, personal_best, global_best, rng):
        r1 = rng.random(positions.shape)
        r2= rng.random(positions.shape)
        velocities = (
            self.Config.inertia * velocities
            + self.Config.cognitive * r1 * (personal_best - positions)
            + self.Config.social* r2 * (global_best - positions)
        )
        return np.clip(velocities, -self.vmax, self.vmax)
    def move_particle(self, positions, velocities):
        positions_new = np.array([self.repair(position) for position in positions + velocities])
        velocities = velocities.copy()
        return positions_new, velocities
    def run(self):
        started = perf_counter()
        rng = np.random.default_rng(self.Config.seed)

        # 1. Initialize and evaluate the swarm.
        positions, velocities = self.initialize_swarm(rng)
        scores = self.evaluate(positions)

        # 2. Save each particle's best position and score.
        personal_best = positions.copy()
        personal_scores = scores.copy()

        # 3. Save the best position found by the entire swarm.
        best_index = np.argmin(personal_scores)
        global_best = personal_best[best_index].copy()
        global_score = float(personal_scores[best_index])

        # Lists can grow using append().
        score_history = [global_score]
        position_history = [global_best.copy()]
        # Index 0 includes initialization and the first swarm evaluation.
        elapsed_history = [perf_counter() - started]

        iteration = 0
        fitness_change = float("inf")

        while (
            iteration < self.Config.iterations_max
            and (
                iteration < self.Config.iterations_min
                or fitness_change > self.Config.target
            )
        ):
            previous_score = global_score

            # 4. Move the swarm and evaluate the repaired positions.
            velocities = self.update_velocity(
                positions, velocities, personal_best, global_best, rng
            )
            positions, velocities = self.move_particle(
                positions, velocities
            )
            scores = self.evaluate(positions)

            # 5. Update personal bests.
            improved = scores < personal_scores
            personal_best[improved] = positions[improved].copy()
            personal_scores[improved] = scores[improved]

            # 6. Update global best.
            best_index = np.argmin(personal_scores)

            if personal_scores[best_index] < global_score:
                global_best = personal_best[best_index].copy()
                global_score = float(personal_scores[best_index])

            # 7. Record progress and calculate relative improvement.
            score_history.append(global_score)
            position_history.append(global_best.copy())

            fitness_change = (
                (previous_score - global_score)
                / max(abs(previous_score), 1e-12)
            )

            iteration += 1
            elapsed_history.append(perf_counter() - started)

        # 8. Return an instance containing this run's results.
        return self.Result(
            best_position=global_best.copy(),
            best_position_history=np.array(position_history),
            best_score=global_score,
            best_score_history=np.array(score_history),
            elapsed_seconds_history=np.array(elapsed_history),
            objective_evaluations=(self.Config.swarm_size * (iteration + 1)),
            search_seconds=perf_counter() - started,
        )


def verify_results(design, model, tolerance_GHz = 0.01):
    started = perf_counter()
    predicted = float(model.predict(design))
    prediction_seconds = perf_counter() - started

    started = perf_counter()
    antenna = RMA(*design)
    actual = antenna.ResonanceF_GHz
    verification_seconds = perf_counter() - started 

    target_error = abs(actual - 2.4)
    prediction_error = abs(predicted - actual)
    geometry_ok = antenna.geometryCheck()

    return {
        "design_mm": design.tolist(),
        "volume_mm3": float(antenna.volume_mm3),
        "predicted_GHz": predicted,
        "analytical_GHz": float(actual),
        "target_error_MHz": float(target_error * 1000),
        "prediction_error_MHz": float(prediction_error * 1000),
        "geometry_feasible": geometry_ok,
        "success": bool(geometry_ok and target_error <= tolerance_GHz),
        "final_prediction_seconds": prediction_seconds,
        "analytical_verification_seconds": verification_seconds,
    }
def make_objective(predict_frequency, penalty_weight):
    def objective(design):
        W, L, h = design

        predicted_GHz = float(predict_frequency(design))
        volume_mm3 = W * L * h

        return (
            volume_mm3 / 11250.0
            + penalty_weight * abs(predicted_GHz - 2.4)
        )

    return objective

def compare_models(nn, gp, penalty_weight, frequency_tolerance_GHz, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents= True, exist_ok = True)
    seeds = [1,2,3,4,5]
    report = {
        "penalty_weight": penalty_weight,
        "frequency_tolerance_GHz": frequency_tolerance_GHz,
        "target_GHz": 2.4,
        "objective": "volume_mm3/11250 + penalty_weight*abs(predicted_GHz-2.4)",
        "timing_scope": "PSO initialization and search, including surrogate predictions and geometry repair; excludes model loading, training, plotting, and final verification.",
        "runs": [],
        "summary": {},
    }
    
    for name, model in {"NN":nn, "GP": gp}.items():
        objective = make_objective(model.predict, penalty_weight)
        model_runs= []
        for seed in seeds:
            config = PSO.Config()
            config.seed = seed
            optimizer = PSO(
                objective,
                bounds= AntennaBounds_mm,
                Config= config,
                repair=PSO.repair_design
            )
            result = optimizer.run()
            checked = verify_results(
                result.best_position,
                model,
                tolerance_GHz= frequency_tolerance_GHz
            )
            settings = asdict(config)
            settings.update({
                "seed": config.seed,
                "target": config.target,
                "velocity_fraction": config.velocity_fraction,
            })

            row = {
                "model": name,
                "seed": seed,
                "config": settings,
                "iterations_completed": len(result.best_score_history) - 1,
                "best_score": float(result.best_score),
                "objective_evaluations": int(result.objective_evaluations),
                "search_seconds": float(result.search_seconds),
                **checked,
            }

            report["runs"].append(row)
            model_runs.append(row)

            # 6. Save convergence history for this model and seed.
            # Measured elapsed time accompanies each recorded best score.
            history = np.column_stack([
                np.arange(len(result.best_score_history)),
                result.best_score_history,
                result.best_position_history,
                result.elapsed_seconds_history,
            ])

            np.savetxt(
                output_dir / f"{name}_seed_{seed}_history.csv",
                history,
                delimiter=",",
                header="iteration,best_score,W_mm,L_mm,h_mm,elapsed_seconds",
                comments="",
            )

            print(
                f"{name}, seed {seed}: "
                f"volume={checked['volume_mm3']:.3f} mm^3, "
                f"target error={checked['target_error_MHz']:.3f} MHz, "
                f"prediction error={checked['prediction_error_MHz']:.3f} MHz, "
                f"search time={result.search_seconds:.4f} s, "
                f"success={checked['success']}"
            )

        # 7. Summarize all runs for the current model.
        successful_runs = [
            row for row in model_runs if row["success"]
        ]

        summary = {
            "runs": len(model_runs),
            "successes": len(successful_runs),
            "success_rate": len(successful_runs) / len(model_runs),
            "mean_search_seconds": float(np.mean([
                row["search_seconds"] for row in model_runs
            ])),
            "std_search_seconds": float(np.std([
                row["search_seconds"] for row in model_runs
            ])),
            "mean_target_error_MHz": float(np.mean([
                row["target_error_MHz"] for row in model_runs
            ])),
            "mean_successful_volume_mm3": (
                float(np.mean([
                    row["volume_mm3"] for row in successful_runs
                ]))
                if successful_runs else None
            ),
        }

        report["summary"][name] = summary

        print(
            f"{name} success rate: "
            f"{summary['successes']}/{summary['runs']}; "
            f"mean search time={summary['mean_search_seconds']:.4f} s"
        )

    # 8. Save the complete comparison.
    report_path = output_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    plot_convergence(report, output_dir)
    return report


def plot_convergence(report, output_dir):
    """Plot every seed using its measured timestamps, not estimated timings."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for name, color in (("NN", "tab:blue"), ("GP", "tab:orange")):
        runs = [row for row in report["runs"] if row["model"] == name]
        summary = report["summary"][name]
        label = (f"{name}: mean {summary['mean_search_seconds']:.3f} s "
                 f"({len(runs)} seeds)")
        for index, row in enumerate(runs):
            history = np.genfromtxt(output_dir / f"{name}_seed_{row['seed']}_history.csv",
                                    delimiter=",", names=True, ndmin=1)
            for ax, x in zip(axes, (history["iteration"], history["elapsed_seconds"])):
                ax.step(x, history["best_score"], where="post", color=color,
                        alpha=0.65, linewidth=1.2, label=label if index == 0 else None)
    axes[0].set_xlabel("PSO iteration (0 = initial swarm)")
    axes[0].set_ylabel("Best surrogate objective so far (lower is better)")
    axes[0].set_title("Convergence by iteration")
    axes[1].set_xlabel("Measured elapsed search time (s)")
    axes[1].set_title("Convergence by elapsed time")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("NN vs GP: PSO convergence with fixed trained surrogates")
    fig.text(0.5, 0.02,
             "Each line is one seed. Time includes initialization, surrogate predictions, and particle updates.\n"
             "Model loading/training, plotting, and final analytical verification are excluded.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    path = output_dir / "convergence_comparison.png"
    fig.savefig(path, dpi=200)
    print(f"Saved convergence plot: {path}")
    return fig

def main(argv=None):
    # Find the project root from this file's location.
    project_root = Path(__file__).resolve().parents[1]

    # Saved models: no retraining during PSO.
    nn_path = project_root / "results/nn/nn_model_line_search.npz"
    gp_path = project_root / "results/gp/gp_model.npz"

    parser = argparse.ArgumentParser(description="Compare NN and GP PSO convergence and measured search time.")
    parser.add_argument("--output-dir", type=Path,
                        default=project_root / "results/pso_comparison_cpu")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening a window.")
    args = parser.parse_args(argv)
    output_dir = args.output_dir

    # Experiment settings—use the same values for NN and GP.
    penalty_weight = 1.0
    frequency_tolerance_GHz = 0.01  # 10 MHz

    # Check that both model files exist.
    for path in (nn_path, gp_path):
        if not path.is_file():
            raise FileNotFoundError(f"Saved model not found: {path}")

    # Load the trained NN and GP, including their saved scaling.
    print(f"Loading NN on CPU: {nn_path}")
    nn = NeuralNetwork.load(nn_path, cpu=True)

    print(f"Loading GP: {gp_path}")
    gp = GaussianProcessRegressor.load(gp_path)

    # Run separate PSO searches and verify the returned designs.
    compare_models(
        nn=nn,
        gp=gp,
        penalty_weight=penalty_weight,
        frequency_tolerance_GHz=frequency_tolerance_GHz,
        output_dir=output_dir,
    )

    print(f"Comparison completed. Results saved in: {output_dir}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()    
        
            
            
            
            

            
            
        
        
        
    
        
               
        
