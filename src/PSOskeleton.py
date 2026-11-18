"""Complete, commented NumPy particle swarm optimization for EE5606 Part II.

From the project root, first train the GP with GPRskeleton.py, then:
    python src/PSOskeleton.py --surrogate gp
    python src/PSOskeleton.py --surrogate nn --nn-model results/nn/nn_model_raw.npz
    python src/PSOskeleton.py --surrogate both --seeds 1 2 3 4 5
For your ReLU NN use --nn-model results/nn/nn_model_relu_h16_gpu.npz.
Saved ReLU models run on CPU here, including those trained on GPU.

Build/read in this order (matching the STEP comments below):
1. PSOConfig: settings and units.
2. geometry_is_feasible / repair_antenna_design: enforce RMA's geometry rules.
3. make_antenna_objective: use RMA's score formula with a surrogate frequency.
4-9. ParticleSwarmOptimizer.run: initialize, remember, move, score, return.
10. verify_design: use RMA's analytical frequency after the search.
Finally, main connects the saved models, optimizer, and output files.

Each particle is one possible design [W, L, h] in mm. Position and velocity
arrays have shape (swarm_size, 3); scores have shape (swarm_size,).
The generic optimizer accepts an objective callable, so it can also solve a
simple test problem. It contains no antenna physics or surrogate training.

Part II uses a FIXED trained surrogate for every search evaluation. Constructing
RMA evaluates the analytical formula, so use it only for the reference check
and final verification, after searching with array-based geometry and scoring.
Use the same settings for both NN and GP comparisons.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

if __package__:
    from .PatchFormula import RMA, AntennaBounds_mm
    from .GPRskeleton import GaussianProcessRegressor
else:
    from PatchFormula import RMA, AntennaBounds_mm
    from GPRskeleton import GaussianProcessRegressor


@dataclass(frozen=True)
class PSOConfig:
    # STEP 1: settings. One particle has three coordinates [W, L, h] in mm.
    """Starting choices, not lecturer requirements. Use identical NN/GP settings."""
    swarm_size: int = 40           # Number of candidate designs, N.
    iterations: int = 150          # Number of movement rounds, not a tolerance.
    inertia: float = 0.7           # w: retains some previous velocity.
    cognitive: float = 1.6         # c1: attraction to each particle's own best.
    social: float = 1.6            # c2: attraction to the whole swarm's best.
    velocity_fraction: float = 0.2 # Maximum step as a fraction of each bound span.
    seed: int = 1                  # Makes random choices reproducible.

    def __post_init__(self):
        for name in ("swarm_size", "iterations", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer.")
        if self.swarm_size < 2 or self.iterations < 0 or self.seed < 0:
            raise ValueError("Use swarm_size >= 2, iterations >= 0 and seed >= 0.")
        for name in ("inertia", "cognitive", "social"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not np.isfinite(self.velocity_fraction) or not 0 < self.velocity_fraction <= 1:
            raise ValueError("velocity_fraction must be in (0, 1].")


@dataclass
class PSOResult:
    # Index 0 is the initial swarm; subsequent entries follow each update.
    best_position: np.ndarray
    best_score: float
    best_score_history: np.ndarray
    best_position_history: np.ndarray
    objective_evaluations: int
    search_seconds: float


class ParticleSwarmOptimizer:
    def __init__(self, objective, bounds, config=None, repair=None):
        """objective(position) -> scalar to MINIMIZE; repair(position) -> copy.

        Bounds have shape (D, 2), one [lower, upper] pair per coordinate.
        An optional repair callback enforces additional constraints BEFORE
        every objective evaluation, including initialization.
        """
        self.objective, self.repair = objective, repair
        self.config = config if config is not None else PSOConfig()
        self.bounds = np.array(bounds, dtype=float, copy=True)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2 or len(self.bounds) == 0
                or not np.all(np.isfinite(self.bounds))
                or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError("bounds must contain finite increasing [lower, upper] pairs.")

    def _repair_positions(self, positions):
        fixed = np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])
        if self.repair is not None:
            fixed = np.asarray([self.repair(row.copy()) for row in fixed], dtype=float)
        if (fixed.shape != positions.shape or not np.all(np.isfinite(fixed))
                or np.any(fixed < self.bounds[:, 0]) or np.any(fixed > self.bounds[:, 1])):
            raise ValueError("The repair callback must return finite positions inside the bounds.")
        return fixed

    def _evaluate(self, positions):
        scores = np.array([float(self.objective(row.copy())) for row in positions])
        # +inf can mark an infeasible point; NaN/-inf break best comparisons.
        if np.any(np.isnan(scores)) or np.any(np.isneginf(scores)):
            raise ValueError("Objective returned NaN or negative infinity.")
        return scores

    def run(self):
        """Run standard global-best PSO; exactly config.iterations updates.

        v <- w*v + c1*r1*(personal_best-x) + c2*r2*(global_best-x)
        x <- repair(x + v)

        r1 and r2 are fresh random numbers for EVERY particle and coordinate.
        Personal best stores each particle's best position ever visited;
        global best is the best of all personal bests. Always copy these
        arrays so later position updates cannot overwrite the saved answers.
        """
        started = perf_counter()
        cfg = self.config
        # STEP 4: create positions and velocities, both shape (N, D).
        # D=3 for antennas. Repair even the initial positions before scoring.
        rng = np.random.default_rng(cfg.seed)  # A run is reproducible from its seed.
        span = self.bounds[:, 1] - self.bounds[:, 0]
        vmax = cfg.velocity_fraction * span
        shape = (cfg.swarm_size, len(span))
        positions = self._repair_positions(rng.uniform(self.bounds[:, 0], self.bounds[:, 1], shape))
        velocities = rng.uniform(-vmax, vmax, shape)
        scores = self._evaluate(positions)
        if not np.any(np.isfinite(scores)):
            raise ValueError("No finite objective score in the initial swarm.")
        # STEP 5: remember each personal best (N, D) and the global best (D,).
        # Copies preserve these memories when the current positions move.
        personal_best = positions.copy()
        personal_scores = scores.copy()
        best_index = int(np.argmin(personal_scores))
        global_best = personal_best[best_index].copy()
        global_score = float(personal_scores[best_index])
        score_history = [global_score]
        position_history = [global_best.copy()]

        for _ in range(cfg.iterations):
            # STEP 6: update velocity using momentum, personal best, global best.
            # global_best broadcasts from (D,) to (N, D) in the subtraction.
            r1, r2 = rng.random(shape), rng.random(shape)
            velocities = (cfg.inertia * velocities
                          + cfg.cognitive * r1 * (personal_best - positions)
                          + cfg.social * r2 * (global_best - positions))
            # Coordinate-wise limits matter: h has a much smaller range than W.
            velocities = np.clip(velocities, -vmax, vmax)
            # STEP 7: move, then repair constraints before evaluating anything.
            proposed = positions + velocities
            positions = self._repair_positions(proposed)
            # A wall/ratio repair interrupts motion along changed coordinates.
            # Reset those velocities rather than repeatedly pushing outwards.
            velocities[positions != proposed] = 0.0
            # STEP 8: score the new positions, update memories, save history.
            # improved is a Boolean array (N,); only better particles replace pbest.
            scores = self._evaluate(positions)
            improved = scores < personal_scores
            personal_best[improved] = positions[improved].copy()
            personal_scores[improved] = scores[improved]
            best_index = int(np.argmin(personal_scores))
            if personal_scores[best_index] < global_score:
                global_best = personal_best[best_index].copy()
                global_score = float(personal_scores[best_index])
            score_history.append(global_score)
            position_history.append(global_best.copy())

        # STEP 9: return the best EVER found, not just the final swarm position.
        return PSOResult(global_best, global_score, np.asarray(score_history),
                         np.asarray(position_history), cfg.swarm_size * (cfg.iterations + 1),
                         perf_counter() - started)


def geometry_is_feasible(design_mm):
    """STEP 2a: check RMA's bounds and ratio without evaluating its frequency."""
    design = np.asarray(design_mm, dtype=float)
    return bool(design.shape == (3,) and np.all(np.isfinite(design))
                and np.all(design >= AntennaBounds_mm[:, 0])
                and np.all(design <= AntennaBounds_mm[:, 1])
                and design[0] / design[1] >= 1.2)


def repair_antenna_design(design_mm):
    """STEP 2b: apply RMA's geometry repair rules without evaluating frequency."""
    design = np.array(design_mm, dtype=float, copy=True)
    if design.shape != (3,) or not np.all(np.isfinite(design)):
        raise ValueError("A design must contain three finite values [W, L, h].")
    max_length = np.nextafter(50.0 / 1.2, -np.inf)
    design[1] = np.clip(design[1], 25.0, max_length)
    design[0] = np.clip(design[0], 1.2 * design[1], 50.0)
    design[2] = np.clip(design[2], 1.0, 5.0)
    if design[0] / design[1] < 1.2:
        design[0] = np.nextafter(design[0], np.inf)
    return design


def make_antenna_objective(predict_frequency, penalty_weight=1.0, target_GHz=2.4):
    """STEP 3: create objective(design_mm) -> score with RMA.fitness's formula.

        J = (W*L*h)/11250 + penalty_weight * abs(predicted_f_GHz - 2.4)

    11250 = 50*45*5 mm^3 is a fixed box-volume reference, not a feasible
    geometry. The penalty uses frequency error IN GHz; if normalizing it by
    2.4 instead, rescale the weight consistently in GA, NN-PSO and GP-PSO.
    Use the SAME weight for all comparisons. A low score does not certify
    frequency feasibility; verify the final analytical target error separately.
    """
    if not np.isfinite(penalty_weight) or penalty_weight < 0:
        raise ValueError("penalty_weight must be finite and nonnegative.")
    if not np.isfinite(target_GHz) or target_GHz <= 0:
        raise ValueError("target_GHz must be finite and positive.")

    def objective(design_mm):
        if not geometry_is_feasible(design_mm):
            return float("inf")
        frequency = float(predict_frequency(design_mm))
        if not np.isfinite(frequency):
            raise ValueError("Surrogate returned a non-finite frequency.")
        return float(np.prod(design_mm) / 11250.0 + penalty_weight * abs(frequency - target_GHz))

    return objective


def verify_design(design_mm, predict_frequency, tolerance_GHz=0.01, target_GHz=2.4):
    """STEP 10: AFTER PSO, compare RMA's analytical frequency with the prediction."""
    if not np.isfinite(tolerance_GHz) or tolerance_GHz < 0:
        raise ValueError("tolerance_GHz must be finite and nonnegative.")
    started = perf_counter()
    predicted = float(predict_frequency(design_mm))
    prediction_seconds = perf_counter() - started
    started = perf_counter()
    antenna = RMA(*design_mm)
    actual = antenna.ResonanceF_GHz
    verification_seconds = perf_counter() - started
    geometry_ok = antenna.geometryCheck()
    target_error = abs(actual - target_GHz)
    return {"design_mm": np.asarray(design_mm).tolist(), "volume_mm3": antenna.volume_mm3,
            "predicted_frequency_GHz": predicted, "analytical_frequency_GHz": actual,
            "surrogate_target_deviation_MHz": 1000 * abs(predicted - target_GHz),
            "analytical_target_deviation_MHz": 1000 * target_error,
            "surrogate_prediction_error_MHz": 1000 * abs(predicted - actual),
            "geometry_feasible": geometry_ok,
            "frequency_acceptable": bool(target_error <= tolerance_GHz),
            "success": bool(geometry_ok and target_error <= tolerance_GHz),
            "final_surrogate_prediction_seconds": prediction_seconds,
            "analytical_verification_seconds": verification_seconds}


def load_neural_network(filename):
    """Use existing NN loaders, choosing architecture from saved metadata."""
    with np.load(filename, allow_pickle=False) as data:
        representation = str(data["data_representation"].item())
        activation = str(data["activation_hidden"].item()) if "activation_hidden" in data else "tanh"
    if representation == "scaled_mm_GHz_multilayer_tanh":
        if __package__:
            from .neuralnetworktest import NeuralNetwork
        else:
            from neuralnetworktest import NeuralNetwork
        return NeuralNetwork.load(filename, use_gpu=False)
    if representation == "scaled_mm_GHz" and activation == "tanh":
        if __package__:
            from .NerualNetwork import NerualNetwork
        else:
            from NerualNetwork import NerualNetwork
        return NerualNetwork.load(filename)
    if representation == "mm_GHz" and activation == "relu":
        if __package__:
            from .NeuralNetworkReLU import NeuralNetworkReLU
        else:
            from NeuralNetworkReLU import NeuralNetworkReLU
        return NeuralNetworkReLU.load(filename, device="cpu")
    if representation == "raw_mm_GHz" and activation == "tanh":
        if __package__:
            from .NeuralNetworkSkeleton import NeuralNetwork
        else:
            from NeuralNetworkSkeleton import NeuralNetwork
        return NeuralNetwork.load(filename)
    if representation == "mm_GHz" and activation == "tanh":
        # NerualNetwork.py saves raw-unit tanh weights with the same shapes.
        # Reuse the CPU skeleton's forward/predict to avoid a GPU dependency.
        if __package__:
            from .NeuralNetworkSkeleton import NeuralNetwork
        else:
            from NeuralNetworkSkeleton import NeuralNetwork
        with np.load(filename, allow_pickle=False) as data:
            model = NeuralNetwork(hidden_size=int(data["hidden_size"]), seed=int(data["seed"]))
            for name in ("W_hidden", "B_hidden", "W_output", "B_output"):
                setattr(model, name, data[name].copy())
        return model
    raise ValueError("Unsupported NN archive. Use a model saved by the current NN scripts.")


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surrogate", choices=("gp", "nn", "both"), default="both")
    parser.add_argument("--gp-model", type=Path, default=root / "results/gp/gp_model.npz")
    parser.add_argument("--nn-model", type=Path, default=root / "results/nn/nn_model_raw.npz")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--swarm-size", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--frequency-tolerance-GHz", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=root / "results/pso")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct seeds for independent repeated runs.")
    if not np.isfinite(args.frequency_tolerance_GHz) or args.frequency_tolerance_GHz < 0:
        parser.error("Frequency tolerance must be finite and nonnegative.")
    # Check the local analytical model before using it to verify any result.
    reference = RMA(38.0, 30.0, 1.6)
    if not np.isclose(reference.ResonanceF_GHz, 2.357548542, rtol=0, atol=1e-9):
        raise ValueError("PatchFormula failed the worksheet reference check.")
    selected = ("nn", "gp") if args.surrogate == "both" else (args.surrogate,)
    models = {}
    for name in selected:
        path = args.gp_model if name == "gp" else args.nn_model
        started = perf_counter()
        model = GaussianProcessRegressor.load(path) if name == "gp" else load_neural_network(path)
        models[name] = (model, path, perf_counter() - started)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"objective": "volume_mm3/11250 + penalty_weight*abs(predicted_GHz-2.4)",
              "penalty_weight": args.penalty_weight, "target_GHz": 2.4,
              "frequency_tolerance_GHz": args.frequency_tolerance_GHz,
              "models": {}, "runs": [], "summary": {}}
    for name, (model, path, load_seconds) in models.items():
        objective = make_antenna_objective(model.predict, args.penalty_weight)
        report["models"][name] = {"path": str(path.resolve()),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "load_seconds": load_seconds}
        for seed in args.seeds:
            config = PSOConfig(swarm_size=args.swarm_size, iterations=args.iterations, seed=seed)
            optimizer = ParticleSwarmOptimizer(objective, AntennaBounds_mm, config, repair_antenna_design)
            result = optimizer.run()
            checked = verify_design(result.best_position, model.predict, args.frequency_tolerance_GHz)
            row = {"surrogate": name, "seed": seed, "config": asdict(config),
                   "best_score": result.best_score, "search_seconds": result.search_seconds,
                   "objective_evaluations": result.objective_evaluations, **checked}
            report["runs"].append(row)
            with (args.output_dir / f"{name}_seed_{seed}_history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["iteration", "best_score", "W_mm", "L_mm", "h_mm"])
                for iteration, (score, design) in enumerate(zip(result.best_score_history, result.best_position_history)):
                    writer.writerow([iteration, score, *design])
            print(f"{name.upper()} seed {seed}: V={checked['volume_mm3']:.3f} mm^3, "
                  f"analytical target error={checked['analytical_target_deviation_MHz']:.4f} MHz, "
                  f"prediction error={checked['surrogate_prediction_error_MHz']:.4f} MHz, "
                  f"success={checked['success']}")
        rows = [row for row in report["runs"] if row["surrogate"] == name]
        successful = [row for row in rows if row["success"]]
        report["summary"][name] = {
            "runs": len(rows), "successes": len(successful),
            "success_rate": len(successful) / len(rows),
            "mean_search_seconds": float(np.mean([row["search_seconds"] for row in rows])),
            "mean_analytical_target_deviation_MHz": float(np.mean([row["analytical_target_deviation_MHz"] for row in rows])),
            "mean_successful_volume_mm3": (float(np.mean([row["volume_mm3"] for row in successful])) if successful else None),
        }
    output = args.output_dir / "comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    for name, summary in report["summary"].items():
        print(f"{name.upper()} success rate: {summary['successes']}/{summary['runs']}")
    # A failed design is a valid experimental outcome, saved and clearly marked.
    # Exit code 0 means the experiment completed, not that every design succeeded.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
