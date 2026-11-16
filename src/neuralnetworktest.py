"""Experiment copy of NerualNetwork.py with configurable tanh hidden layers.

Examples: hiddenSizes=(32,) adds neurons; (16, 16) adds a second hidden layer.
All derivatives and training updates are handwritten. NumPy/CuPy only supply
array operations and a linear solve. Public predict() takes mm and returns GHz.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


class NeuralNetwork:
    def __init__(self, hiddenSizes=(16, 16), seed=1, use_gpu=True):
        self.hiddenSizes = tuple(hiddenSizes)
        if not self.hiddenSizes or any(size < 1 for size in self.hiddenSizes):
            raise ValueError("Use at least one hidden layer, with positive sizes.")
        self.seed = seed
        self.xp = cp if use_gpu and cp is not None and cp.is_available() else np
        xp = self.xp
        rng = np.random.default_rng(seed)
        sizes = (3, *self.hiddenSizes, 1)
        self.weights, self.biases = [], []
        for inputs, outputs in zip(sizes[:-1], sizes[1:]):
            limit = np.sqrt(6 / (inputs + outputs))  # Xavier initialization
            self.weights.append(xp.asarray(rng.uniform(-limit, limit, (outputs, inputs))))
            self.biases.append(xp.zeros((outputs, 1)))
        self.history = {"training_loss": [], "validation_loss": []}
        self.best_epoch = 0
        self.training_settings = {}

    def to_numpy(self, array):
        return np.asarray(array) if self.xp is np else self.xp.asnumpy(array)

    def forward(self, X):
        """Internal forward pass: scaled (N, 3) -> scaled (N, 1)."""
        A = self.xp.asarray(X).T
        cache = [A]
        for layer, (W, B) in enumerate(zip(self.weights, self.biases)):
            Z = W @ A + B
            A = self.xp.tanh(Z) if layer < len(self.weights) - 1 else Z
            cache.append(A)
        return A.T, cache

    def loss(self, Y_pred, Y_actual):
        return float(self.xp.mean((Y_pred - Y_actual) ** 2))

    def backward(self, Y_pred, Y_actual, cache):
        """MSE gradients, extending the original backpropagation to any depth."""
        xp = self.xp
        delta = (2 / len(Y_actual)) * (Y_pred - Y_actual).T
        gradients = [None] * len(self.weights)
        for layer in reversed(range(len(self.weights))):
            gradients[layer] = (delta @ cache[layer].T,
                                xp.sum(delta, axis=1, keepdims=True))
            if layer > 0:
                delta = (self.weights[layer].T @ delta) * (1 - cache[layer] ** 2)
        return gradients

    def jacobian(self, cache):
        """One row per sample: derivative of its output w.r.t. all parameters.

        This is the same chain rule as backward(), without the MSE factor.
        A layer's weight derivatives have shape (samples, outputs, inputs).
        """
        xp = self.xp
        N = cache[0].shape[1]
        delta = xp.ones((1, N))
        blocks = [None] * (2 * len(self.weights))
        for layer in reversed(range(len(self.weights))):
            blocks[2 * layer] = xp.einsum("in,jn->nij", delta, cache[layer]).reshape(N, -1)
            blocks[2 * layer + 1] = delta.T
            if layer > 0:
                delta = (self.weights[layer].T @ delta) * (1 - cache[layer] ** 2)
        return xp.concatenate(blocks, axis=1)

    def parameters(self):
        return self.xp.concatenate([a.ravel() for pair in zip(self.weights, self.biases)
                                    for a in pair])

    def set_parameters(self, values):
        start = 0
        for array in [a for pair in zip(self.weights, self.biases) for a in pair]:
            array[:] = values[start:start + array.size].reshape(array.shape)
            start += array.size

    def fit(self, dataTrain, dataValidation, epochs=500, damping=0.001,
            target_khz=5.0, patience=80):
        """Full-batch Levenberg-Marquardt (damped least squares).

        Solve (J.T @ J + damping * I) step = J.T @ error, then subtract step.
        Increasing damping makes a rejected step smaller. No explicit inverse,
        automatic differentiation, or library optimizer is used.
        """
        if epochs < 1 or damping <= 0 or patience < 1:
            raise ValueError("epochs, damping, and patience must be positive.")
        xp = self.xp
        X, Y = xp.asarray(dataTrain["X_mm"]), xp.asarray(dataTrain["Y_GHz"])
        # Learn all four scaling statistics ONLY from the training split.
        self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0)
        self.Y_mean, self.Y_std = Y.mean(axis=0), Y.std(axis=0)
        self.X_std = xp.where(self.X_std == 0, 1.0, self.X_std)
        self.Y_std = xp.where(self.Y_std == 0, 1.0, self.Y_std)
        X = (X - self.X_mean) / self.X_std
        Y = (Y - self.Y_mean) / self.Y_std
        X_val = (xp.asarray(dataValidation["X_mm"]) - self.X_mean) / self.X_std
        Y_val = (xp.asarray(dataValidation["Y_GHz"]) - self.Y_mean) / self.Y_std
        scale_squared = float(self.Y_std[0]) ** 2
        self.history = {"training_loss": [], "validation_loss": []}
        self.training_settings = dict(epochs=epochs, damping=damping,
                                      target_khz=target_khz, patience=patience,
                                      optimizer="handwritten Levenberg-Marquardt")
        best_parameters = self.parameters().copy()
        best_loss = self.loss(self.forward(X_val)[0], Y_val)
        self.best_epoch = 0
        identity = xp.eye(best_parameters.size)

        for epoch in range(1, epochs + 1):
            prediction, cache = self.forward(X)
            training_loss = self.loss(prediction, Y)
            J = self.jacobian(cache)
            normal_matrix = J.T @ J
            gradient = J.T @ (prediction - Y).ravel()
            previous = self.parameters().copy()
            # Retry the same epoch with more damping if the loss increases.
            accepted = False
            for _ in range(15):
                step = xp.linalg.solve(normal_matrix + damping * identity, gradient)
                self.set_parameters(previous - step)
                candidate_loss = self.loss(self.forward(X)[0], Y)
                if np.isfinite(candidate_loss) and candidate_loss < training_loss:
                    training_loss = candidate_loss
                    damping = max(damping / 3, 1e-14)
                    accepted = True
                    break
                self.set_parameters(previous)
                damping *= 10
            validation_loss = self.loss(self.forward(X_val)[0], Y_val)
            # Store physical-unit MSE, so plotted RMSE is meaningful in kHz.
            self.history["training_loss"].append(training_loss * scale_squared)
            self.history["validation_loss"].append(validation_loss * scale_squared)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_parameters = self.parameters().copy()
                self.best_epoch = epoch
            validation_khz = np.sqrt(best_loss * scale_squared) * 1e6
            if validation_khz < target_khz or epoch - self.best_epoch >= patience or not accepted:
                break
        self.set_parameters(best_parameters)
        if xp is not np:
            xp.cuda.get_current_stream().synchronize()
        return self.history

    def predict(self, design_mm):
        """A single design returns float GHz; a batch returns a NumPy (N, 1) array."""
        X = self.xp.asarray(design_mm, dtype=float)
        single = X.ndim == 1
        X = X.reshape(-1, 3)
        prediction, _ = self.forward((X - self.X_mean) / self.X_std)
        result = self.to_numpy(prediction * self.Y_std + self.Y_mean)
        return float(result[0, 0]) if single else result

    def save(self, filename):
        data = {"hidden_sizes": self.hiddenSizes, "seed": self.seed,
                "data_representation": "scaled_mm_GHz_multilayer_tanh",
                "best_epoch": self.best_epoch,
                "training_settings": json.dumps(self.training_settings), **self.history}
        for name in ("X_mean", "X_std", "Y_mean", "Y_std"):
            data[name] = self.to_numpy(getattr(self, name))
        for layer, (W, B) in enumerate(zip(self.weights, self.biases)):
            data[f"W_{layer}"] = self.to_numpy(W)
            data[f"B_{layer}"] = self.to_numpy(B)
        np.savez(filename, **data)

    @classmethod
    def load(cls, filename, use_gpu=True):
        with np.load(filename) as data:
            if data["data_representation"].item() != "scaled_mm_GHz_multilayer_tanh":
                raise ValueError("Expected a model saved by neuralnetworktest.py.")
            model = cls(data["hidden_sizes"].tolist(), int(data["seed"]), use_gpu)
            for name in ("X_mean", "X_std", "Y_mean", "Y_std"):
                setattr(model, name, model.xp.asarray(data[name]))
            for layer in range(len(model.weights)):
                model.weights[layer] = model.xp.asarray(data[f"W_{layer}"])
                model.biases[layer] = model.xp.asarray(data[f"B_{layer}"])
            model.best_epoch = int(data["best_epoch"])
            model.training_settings = json.loads(data["training_settings"].item())
            model.history = {name: data[name].tolist() for name in model.history}
        return model


def rmse_khz(predicted_GHz, analytical_GHz):
    return float(np.sqrt(np.mean((predicted_GHz - analytical_GHz) ** 2)) * 1e6)


def audit_designs(count=10000, seed=5606):
    """Independent feasible designs for a final check, never used for tuning."""
    rng = np.random.default_rng(seed)
    batches, collected = [], 0
    while collected < count:
        X = rng.uniform([25, 25, 1], [50, 45, 5], size=(2 * count, 3))
        X = X[X[:, 0] / X[:, 1] >= 1.2]
        batches.append(X)
        collected += len(X)
    return np.concatenate(batches)[:count]


def main():
    import matplotlib.pyplot as plt
    from PatchFormula import RMA

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", default=["8", "16", "32", "16,8", "16,16"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output", type=Path, default=root / "results/nn/architecture_test")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(root / "antennaSample_training.npz") as data:
        X, Y = data["X_mm"], data["Y_GHz"]
        splits = {name: data[f"{name}_indices"] for name in ("train", "validation", "test")}
    np.savez(args.output / "split_indices.npz", **splits)
    training = {"X_mm": X[splits["train"]], "Y_GHz": Y[splits["train"]]}
    validation = {"X_mm": X[splits["validation"]], "Y_GHz": Y[splits["validation"]]}
    records, models = [], []
    print("Target: test RMSE < 10 kHz = 0.00001 GHz", flush=True)
    for architecture in args.architectures:
        sizes = tuple(int(size) for size in architecture.split(","))
        for seed in args.seeds:
            model = NeuralNetwork(sizes, seed, use_gpu=not args.cpu)
            start = perf_counter()
            model.fit(training, validation, epochs=args.epochs)
            elapsed = perf_counter() - start
            record = dict(hidden_sizes=list(sizes), seed=seed,
                          parameters=int(model.parameters().size),
                          device="CPU" if model.xp is np else "GPU",
                          epochs_run=len(model.history["training_loss"]),
                          best_epoch=model.best_epoch, fit_seconds=elapsed,
                          training_rmse_khz=rmse_khz(model.predict(training["X_mm"]), training["Y_GHz"]),
                          validation_rmse_khz=rmse_khz(model.predict(validation["X_mm"]), validation["Y_GHz"]))
            model.save(args.output / f"model_{architecture.replace(',', '_')}_seed{seed}.npz")
            records.append(record)
            models.append(model)
            print(f"{sizes}, seed {seed}, {record['device']}: validation RMSE "
                  f"{record['validation_rmse_khz']:.3f} kHz, {elapsed:.2f} s", flush=True)

    # Select using validation ONLY. The test split is first evaluated below.
    winner = min(range(len(records)), key=lambda i: records[i]["validation_rmse_khz"])
    best = models[winner]
    best.save(args.output / "best_model.npz")
    X_test = X[splits["test"]]
    verify_start = perf_counter()
    analytical = np.array([[RMA(*design).ResonanceF_GHz] for design in X_test])
    verification_seconds = perf_counter() - verify_start
    predict_start = perf_counter()
    predicted = best.predict(X_test)
    prediction_seconds = perf_counter() - predict_start
    test_rmse = rmse_khz(predicted, analytical)
    # A larger, independently generated check of the already selected model.
    audit_start = perf_counter()
    X_audit = audit_designs()
    audit_analytical = np.array([[RMA(*design).ResonanceF_GHz] for design in X_audit])
    audit_generation_seconds = perf_counter() - audit_start
    audit_predicted = best.predict(X_audit)
    audit_rmse = rmse_khz(audit_predicted, audit_analytical)
    success_rates = {}
    for architecture in args.architectures:
        sizes = list(map(int, architecture.split(",")))
        trials = [trial for trial in records if trial["hidden_sizes"] == sizes]
        success_rates[architecture] = sum(t["validation_rmse_khz"] < 10 for t in trials) / len(trials)
    summary = dict(target_khz=10, trials=records, selected_trial=winner,
                   training_settings=best.training_settings,
                   split_sizes={name: len(indices) for name, indices in splits.items()},
                   test_rmse_khz=test_rmse, target_met=bool(test_rmse < 10),
                   validation_success_rates=success_rates,
                   independent_audit=dict(samples=len(X_audit), seed=5606,
                       rmse_khz=audit_rmse, target_met=bool(audit_rmse < 10),
                       max_absolute_error_khz=float(np.max(abs(audit_predicted - audit_analytical)) * 1e6),
                       generation_and_label_seconds=audit_generation_seconds),
                   test_max_absolute_error_khz=float(np.max(abs(predicted - analytical)) * 1e6),
                   test_prediction_seconds=prediction_seconds,
                   analytical_verification_seconds=verification_seconds)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(args.output / "test_predictions.npz", X_mm=X_test,
             predicted_GHz=predicted, analytical_GHz=analytical)
    np.savez(args.output / "independent_audit.npz", X_mm=X_audit,
             predicted_GHz=audit_predicted, analytical_GHz=audit_analytical, seed=5606)
    print(f"Selected {best.hiddenSizes}, seed {best.seed}; test RMSE = {test_rmse:.3f} kHz")
    print("10 kHz target:", "PASS" if test_rmse < 10 else "NOT MET")
    print(f"Independent 10000-design audit RMSE: {audit_rmse:.3f} kHz")
    print("This is held-out test RMSE, not a maximum-error guarantee over all designs.")
    print(f"Saved models, histories, test predictions, and summary to {args.output}")
    for label, losses in best.history.items():
        plt.semilogy(range(1, len(losses) + 1), np.sqrt(losses) * 1e6,
                     label=label.replace("_loss", "").capitalize())
    plt.axhline(10, color="red", linestyle="--", label="10 kHz target")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (kHz)")
    plt.title(f"3 -> {' -> '.join(map(str, best.hiddenSizes))} -> 1, seed {best.seed}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output / "training_curve.png", dpi=150)
    if not args.no_show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
