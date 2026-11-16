"""Complete, commented NumPy Gaussian process regression for EE5606 Part II.

Run from the project root:
    python src/GPRskeleton.py
    python src/GPRskeleton.py --evaluate-test

The second command also evaluates the held-out test set. Use it only after
choosing the settings; do not tune against test errors. Both commands reuse
antennaSample_training.npz, so NN and GP can use exactly the same split.

Read in this order: kernel -> _factor -> fit -> predict -> save/load.
    X_mm: (N, 3), rows [W_mm, L_mm, h_mm]
    Y_GHz: (N, 1), analytical frequency labels
    K, L: (N, N), covariance and its lower Cholesky factor
    alpha: (N, 1), cached coefficients for future predictions

Unlike an NN, this GP does not learn neuron weights by backpropagation.
It describes how similar two designs are using a kernel, then conditions on
the training labels. A small, explicit grid search selects kernel settings
using validation RMSE. RBF and the grid are design choices, not PDF mandates.
Only NumPy and the standard library are used; no ML/optimization packages.

Equations: Rasmussen & Williams, GPML, chapter 2, Algorithm 2.1:
https://gaussianprocess.org/gpml/chapters/RW2.pdf
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def _inputs(X_mm):
    """Validate a nonempty batch of three-dimensional designs."""
    X = np.asarray(X_mm, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or len(X) == 0:
        raise ValueError("X_mm must have shape (N, 3), with N >= 1.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X_mm must contain finite values.")
    return X


def _dataset(data):
    X = _inputs(data["X_mm"])
    Y = np.asarray(data["Y_GHz"], dtype=float)
    if Y.shape == (len(X),):
        Y = Y[:, None]
    if Y.shape != (len(X), 1) or not np.all(np.isfinite(Y)):
        raise ValueError("Y_GHz must contain one finite frequency per input row.")
    return X, Y


def regression_metrics(actual, predicted):
    """Errors are in MHz; the public prediction interface uses GHz."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted frequency shapes must match.")
    error = (predicted - actual) * 1000.0
    return {"rmse_MHz": float(np.sqrt(np.mean(error**2))),
            "mae_MHz": float(np.mean(np.abs(error))),
            "max_absolute_error_MHz": float(np.max(np.abs(error)))}


class GaussianProcessRegressor:
    def __init__(self, jitter=1e-10):
        # Jitter is numerical stabilization, not measured antenna noise.
        if not np.isfinite(jitter) or jitter <= 0:
            raise ValueError("jitter must be finite and positive.")
        self.jitter = float(jitter)
        self.fitted = False
        self.history = []
        self.metadata = {}

    @staticmethod
    def kernel(X, Z, length_scale=1.0, signal_std=1.0):
        """RBF covariance between standardized rows: result shape (N, M).

        k(x,z) = signal_std^2 * exp(-0.5 * sum(((x-z)/ell)^2)).
        ell may be a scalar or three length scales, one for W, L and h.
        Large ell means frequency is assumed to vary more smoothly.
        signal_std is in standardized output units, not GHz.
        """
        X, Z = _inputs(X), _inputs(Z)
        ell = np.asarray(length_scale, dtype=float)
        if ell.shape not in ((), (3,)) or not np.all(np.isfinite(ell)) or np.any(ell <= 0):
            raise ValueError("length_scale must be positive: scalar or shape (3,).")
        if not np.isfinite(signal_std) or signal_std <= 0:
            raise ValueError("signal_std must be finite and positive.")
        differences = (X[:, None, :] - Z[None, :, :]) / ell  # (N, M, 3)
        return signal_std**2 * np.exp(-0.5 * np.sum(differences**2, axis=2))

    def _factor(self, length_scale, signal_std, noise_std):
        """Build covariance, solve for alpha, and calculate training evidence."""
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative.")
        K = self.kernel(self.X_train, self.X_train, length_scale, signal_std)
        # Tiny noise is regularization for our deterministic analytical labels.
        # Never explicitly invert K: Cholesky + solves is more stable.
        used_jitter = self.jitter
        identity = np.eye(len(K))
        for _ in range(9):
            try:
                L = np.linalg.cholesky(K + (noise_std**2 + used_jitter) * identity)
                break
            except np.linalg.LinAlgError:
                used_jitter *= 10.0
        else:
            raise np.linalg.LinAlgError("Covariance remained unstable after jitter retries.")
        # L @ L.T = covariance. Solve L b = y, then L.T alpha = b.
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.Y_train))
        nll = (0.5 * float((self.Y_train.T @ alpha).item())
               + float(np.log(np.diag(L)).sum())
               + 0.5 * len(K) * np.log(2.0 * np.pi))
        return L, alpha, float(nll), used_jitter

    def fit(self, dataTrain, dataValidation, *,
            length_scales=(0.5, 1.0, 2.0, 4.0, 8.0),
            signal_stds=(1.0,), noise_stds=(1e-6, 1e-4)):
        """Fit on training rows and select settings using validation RMSE.

        Both arguments are dictionaries with X_mm and Y_GHz, as in your NN.
        Scaling statistics come ONLY from training rows. Validation labels
        select hyperparameters but never enter the GP covariance solve.
        Test data is not an argument. Returns one history row per candidate.
        """
        X, Y = _dataset(dataTrain)
        X_val, Y_val = _dataset(dataValidation)
        candidates = list(itertools.product(length_scales, signal_stds, noise_stds))
        if not candidates:
            raise ValueError("Provide at least one candidate for each hyperparameter.")
        self.fitted = False
        self.metadata = {}
        started = perf_counter()
        self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0)
        self.Y_mean, self.Y_std = Y.mean(axis=0), Y.std(axis=0)
        # A constant feature/target should not cause division by zero.
        self.X_std = np.where(self.X_std > 0, self.X_std, 1.0)
        self.Y_std = np.where(self.Y_std > 0, self.Y_std, 1.0)
        self.X_train = (X - self.X_mean) / self.X_std
        self.Y_train = (Y - self.Y_mean) / self.Y_std
        scaled_val = (X_val - self.X_mean) / self.X_std
        self.history, best_rmse = [], np.inf
        for length_scale, signal_std, noise_std in candidates:
            candidate_started = perf_counter()
            L, alpha, nll, used_jitter = self._factor(length_scale, signal_std, noise_std)
            K_val = self.kernel(scaled_val, self.X_train, length_scale, signal_std)
            predicted = (K_val @ alpha) * self.Y_std + self.Y_mean
            rmse = regression_metrics(Y_val, predicted)["rmse_MHz"]
            if not np.isfinite(rmse):
                raise ValueError("Non-finite GP prediction; check data and kernel settings.")
            self.history.append({
                "length_scale": np.asarray(length_scale).tolist(),
                "signal_std": float(signal_std), "noise_std": float(noise_std),
                "jitter_used": used_jitter, "validation_rmse_MHz": rmse,
                "negative_log_marginal_likelihood": nll,
                "seconds": perf_counter() - candidate_started,
            })
            if rmse < best_rmse:
                best_rmse = rmse
                self.length_scale = np.asarray(length_scale, dtype=float).copy()
                self.signal_std, self.noise_std = float(signal_std), float(noise_std)
                self.L, self.alpha = L, alpha
                self.jitter_used, self.nll = used_jitter, nll
        self.fit_seconds = perf_counter() - started
        self.best_validation_rmse_MHz = best_rmse
        self.fitted = True
        return self.history

    def predict(self, design_mm, return_std=False):
        """Return GHz: float for (3,), or (N, 1) for a batch (N, 3).

        With return_std=True return (mean, standard_deviation), both in GHz.
        The uncertainty is for the underlying frequency, excluding observation
        noise and jitter; it is not a guarantee of analytical accuracy.
        PSO uses only the mean. No training or analytical formula runs here.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() or load() before predict().")
        X = np.asarray(design_mm, dtype=float)
        single = X.ndim == 1
        X = _inputs(X[None, :] if single else X)
        X = (X - self.X_mean) / self.X_std
        K_star = self.kernel(X, self.X_train, self.length_scale, self.signal_std)
        mean = (K_star @ self.alpha) * self.Y_std + self.Y_mean
        mean_result = float(mean[0, 0]) if single else mean
        if not return_std:
            return mean_result
        # var(f*) = k(x*,x*) - k_*^T covariance^-1 k_*.
        # Solving L v = k_* gives the equivalent, stable k(x*,x*) - v^T v.
        v = np.linalg.solve(self.L, K_star.T)
        variance = self.signal_std**2 - np.sum(v**2, axis=0)
        std = np.sqrt(np.maximum(variance, 0.0))[:, None] * self.Y_std
        return mean_result, float(std[0, 0]) if single else std

    def save(self, filename):
        """Save scaling, training rows, cached factors, search history and metadata."""
        if not self.fitted:
            raise RuntimeError("Fit before saving.")
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, model_type="numpy_rbf_gp_v1", jitter=self.jitter,
                 X_train=self.X_train, Y_train=self.Y_train,
                 X_mean=self.X_mean, X_std=self.X_std,
                 Y_mean=self.Y_mean, Y_std=self.Y_std,
                 length_scale=self.length_scale, signal_std=self.signal_std,
                 noise_std=self.noise_std, jitter_used=self.jitter_used,
                 L=self.L, alpha=self.alpha, nll=self.nll,
                 fit_seconds=self.fit_seconds,
                 best_validation_rmse_MHz=self.best_validation_rmse_MHz,
                 history_json=json.dumps(self.history), metadata_json=json.dumps(self.metadata))

    @classmethod
    def load(cls, filename):
        """Restore inference directly; loading does not fit or tune again."""
        with np.load(filename, allow_pickle=False) as data:
            if data["model_type"].item() != "numpy_rbf_gp_v1":
                raise ValueError("This archive is not a GPRskeleton model.")
            model = cls(jitter=float(data["jitter"]))
            for name in ("X_train", "Y_train", "X_mean", "X_std", "Y_mean", "Y_std",
                         "length_scale", "L", "alpha"):
                setattr(model, name, data[name].copy())
            for name in ("signal_std", "noise_std", "jitter_used", "nll", "fit_seconds",
                         "best_validation_rmse_MHz"):
                setattr(model, name, float(data[name]))
            model.history = json.loads(data["history_json"].item())
            model.metadata = json.loads(data["metadata_json"].item())
        model.fitted = True
        return model


def load_prepared_split(filename):
    """Reuse and check the saved NN split, without reshuffling or relabelling."""
    with np.load(filename, allow_pickle=False) as data:
        X, Y = _dataset(data)
        names = ("train_indices", "validation_indices", "test_indices")
        indices = [data[name].copy() for name in names]
        for index in indices:
            if index.ndim != 1 or len(index) == 0 or index.dtype.kind not in "iu":
                raise ValueError("Split indices must be nonempty integer vectors.")
        combined = np.concatenate(indices)
        if len(combined) != len(X) or not np.array_equal(np.sort(combined), np.arange(len(X))):
            raise ValueError("Train/validation/test must partition the dataset without overlap.")
        metadata = {name: index.tolist() for name, index in zip(names, indices)}
        for name in ("data_seed", "split_seed"):
            if name in data:
                metadata[name] = int(data[name])
    metadata.update(dataset_path=str(Path(filename).resolve()),
                    dataset_sha256=hashlib.sha256(Path(filename).read_bytes()).hexdigest())
    splits = [{"X_mm": X[index], "Y_GHz": Y[index]} for index in indices]
    return (*splits, metadata)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "antennaSample_training.npz")
    parser.add_argument("--output-dir", type=Path, default=root / "results/gp")
    parser.add_argument("--length-scales", nargs="+", type=float, default=[0.5, 1, 2, 4, 8])
    parser.add_argument("--signal-stds", nargs="+", type=float, default=[1.0])
    parser.add_argument("--noise-stds", nargs="+", type=float, default=[1e-6, 1e-4])
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args(argv)
    started = perf_counter()
    train, validation, test, metadata = load_prepared_split(args.data)
    load_seconds = perf_counter() - started
    model = GaussianProcessRegressor()
    model.fit(train, validation, length_scales=args.length_scales,
              signal_stds=args.signal_stds, noise_stds=args.noise_stds)
    model.metadata = metadata
    model_path = args.output_dir / "gp_model.npz"
    model.save(model_path)
    report = {"model": str(model_path), "data_loading_seconds": load_seconds,
              "dataset_generation_seconds": None,  # Existing data has no recorded generation timing.
              "fit_and_selection_seconds": model.fit_seconds,
              "training_samples": len(train["X_mm"]), "validation_samples": len(validation["X_mm"]),
              "training": regression_metrics(train["Y_GHz"], model.predict(train["X_mm"])),
              "validation": regression_metrics(validation["Y_GHz"], model.predict(validation["X_mm"])),
              "hyperparameter_search": model.history, "test_evaluated": args.evaluate_test}
    if args.evaluate_test:
        started = perf_counter()
        prediction = model.predict(test["X_mm"])
        report["test_prediction_seconds"] = perf_counter() - started
        report["test"] = regression_metrics(test["Y_GHz"], prediction)
        np.savez(args.output_dir / "test_predictions.npz", X_mm=test["X_mm"],
                 actual_GHz=test["Y_GHz"], predicted_GHz=prediction,
                 test_indices=metadata["test_indices"])
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {model_path}")
    print(f"Selected length scale={model.length_scale}, signal std={model.signal_std:g}, noise std={model.noise_std:g}")
    print(f"Validation RMSE: {model.best_validation_rmse_MHz:.6f} MHz; fit/search: {model.fit_seconds:.3f} s")
    if args.evaluate_test:
        print(f"Test RMSE: {report['test']['rmse_MHz']:.6f} MHz")
    else:
        print("Test labels were not evaluated. Use --evaluate-test after choosing settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
