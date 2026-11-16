from pathlib import Path
import numpy as np
from time import perf_counter

# CuPy runs the same array operations on the GPU; NumPy is the CPU fallback.
# Install GPU support: python -m pip install "cupy-cuda12x[ctk]" matplotlib
try:
    import cupy as cp
    xp = cp if cp.is_available() else np
except (ImportError, OSError, RuntimeError):
    xp = np


def approximate_line_search(evaluate, initial_step=1.0, beta=1e-4, sigma=0.1,
                            max_iterations=40):
    """Strong Wolfe search from optimization methods(A), slides 23, 34, 44-48.

    evaluate(alpha) returns f(x + alpha*d) and grad(f).T @ d.
    Accept sufficient decrease AND |slope| <= -sigma * initial_slope.
    Double alpha to bracket, then bisect (zoom). No library optimizer is used.
    """
    if not (np.isfinite(initial_step) and initial_step > 0
            and 0 < beta < sigma < 1 and max_iterations >= 1):
        raise ValueError("Require alpha > 0, 0 < beta < sigma < 1, and a positive search limit.")
    y0, g0 = evaluate(0.0)
    if not (np.isfinite(y0) and np.isfinite(g0)) or g0 >= 0:
        raise ValueError("Line search requires a finite loss and a descent direction.")

    def zoom(lo, hi, y_lo):
        for _ in range(max_iterations):
            alpha = (lo + hi) / 2
            if alpha == lo or alpha == hi:
                break
            y, g = evaluate(alpha)
            if (not np.isfinite(y) or not np.isfinite(g)
                    or y > y0 + beta * alpha * g0 or y >= y_lo):
                hi = alpha
            else:
                if abs(g) <= -sigma * g0:
                    return alpha
                if g * (hi - lo) >= 0:
                    hi = lo  # Keep the OLD lo before replacing it (slide 48).
                lo, y_lo = alpha, y
        raise RuntimeError("Strong Wolfe zoom did not find an acceptable step.")

    previous, y_previous, alpha = 0.0, y0, initial_step
    for iteration in range(max_iterations):
        y, g = evaluate(alpha)
        if (not np.isfinite(y) or not np.isfinite(g)
                or y > y0 + beta * alpha * g0
                or (iteration > 0 and y >= y_previous)):
            return zoom(previous, alpha, y_previous)
        if abs(g) <= -sigma * g0:
            return alpha
        if g >= 0:
            return zoom(alpha, previous, y)
        previous, y_previous = alpha, y
        alpha *= 2
    raise RuntimeError("Strong Wolfe bracketing did not find an acceptable step.")


class NerualNetwork:
    inputSize = 3
    outputSize = 1
    activationHidden = "tanh"
    def __init__(self, hiddenSize, seed):
        self.hiddenSize = hiddenSize
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # fit() learns these from training data only; identity scaling before fit.
        self.X_mean, self.X_std = xp.zeros((1, 3)), xp.ones((1, 3))
        self.Y_mean, self.Y_std = xp.zeros((1, 1)), xp.ones((1, 1))
        # Xavier initialization for tanh with standardized inputs.
        self.limit_hidden = np.sqrt(6/(self.inputSize+self.hiddenSize))
        self.W_hidden = xp.asarray(self.rng.uniform(-self.limit_hidden,self.limit_hidden, size=(self.hiddenSize,self.inputSize)))
        self.B_hidden = xp.zeros((self.hiddenSize,1))
        self.limit_output = np.sqrt(6/(self.hiddenSize+self.outputSize))
        self.W_output = xp.asarray(self.rng.uniform(-self.limit_output,self.limit_output, size=(self.outputSize,self.hiddenSize)))
        self.B_output = xp.zeros((self.outputSize,1))
        
        self.history = {"training_loss": [], "validation_loss": []}
        self.training_settings={}
        self.best_epoch=0
        self.step_sizes = []
        self.stop_reason = "not_trained"
    def activationFunction(self,Z):
        if self.activationHidden == "tanh":
            return xp.tanh(Z)
    def forward(self, X):
        # Internal forward pass uses standardized inputs and outputs.
        # Use predict() for a design in mm and a frequency in GHz.
        X = xp.asarray(X)
        Z_hidden = self.W_hidden @ X.T + self.B_hidden # (hiddenSize, 3) @ (3, N)
        A_hidden = self.activationFunction(Z_hidden) # (hiddenSize, N)
        Z_output = self.W_output @ A_hidden + self.B_output # (1, hiddenSize) @ (hiddenSize, N)
        cache = {"X": X.T, "Z_hidden": Z_hidden, "A_hidden": A_hidden}
        return Z_output.T, cache # (N, 1)
    def loss(self, Y_pred, Y_actually):
        # return MSE
        return float(xp.mean(xp.square(Y_pred-xp.asarray(Y_actually))))
    def backward(self, Y_pred, Y_actual, cache :dict):
        # backward propagation using gradient
        Y_actual = xp.asarray(Y_actual)
        X = cache["X"]
        Z_hidden = cache["Z_hidden"]
        A_hidden = cache["A_hidden"]
        N = X.shape[1]
        E_output = (Y_pred- Y_actual) * (2.0/N) # (N, 1)
        dW_output = E_output.T @ A_hidden.T # (1, hiddenSize)
        dB_output = xp.sum(E_output, axis=0, keepdims=True)
        # different activation function have different E_hidden
        if(self.activationHidden == "tanh"):
            E_hidden = (self.W_output.T @ E_output.T) * (1- A_hidden**2) # (hiddenSize, N)
        dW_hidden = E_hidden @ X.T # (hiddenSize, 3)
        dB_hidden = xp.sum(E_hidden, axis =1, keepdims = True) # (hiddenSize, 1)
        return {
            "dW_hidden": dW_hidden,
            "dB_hidden": dB_hidden,
            "dW_output": dW_output,
            "dB_output": dB_output,
        }
    def update_parameters(self, gradients:dict, learning_rate):
        # update W and B
        self.W_output -= learning_rate* gradients["dW_output"]
        self.B_output -= learning_rate* gradients["dB_output"]
        self.W_hidden -= learning_rate* gradients["dW_hidden"]
        self.B_hidden -= learning_rate* gradients["dB_hidden"]       
    def line_search_step(self, X, Y, gradients, initial_step, beta, sigma, max_iterations):
        # x in the slides represents ALL weights and biases; d = -gradient.
        names = ("W_hidden", "B_hidden", "W_output", "B_output")
        original = {name: getattr(self, name).copy() for name in names}
        norm_squared = float(sum(xp.sum(g*g) for g in gradients.values()))
        if norm_squared == 0:
            return 0.0

        def evaluate(alpha):
            # Every trial starts from the same parameters, never a rejected trial.
            for name in names:
                getattr(self, name)[:] = original[name] - alpha * gradients["d" + name]
            prediction, cache = self.forward(X)
            value = self.loss(prediction, Y)
            new_gradients = self.backward(prediction, Y, cache)
            slope = -float(sum(xp.sum(new_gradients[key]*gradients[key]) for key in gradients))
            return value, slope

        try:
            return approximate_line_search(evaluate, initial_step, beta, sigma, max_iterations)
        except Exception:
            for name in names:
                getattr(self, name)[:] = original[name]
            raise

    def fit(
        self,
        dataTrain,
        dataValidation,
        epochs = 1000,
        learning_rate = 1.0,  # Initial trial alpha when line_search=True.
        batch_size = None,  # None uses the full training set for line search.
        patience = 2000,
        target_rmse_khz = 10.0,
        verbose = False,
        line_search = True,
        beta = 1e-4,
        sigma = 0.1,
        max_line_search_iterations = 40,
    ):
        if batch_size is None:
            batch_size = len(dataTrain["X_mm"])
        if epochs < 1 or batch_size < 1 or patience < 1 or target_rmse_khz < 0:
            raise ValueError("Use positive epochs, batch_size, patience and a nonnegative RMSE target.")
        if not np.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative.")
        if line_search and not (learning_rate > 0 and 0 < beta < sigma < 1
                                and max_line_search_iterations >= 1):
            raise ValueError("Require alpha > 0, 0 < beta < sigma < 1, and a positive search limit.")
        # Transfer the datasets once; batches and parameters stay on the GPU.
        X_train = xp.asarray(dataTrain["X_mm"])
        Y_train = xp.asarray(dataTrain["Y_GHz"])
        X_validation = xp.asarray(dataValidation["X_mm"])
        Y_validation = xp.asarray(dataValidation["Y_GHz"])
        self.X_mean = X_train.mean(axis=0, keepdims=True)
        self.X_std = X_train.std(axis=0, keepdims=True)
        self.Y_mean = Y_train.mean(axis=0, keepdims=True)
        self.Y_std = Y_train.std(axis=0, keepdims=True)
        self.X_std = xp.where(self.X_std == 0, 1.0, self.X_std)
        self.Y_std = xp.where(self.Y_std == 0, 1.0, self.Y_std)
        X_train = (X_train-self.X_mean)/self.X_std
        Y_train = (Y_train-self.Y_mean)/self.Y_std
        X_validation = (X_validation-self.X_mean)/self.X_std
        Y_validation = (Y_validation-self.Y_mean)/self.Y_std
        loss_scale = float(self.Y_std[0, 0])**2
        
        self.history = {"training_loss" :[], "validation_loss": []}
        self.best_epoch =0
        self.step_sizes = []
        self.stop_reason = "epoch_limit"
        self.training_settings = {
            "epochs":epochs, 
            "learning_rate":learning_rate, 
            "batch_size":batch_size,
            "patience":patience,
            "target_rmse_khz":target_rmse_khz,
            "line_search":line_search, "beta":beta, "sigma":sigma,
            "max_line_search_iterations":max_line_search_iterations,
        }
        best_validation_loss = np.inf
        best_parameters=(
            self.W_hidden.copy(),self.B_hidden.copy(),
            self.W_output.copy(),self.B_output.copy(),
        )
        # one epoch visits every training sample once
        for epoch in range(epochs):
            stalled = False
            # randomlize the data set order
            order = (xp.arange(len(X_train)) if batch_size >= len(X_train)
                     else xp.asarray(self.rng.permutation(len(X_train))))
            for start in range(0, len(X_train), batch_size):
                indices = order[start:start+batch_size]
                X_batch = X_train[indices]
                Y_batch = Y_train[indices]
                Y_pred , cache = self.forward(X_batch)
                gradients = self.backward(Y_pred= Y_pred, Y_actual= Y_batch, cache=cache)
                if line_search:
                    try:
                        alpha = self.line_search_step(X_batch, Y_batch, gradients,
                            learning_rate, beta, sigma, max_line_search_iterations)
                    except RuntimeError as error:
                        self.stop_reason = str(error)
                        stalled = True
                        break
                    if alpha == 0:
                        self.stop_reason = "zero_gradient"
                        stalled = True
                else:
                    alpha = learning_rate
                    self.update_parameters(gradients=gradients, learning_rate=alpha)
                self.step_sizes.append(alpha)
                if stalled:
                    break
            
            # Validates the model
            training_pred, _ = self.forward(X_train)
            validation_pred, _ = self.forward(X_validation)
            # Keep histories in GHz^2 although gradients use standardized targets.
            training_loss = self.loss(training_pred, Y_train)*loss_scale
            validation_loss = self.loss(validation_pred, Y_validation)*loss_scale
            self.history["training_loss"].append(training_loss)
            self.history["validation_loss"].append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self.best_epoch = epoch + 1
                best_parameters = (
                    self.W_hidden.copy(), self.B_hidden.copy(),
                    self.W_output.copy(), self.B_output.copy(),
                )
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}: validation RMSE = {np.sqrt(validation_loss)*1e6:.3f} kHz", flush=True)
            # Stop at the target, after no validation improvement, or at epochs.
            if np.sqrt(best_validation_loss)*1e6 < target_rmse_khz:
                self.stop_reason = "validation_target"
                break
            if epoch + 1 - self.best_epoch >= patience:
                self.stop_reason = "validation_patience"
                break
            if stalled:
                break
        self.W_hidden, self.B_hidden, self.W_output, self.B_output = best_parameters            
        return self.history
    def predict(self, design_mm):
        X_mm = xp.asarray(design_mm, dtype=float).reshape(1,3)
        Y_scaled , _ = self.forward((X_mm-self.X_mean)/self.X_std)
        Y_GHz = Y_scaled*self.Y_std+self.Y_mean
        return float(Y_GHz[0,0])
    def save(self,filename):
        # NPZ files use CPU arrays so they can also be loaded without a GPU.
        to_numpy = np.asarray if xp is np else xp.asnumpy
        np.savez(
            filename,
            W_hidden=to_numpy(self.W_hidden), B_hidden=to_numpy(self.B_hidden),
            W_output=to_numpy(self.W_output), B_output=to_numpy(self.B_output),
            data_representation="scaled_mm_GHz",
            X_mean=to_numpy(self.X_mean), X_std=to_numpy(self.X_std),
            Y_mean=to_numpy(self.Y_mean), Y_std=to_numpy(self.Y_std),
            hidden_size=self.hiddenSize, 
            seed=self.seed,
            best_epoch=self.best_epoch,
            epochs=self.training_settings["epochs"],
            learning_rate=self.training_settings["learning_rate"],
            batch_size=self.training_settings["batch_size"],
            patience=self.training_settings["patience"],
            target_rmse_khz=self.training_settings["target_rmse_khz"],
            line_search=self.training_settings["line_search"],
            beta=self.training_settings["beta"], sigma=self.training_settings["sigma"],
            max_line_search_iterations=self.training_settings["max_line_search_iterations"],
            step_sizes=self.step_sizes, stop_reason=self.stop_reason,
            training_loss=self.history["training_loss"],
            validation_loss=self.history["validation_loss"],
        )
    @classmethod
    def load(cls, filename):
        with np.load(filename) as data:
            if("data_representation" not in data
               or data ["data_representation"].item() not in ("mm_GHz", "scaled_mm_GHz")):
                raise ValueError("Load a model saved by NerualNetwork.py.")
            model = cls(
                    hiddenSize=int(data["hidden_size"]),
                    seed=int(data["seed"]),)
            # Older raw-unit models keep identity scaling for compatibility.
            if data["data_representation"].item() == "scaled_mm_GHz":
                for name in ("X_mean", "X_std", "Y_mean", "Y_std"):
                    setattr(model, name, xp.asarray(data[name]))
            model.W_hidden = xp.asarray(data["W_hidden"])
            model.B_hidden = xp.asarray(data["B_hidden"])
            model.W_output = xp.asarray(data["W_output"])
            model.B_output = xp.asarray(data["B_output"])
            model.best_epoch = int(data["best_epoch"])
            model.training_settings = {
                            "epochs": int(data["epochs"]),
                            "learning_rate": float(data["learning_rate"]),
                            "batch_size": int(data["batch_size"]),
                            "patience": int(data["patience"]) if "patience" in data else 2000,
                            "target_rmse_khz": float(data["target_rmse_khz"]) if "target_rmse_khz" in data else 10.0,
                            "line_search": bool(data["line_search"]) if "line_search" in data else False,
                            "beta": float(data["beta"]) if "beta" in data else 1e-4,
                            "sigma": float(data["sigma"]) if "sigma" in data else 0.1,
                            "max_line_search_iterations": int(data["max_line_search_iterations"]) if "max_line_search_iterations" in data else 40,
                        }
            model.step_sizes = data["step_sizes"].tolist() if "step_sizes" in data else []
            model.stop_reason = data["stop_reason"].item() if "stop_reason" in data else "loaded_legacy"
            model.history = {
                            "training_loss": data["training_loss"].tolist(),
                            "validation_loss": data["validation_loss"].tolist(),
                        }
        return model


def plot_prediction_comparison(model, X_test_mm, output_path):
    """Compare a trained model with the analytical formula on held-out designs."""
    import matplotlib.pyplot as plt
    if __package__:
        from .PatchFormula import RMA
    else:
        from PatchFormula import RMA

    analytical = np.array([RMA(*design).ResonanceF_GHz for design in X_test_mm])
    predicted = np.array([model.predict(design) for design in X_test_mm])
    error_khz = (predicted - analytical) * 1e6
    rmse_khz = np.sqrt(np.mean(error_khz**2))
    print(f"Test RMSE against analytical formula: {rmse_khz:.3f} kHz")
    print(f"Test maximum absolute error: {np.max(np.abs(error_khz)):.3f} kHz")

    # Sort both arrays together for a readable overlay. This is not a geometry sweep.
    order = np.argsort(analytical)
    sample = np.arange(1, len(order) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(sample, analytical[order], color="black", label="Analytical formula")
    axes[0].plot(sample, predicted[order], "o--", color="tab:orange",
                 markersize=3, linewidth=1, label="NN prediction")
    axes[0].set_ylabel("Resonant frequency (GHz)")
    axes[0].set_title(f"Analytical formula vs NN: {len(order)} test designs\n"
                      f"Test RMSE = {rmse_khz:.3f} kHz")
    axes[0].legend()
    axes[1].plot(sample, error_khz[order], ".-", color="tab:blue", linewidth=1)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("NN - analytical (kHz)")
    axes[1].set_xlabel("Test design rank (sorted by analytical frequency)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved prediction comparison: {output_path}")
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root/"antennaSample_training.npz"
    with np.load(dataset_path) as data:
        train_indices = data["train_indices"]
        validation_indices = data["validation_indices"]
        X_test_mm = data["X_mm"][data["test_indices"]]
        
        dataTrain = {
            "X_mm": data["X_mm"][train_indices],
            "Y_GHz": data["Y_GHz"][train_indices]
        }
        dataValidation = {
            "X_mm": data["X_mm"][validation_indices],
            "Y_GHz": data["Y_GHz"][validation_indices]
        }
        model = NerualNetwork(hiddenSize = 32, seed=1)
    print("Training shapes:", dataTrain["X_mm"].shape, dataTrain["Y_GHz"].shape)
    print("Validation shapes:", dataValidation["X_mm"].shape, dataValidation["Y_GHz"].shape)
    print("Weight shapes:", model.W_hidden.shape, model.W_output.shape)
    print("Bias shapes:", model.B_hidden.shape, model.B_output.shape)
    print("Public units: X in mm, Y in GHz; training uses standardized X and Y")
    print("Scaling statistics are fitted on training data only")
    print("Architecture: 3 -> 32 tanh -> 1 linear output (161 trainable parameters)")
    print("Training device:", "CPU (NumPy)" if xp is np else "GPU (CuPy)")
    
    fit_start = perf_counter()
    # Slides 44-48: strong Wolfe line search selects alpha from an initial trial of 1.
    # Full-batch gradients and trial losses use the same 420 training samples.
    history = model.fit(dataTrain=dataTrain,dataValidation=dataValidation,learning_rate=1.0,epochs=30000,verbose=True)
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()  # Finish GPU work before timing.
    fit_time = perf_counter() - fit_start
    print(f"Training time: {fit_time:.3f} s")
    print(f"Best validation epoch: {model.best_epoch}")
    print(f"Best validation MSE (GHz^2): {min(history['validation_loss']):.6g}")
    print(f"Best validation RMSE: {np.sqrt(min(history['validation_loss']))*1e6:.3f} kHz")
    print(f"Stopped: {model.stop_reason}")
    model_path = project_root / "results" / "nn" / "nn_model_line_search.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved model: {model_path}")

    design = [35.614, 29.6778, 1.0]
    restored = NerualNetwork.load(model_path)
    print(
        "Reloaded prediction:",
        restored.predict(design),
        "GHz",
        )

    epochs = range(1, len(history["training_loss"]) + 1)
    plt.semilogy(epochs, np.sqrt(history["training_loss"])*1e6, label="Training")
    plt.semilogy(epochs, np.sqrt(history["validation_loss"])*1e6, label="Validation")
    # plt.axhline(10, color="red", linestyle="--", label="10 kHz target")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (kHz)")
    plt.title("Training curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_path.with_suffix(".png"), dpi=150)
    comparison_path = model_path.with_name(model_path.stem + "_comparison.png")
    plot_prediction_comparison(restored, X_test_mm, comparison_path)
    plt.show()
