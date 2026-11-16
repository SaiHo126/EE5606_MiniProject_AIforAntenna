"""Handwritten ReLU network with NumPy CPU / CuPy GPU execution.

Raw mm inputs and GHz targets are never standardized. Backpropagation and
parameter updates remain handwritten. Only array operations move to the GPU.
The default run fits 768 ReLU neurons with handwritten Levenberg-Marquardt
(LM), then inserts additional learned thresholds and refits the linear output.
The final hidden size is printed. Both stages use float64 and raw mm/GHz.
LM uses 10,420 training samples; the output fit uses 20,420. Original validation
and test designs stay held out. No analytical formula is called by predict().
Use --optimizer sgd for the old gradient-descent fitter.

From the project root:
    python src/NeuralNetworkReLU.py --device gpu --show
    python src/NeuralNetworkReLU.py --optimizer sgd --hidden-size 16 --extra-samples 0
Add --show to open the saved training curve after training finishes.
The test-set frequencies are also compared with PatchFormula.RMA and saved.
The LM target is validation RMSE in kHz; MSE is in GHz^2. A 10 kHz RMSE
corresponds to MSE=1e-10 GHz^2. epochs is the maximum number of update steps.
For SGD, optional stopping is --tolerance-percent 0.1 --patience 10.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from time import perf_counter
class NeuralNetworkReLU:
    inputSize = 3
    outputSize = 1
    activationHidden = "relu"
    def __init__(self, hiddenSize=768, seed=1, device="cpu", dtype=np.float64):
        self.hiddenSize = hiddenSize
        self.seed = seed
        self.device = device
        self.dtype = np.dtype(dtype)
        if device == "gpu":
            import cupy as cp
            self.xp = cp
        elif device == "cpu":
            self.xp = np
        else:
            raise ValueError("device must be 'cpu' or 'gpu'")
        # Initialize and shuffle with the same CPU RNG for comparable CPU/GPU
        # runs. Weights and each epoch's permutation are transferred just once.
        self.rng = np.random.default_rng(seed)
        # Keep the original small hidden-weight range for raw mm inputs.
        self.limit_hidden = 0.01
        self.W_hidden = self.xp.asarray(self.rng.uniform(
            -self.limit_hidden, self.limit_hidden, size=(self.hiddenSize, self.inputSize)
        ), dtype=self.dtype)
        self.B_hidden = self.xp.zeros((self.hiddenSize, 1), dtype=self.dtype)
        self.limit_output = np.sqrt(6/(self.hiddenSize+self.outputSize))
        self.W_output = self.xp.asarray(self.rng.uniform(
            -self.limit_output, self.limit_output, size=(self.outputSize, self.hiddenSize)
        ), dtype=self.dtype)
        self.B_output = self.xp.zeros((self.outputSize, 1), dtype=self.dtype)
        
        self.history = {"training_loss": [], "validation_loss": []}
        self.training_settings={}
        self.best_epoch=0
        self.training_device = device
        self.transfer_seconds = 0.0
        self.fit_seconds = 0.0
        self.stop_reason = "not_trained"

    def to_numpy(self, value):
        """Copy GPU arrays to CPU for saving, reporting, or plotting."""
        return self.xp.asnumpy(value) if self.device == "gpu" else np.asarray(value)

    def synchronize(self):
        """Wait for pending GPU operations before measuring wall-clock time."""
        if self.device == "gpu":
            self.xp.cuda.get_current_stream().synchronize()

    def activationFunction(self,Z):
        # ReLU keeps positive values and sets negative values to zero.
        return self.xp.maximum(0.0, Z)
    def forward(self, X):
        # forward propagation
        # Already-resident arrays stay on the device without another copy.
        X = self.xp.asarray(X, dtype=self.dtype)
        # H = hiddenSize; N = number of samples.
        Z_hidden = self.W_hidden @ X.T + self.B_hidden # (H, 3) @ (3, N) -> (H, N)
        A_hidden = self.activationFunction(Z_hidden) # (H, N)
        # Keep the output linear: only the hidden layer uses ReLU.
        Z_output = self.W_output @ A_hidden + self.B_output # (1, H) @ (H, N) -> (1, N)
        cache = {"X": X.T, "Z_hidden": Z_hidden, "A_hidden": A_hidden}
        return Z_output.T, cache # (N, 1), in GHz
    def loss(self, Y_pred, Y_actually):
        # Public MSE in GHz^2. Reading the scalar synchronizes GPU work.
        Y_pred = self.xp.asarray(Y_pred, dtype=self.dtype)
        Y_actually = self.xp.asarray(Y_actually, dtype=self.dtype)
        return float(self.xp.mean((Y_pred - Y_actually)**2).item())
    def backward(self, Y_pred, Y_actual, cache :dict):
        # backward propagation using gradient
        X = cache["X"]
        Z_hidden = cache["Z_hidden"]
        A_hidden = cache["A_hidden"]
        N = X.shape[1]
        Y_pred = self.xp.asarray(Y_pred, dtype=self.dtype)
        Y_actual = self.xp.asarray(Y_actual, dtype=self.dtype)
        E_output = (Y_pred- Y_actual) * (2.0/N) # (N, 1)
        dW_output = E_output.T @ A_hidden.T # (1, N) @ (N, H) -> (1, H)
        dB_output = self.xp.sum(E_output, axis=0, keepdims=True)
        # ReLU derivative is 1 for positive Z_hidden and 0 for negative values.
        # At Z_hidden == 0, choose derivative 0 (ReLU has a corner there).
        E_hidden = (self.W_output.T @ E_output.T) * (Z_hidden > 0) # (H, N)
        dW_hidden = E_hidden @ X.T # (H, N) @ (N, 3) -> (H, 3)
        dB_hidden = self.xp.sum(E_hidden, axis=1, keepdims=True) #(H,1)
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

    def parameter_vector(self):
        """One vector containing W_hidden, B_hidden, W_output, B_output."""
        return self.xp.concatenate([p.ravel() for p in
            (self.W_hidden, self.B_hidden, self.W_output, self.B_output)])

    def set_parameter_vector(self, vector):
        start = 0
        for parameter in (self.W_hidden, self.B_hidden, self.W_output, self.B_output):
            parameter[:] = vector[start:start + parameter.size].reshape(parameter.shape)
            start += parameter.size

    def prediction_jacobian(self, cache):
        """J[n,p] = derivative of prediction n with respect to parameter p.

        This uses the same ReLU derivative as backward(). Unlike the MSE
        gradient, it keeps a separate row for every training sample.
        """
        X = cache["X"].T                              # (N, 3), raw mm
        A = cache["A_hidden"].T                       # (N, H)
        D = (cache["Z_hidden"].T > 0) * self.W_output  # (N, H)
        dW = (D[:, :, None] * X[:, None, :]).reshape(len(X), -1)
        return self.xp.concatenate((dW, D, A, self.xp.ones((len(X), 1), dtype=self.dtype)), axis=1)

    def fit_lm(self, dataTrain, dataValidation, epochs=500,
               target_rmse_kHz=5.0, initial_damping=0.01,
               uniform_knots=True, progress_every=25):
        """Full-batch Levenberg-Marquardt, written using array operations.

        Raw X and Y are never standardized. Initial ReLU thresholds are in mm;
        every weight and bias remains trainable. The diagonal rescaling below
        conditions the PARAMETER step, not the inputs or output frequencies.
        Validation selects the checkpoint; only training data determines steps.
        """
        self.synchronize()
        transfer_start = perf_counter()
        X_cpu = np.asarray(dataTrain["X_mm"], dtype=float)
        Y_cpu = np.asarray(dataTrain["Y_GHz"], dtype=float)
        X = self.xp.asarray(X_cpu, dtype=self.dtype)
        Y = self.xp.asarray(Y_cpu, dtype=self.dtype)
        XV = self.xp.asarray(dataValidation["X_mm"], dtype=self.dtype)
        YV = self.xp.asarray(dataValidation["Y_GHz"], dtype=self.dtype)
        self.synchronize()
        self.transfer_seconds = perf_counter() - transfer_start
        fit_start = perf_counter()

        # Start with length-, width- and thickness-sensitive neurons. This
        # avoids many neurons having the same active/inactive state everywhere.
        # Allocate half to L, one quarter to W, and one quarter to h.
        self.rng = np.random.default_rng(self.seed)
        coordinates = np.tile([1, 1, 0, 2], (self.hiddenSize + 3) // 4)[:self.hiddenSize]
        W = np.zeros((self.hiddenSize, 3))
        W[np.arange(self.hiddenSize), coordinates] = 0.01
        B = np.empty(self.hiddenSize)
        if uniform_knots:
            for coordinate in range(3):
                rows = np.flatnonzero(coordinates == coordinate)
                thresholds = np.linspace(X_cpu[:, coordinate].min(), X_cpu[:, coordinate].max(), len(rows) + 2)[1:-1]
                B[rows] = -0.01 * thresholds
        else:
            centres = X_cpu[self.rng.integers(len(X_cpu), size=self.hiddenSize)]
            B[:] = -np.sum(W * centres, axis=1)
        self.W_hidden[:] = self.xp.asarray(W)
        self.B_hidden[:] = self.xp.asarray(B[:, None])
        self.W_output[:] = 0.0
        self.B_output[:] = float(Y_cpu.mean())

        self.history = {"training_loss": [], "validation_loss": []}
        self.training_settings = {
            "optimizer": "levenberg_marquardt", "epochs": epochs,
            "lm_device": self.device,
            "learning_rate": None, "batch_size": len(X),
            "tolerance_percent": None, "patience": 1,
            "target_rmse_kHz": target_rmse_kHz,
            "initial_damping": initial_damping, "uniform_knots": uniform_knots,
        }
        self.training_device = self.device
        self.stop_reason = "epoch_limit"
        damping = initial_damping
        best_validation_loss = np.inf
        best_parameters = self.parameter_vector().copy()
        self.best_epoch = 0
        target_mse = (target_rmse_kHz / 1e6)**2  # kHz -> GHz -> GHz^2
        for epoch in range(epochs):
            prediction, cache = self.forward(X)
            residual = (prediction - Y).ravel()
            loss = float(self.xp.mean(residual**2).item())
            J = self.prediction_jacobian(cache)
            normal = J.T @ J / len(X)
            gradient = J.T @ residual / len(X)
            # Solve a damped normal equation; never form a matrix inverse.
            scale = self.xp.sqrt(self.xp.maximum(self.xp.diag(normal), 1e-14))
            scaled_normal = normal / scale[:, None] / scale[None, :]
            parameters = self.parameter_vector()
            accepted = False
            for attempt in range(12):
                system = scaled_normal.copy()
                diagonal = self.xp.arange(len(system))
                system[diagonal, diagonal] += damping
                step = self.xp.linalg.solve(system, -gradient / scale) / scale
                self.set_parameter_vector(parameters + step)
                new_loss = self.loss(self.forward(X)[0], Y)
                if np.isfinite(new_loss) and new_loss < loss:
                    damping = max(damping / 3.0, 1e-12)
                    accepted = True
                    break
                self.set_parameter_vector(parameters)
                new_loss = loss
                damping *= 10.0
            validation_loss = self.loss(self.forward(XV)[0], YV)
            self.history["training_loss"].append(new_loss)
            self.history["validation_loss"].append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_parameters = self.parameter_vector().copy()
                self.best_epoch = epoch + 1
            if progress_every and (epoch == 0 or (epoch + 1) % progress_every == 0):
                print(f"LM step {epoch + 1}: training RMSE={np.sqrt(new_loss)*1e6:.3f} kHz; validation RMSE={np.sqrt(validation_loss)*1e6:.3f} kHz", flush=True)
            if best_validation_loss < target_mse:
                self.stop_reason = "validation_target"
                break
            if not accepted:
                self.stop_reason = "no_improving_step"
                break
        self.set_parameter_vector(best_parameters)
        self.synchronize()
        self.fit_seconds = perf_counter() - fit_start
        return self.history

    def refine_hidden_layer(self, factor=4):
        """Insert ReLU thresholds between learned neighbouring thresholds.

        Neurons retain their learned directions in raw [W,L,h] space. New
        neurons start with zero output weights, so the initial prediction is
        preserved. fit_output() then learns their contributions from labels.
        """
        W = self.to_numpy(self.W_hidden)
        B = self.to_numpy(self.B_hidden).ravel()
        weights, biases = [W], [B]
        coordinates = np.tile([1, 1, 0, 2], (self.hiddenSize + 3) // 4)[:self.hiddenSize]
        for coordinate in range(3):
            for sign in (-1, 1):
                rows = np.flatnonzero((coordinates == coordinate) & (sign * W[:, coordinate] > 1e-10))
                # ReLU(c*z) = c*ReLU(z) for c>0. Express each plane with a
                # common coefficient before interpolating its weights/bias.
                plane_W = W[rows] / np.abs(W[rows, coordinate, None])
                plane_B = B[rows] / np.abs(W[rows, coordinate])
                order = np.argsort(-plane_B / sign)
                plane_W, plane_B = plane_W[order], plane_B[order]
                for k in range(1, factor):
                    t = k / factor
                    weights.append((1-t) * plane_W[:-1] + t * plane_W[1:])
                    biases.append((1-t) * plane_B[:-1] + t * plane_B[1:])
        W_new = np.concatenate(weights)
        B_new = np.concatenate(biases)[:, None]
        model = type(self)(hiddenSize=len(W_new), seed=self.seed, device=self.device, dtype=self.dtype)
        model.W_hidden[:] = self.xp.asarray(W_new)
        model.B_hidden[:] = self.xp.asarray(B_new)
        model.W_output[:] = 0
        model.W_output[:, :self.hiddenSize] = self.W_output
        model.B_output[:] = self.B_output
        model.history = {key: list(values) for key, values in self.history.items()}
        model.training_settings = dict(self.training_settings, base_hidden_size=self.hiddenSize,
                                       refinement_factor=factor,
                                       lm_device=self.training_settings.get("lm_device", self.training_device))
        model.best_epoch = self.best_epoch
        model.fit_seconds = self.fit_seconds
        model.transfer_seconds = self.transfer_seconds
        return model

    def fit_output(self, dataTrain, dataValidation, rcond=1e-10):
        """Fit the linear output by solving its least-squares equations.

        Hidden ReLU weights stay fixed for this polishing stage. A contains
        hidden activations and one bias column; minimize ||A*c - Y||^2 using
        NumPy/CuPy's linear-algebra solver, not an ML training library.
        Column rescaling conditions the coefficient solve only: forward()
        continues using raw mm inputs, ReLU, and raw GHz predictions.
        """
        self.synchronize()
        start = perf_counter()
        previous_output = self.W_output.copy(), self.B_output.copy()
        previous_validation = self.loss(self.forward(dataValidation["X_mm"])[0], dataValidation["Y_GHz"])
        X = self.xp.asarray(dataTrain["X_mm"], dtype=self.dtype)
        Y = self.xp.asarray(dataTrain["Y_GHz"], dtype=self.dtype)
        hidden = self.forward(X)[1]["A_hidden"].T
        A = self.xp.concatenate([hidden, self.xp.ones((len(X), 1), dtype=self.dtype)], axis=1)
        column_scale = self.xp.maximum(self.xp.sqrt(self.xp.sum(A**2, axis=0)), 1e-14)
        coefficient, _, rank, _ = self.xp.linalg.lstsq(A / column_scale, Y, rcond=rcond)
        coefficient /= column_scale[:, None]
        self.W_output[:] = coefficient[:-1].T
        self.B_output[:] = coefficient[-1:]
        training_loss = self.loss(self.forward(X)[0], Y)
        validation_loss = self.loss(self.forward(dataValidation["X_mm"])[0], dataValidation["Y_GHz"])
        self.history["training_loss"].append(training_loss)
        self.history["validation_loss"].append(validation_loss)
        if validation_loss < previous_validation:
            self.best_epoch = len(self.history["training_loss"])
            self.stop_reason = "linear_output_fit"
        else:
            self.W_output[:], self.B_output[:] = previous_output
            self.stop_reason = "output_validation_not_improved"
        self.training_settings.update(optimizer="lm_then_output_least_squares",
                                      output_fit_samples=len(X), output_rcond=rcond,
                                      output_rank=int(rank), output_fit_device=self.device)
        self.training_device = f"{self.training_settings.get('lm_device', self.device)} LM + {self.device} output fit"
        self.synchronize()
        self.fit_seconds += perf_counter() - start
        return self.history
    def fit(
        self,
        dataTrain,
        dataValidation,
        epochs = 1000,
        learning_rate = 0.0001,
        batch_size=32,
        tolerance_percent=None,
        patience=1,
        progress_every=0,
    ):
        """Train on raw values, keeping batches, gradients and weights on-device.

        epochs is a maximum. Optional stopping compares the absolute percentage
        change of consecutive validation losses for patience consecutive epochs.
        History stores one CPU scalar per split and epoch, in GHz^2.
        """
        self.synchronize()
        transfer_start = perf_counter()
        X_train = self.xp.asarray(dataTrain["X_mm"], dtype=self.dtype)
        Y_train = self.xp.asarray(dataTrain["Y_GHz"], dtype=self.dtype)
        X_validation = self.xp.asarray(dataValidation["X_mm"], dtype=self.dtype)
        Y_validation = self.xp.asarray(dataValidation["Y_GHz"], dtype=self.dtype)
        self.synchronize()
        self.transfer_seconds = perf_counter() - transfer_start
        fit_start = perf_counter()
        
        self.history = {"training_loss" :[], "validation_loss": []}
        self.best_epoch =0
        self.training_settings = {
            "epochs":epochs, 
            "learning_rate":learning_rate, 
            "batch_size":batch_size,
            "tolerance_percent":tolerance_percent,
            "patience":patience,
        }
        self.training_device = self.device
        self.stop_reason = "epoch_limit"
        best_validation_loss = np.inf
        best_parameters=(
            self.W_hidden.copy(),self.B_hidden.copy(),
            self.W_output.copy(),self.B_output.copy(),
        )
        previous_validation_loss = None
        stable_epochs = 0
        # one epoch visits every training sample once
        for epoch in range(epochs):
            # randomlize the data set order
            order = self.xp.asarray(self.rng.permutation(len(X_train)))
            for start in range(0, len(X_train), batch_size):
                indices = order[start:start+batch_size]
                X_batch = X_train[indices]
                Y_batch = Y_train[indices]
                Y_pred , cache = self.forward(X_batch)
                gradients = self.backward(Y_pred= Y_pred, Y_actual= Y_batch, cache=cache)
                self.update_parameters(gradients=gradients, learning_rate=learning_rate)
            
            # Validates the model
            training_pred, _ = self.forward(X_train)
            validation_pred, _ = self.forward(X_validation)
            # Transfer just two scalar losses per epoch, rather than predictions
            # or gradients after every batch. Both MSEs are evaluated on-device.
            epoch_losses = self.xp.stack((
                self.xp.mean((training_pred - Y_train)**2),
                self.xp.mean((validation_pred - Y_validation)**2),
            ))
            training_loss, validation_loss = map(float, self.to_numpy(epoch_losses))
            self.history["training_loss"].append(training_loss)
            self.history["validation_loss"].append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self.best_epoch = epoch + 1
                best_parameters = (
                    self.W_hidden.copy(), self.B_hidden.copy(),
                    self.W_output.copy(), self.B_output.copy(),
                )
            if progress_every and (epoch == 0 or (epoch + 1) % progress_every == 0 or epoch + 1 == epochs):
                print(f"Epoch {epoch + 1}/{epochs}: train={training_loss:.6g}, validation={validation_loss:.6g} GHz^2", flush=True)
            if validation_loss == 0.0:
                self.stop_reason = "zero_validation_loss"
                break
            if tolerance_percent is not None and previous_validation_loss is not None:
                change_percent = abs(validation_loss - previous_validation_loss) / previous_validation_loss * 100.0
                stable_epochs = stable_epochs + 1 if change_percent < tolerance_percent else 0
                if stable_epochs >= patience:
                    self.stop_reason = "validation_change"
                    break
            previous_validation_loss = validation_loss
        self.W_hidden, self.B_hidden, self.W_output, self.B_output = best_parameters            
        self.synchronize()
        self.fit_seconds = perf_counter() - fit_start
        return self.history
    def predict(self, design_mm):
        X_mm = np.array(design_mm, dtype=float).reshape(1,3)  
        Y_GHz , _ = self.forward(X_mm)
        return float(Y_GHz[0,0].item())
    def save(self,filename):
        np.savez(
            filename,
            W_hidden=self.to_numpy(self.W_hidden), B_hidden=self.to_numpy(self.B_hidden),
            W_output=self.to_numpy(self.W_output), B_output=self.to_numpy(self.B_output),
            data_representation="mm_GHz",
            activation_hidden=self.activationHidden,
            hidden_size=self.hiddenSize, 
            seed=self.seed,
            best_epoch=self.best_epoch,
            epochs=self.training_settings["epochs"],
            learning_rate=(np.nan if self.training_settings["learning_rate"] is None
                           else self.training_settings["learning_rate"]),
            batch_size=self.training_settings["batch_size"],
            tolerance_percent=(np.nan if self.training_settings["tolerance_percent"] is None
                               else self.training_settings["tolerance_percent"]),
            patience=self.training_settings["patience"],
            training_device=self.training_device,
            dtype=self.dtype.name,
            epochs_run=len(self.history["training_loss"]),
            transfer_seconds=self.transfer_seconds,
            fit_seconds=self.fit_seconds,
            stop_reason=self.stop_reason,
            training_loss=self.history["training_loss"],
            validation_loss=self.history["validation_loss"],
            training_settings_json=json.dumps(self.training_settings),
        )
    @classmethod
    def load(cls, filename, device="cpu"):
        """Load onto CPU or GPU, regardless of which device trained the model."""
        with np.load(filename) as data:
            if("data_representation" not in data
               or data ["data_representation"].item() != "mm_GHz"):
                raise ValueError("Load a ReLU model trained using raw mm/GHz data.")
            # The same weights give different predictions with tanh and ReLU.
            if ("activation_hidden" not in data
                    or data["activation_hidden"].item() != cls.activationHidden):
                raise ValueError("Load a model saved by NeuralNetworkReLU.save().")
            model = cls(
                    hiddenSize=int(data["hidden_size"]),
                    seed=int(data["seed"]), device=device,
                    dtype=data["W_hidden"].dtype)
            model.W_hidden = model.xp.array(data["W_hidden"], copy=True)
            model.B_hidden = model.xp.array(data["B_hidden"], copy=True)
            model.W_output = model.xp.array(data["W_output"], copy=True)
            model.B_output = model.xp.array(data["B_output"], copy=True)
            model.best_epoch = int(data["best_epoch"])
            tolerance = float(data.get("tolerance_percent", np.nan))
            model.training_settings = {
                            "epochs": int(data["epochs"]),
                            "learning_rate": float(data["learning_rate"]),
                            "batch_size": int(data["batch_size"]),
                            "tolerance_percent": None if np.isnan(tolerance) else tolerance,
                            "patience": int(data.get("patience", 1)),
                        }
            if "training_settings_json" in data:
                model.training_settings = json.loads(str(data["training_settings_json"]))
            model.history = {
                            "training_loss": data["training_loss"].tolist(),
                            "validation_loss": data["validation_loss"].tolist(),
                        }
            model.training_device = str(data.get("training_device", "cpu"))
            model.transfer_seconds = float(data.get("transfer_seconds", 0.0))
            model.fit_seconds = float(data.get("fit_seconds", 0.0))
            model.stop_reason = str(data.get("stop_reason", "saved_model"))
        return model

    def plot_history(self, filename, show=False):
        """Save training/validation MSE curves; optionally open a plot window."""
        import matplotlib.pyplot as plt
        epochs = np.arange(1, len(self.history["training_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, self.history["training_loss"], label="Training", color="#2563eb", linewidth=1.5)
        ax.plot(epochs, self.history["validation_loss"], label="Validation", color="#ea580c", linewidth=1.5)
        ax.axvline(self.best_epoch, color="#64748b", linestyle="--", linewidth=1,
                   label=f"Best validation epoch: {self.best_epoch}")
        if self.training_settings.get("optimizer", "").startswith(("levenberg", "lm_then")):
            ax.set_yscale("log")
            ax.axhline(1e-10, color="#15803d", linestyle=":", label="10 kHz RMSE target")
        if "refinement_factor" in self.training_settings:
            ax.axvline(len(epochs), color="#a855f7", linestyle=":", label="Output refit on expanded training data")
        ax.set(title=f"ReLU network: 3 - {self.hiddenSize} - 1 | {self.training_device.upper()}",
               xlabel="Epoch", ylabel=r"Mean squared error (GHz$^2$)")
        if "refinement_factor" in self.training_settings:
            ax.set_xlabel("Training step (last point: output refit)")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
            
if __name__ == "__main__":
    # Support running this file directly and with python -m src.NeuralNetworkReLU.
    if __package__:
        from .PatchFormula import RMA
    else:
        from PatchFormula import RMA
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--optimizer", choices=["lm", "sgd"], default="lm")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--extra-samples", type=int, default=20000)
    parser.add_argument("--lm-samples", type=int, default=10420)
    parser.add_argument("--refinement-factor", type=int, default=8)
    parser.add_argument("--output-fit-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--base-model", type=Path, help="Reuse an LM checkpoint and skip the LM training stage")
    parser.add_argument("--sample-seed", type=int, default=101)
    parser.add_argument("--target-rmse-khz", type=float, default=5.0,
                        help="LM validation stopping target; 5 kHz gives margin below 10 kHz")
    parser.add_argument("--tolerance-percent", type=float, default=None)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--show", action="store_true", help="Open the loss plot, then the frequency comparison")
    parser.add_argument("--output-dir", type=Path, default=project_root / "results" / "nn")
    args = parser.parse_args()
    dataset_path = project_root/"antennaSample_training.npz"
    with np.load(dataset_path) as data:
        original_X = data["X_mm"].copy()
        original_Y = data["Y_GHz"].copy()
        train_indices = data["train_indices"]
        validation_indices = data["validation_indices"]
        test_indices = data["test_indices"]
        X_test_mm = data["X_mm"][test_indices]
        
        dataTrain = {
            "X_mm": data["X_mm"][train_indices],
            "Y_GHz": data["Y_GHz"][train_indices]
        }
        dataValidation = {
            "X_mm": data["X_mm"][validation_indices],
            "Y_GHz": data["Y_GHz"][validation_indices]
        }
    # Generate only additional TRAINING designs. Original validation and test
    # indices are preserved, and this expanded split is saved for reuse by GP.
    generation_start = perf_counter()
    sampling_rng = np.random.default_rng(args.sample_seed)
    extra_batches = []
    remaining = args.extra_samples
    while remaining > 0:
        candidates = sampling_rng.uniform([25, 25, 1], [50, 45, 5], (remaining * 5, 3))
        feasible = candidates[candidates[:, 0] >= 1.2 * candidates[:, 1]][:remaining]
        extra_batches.append(feasible)
        remaining -= len(feasible)
    if args.extra_samples:
        extra_X = np.concatenate(extra_batches)
        extra_Y = np.array([RMA(*design).ResonanceF_GHz for design in extra_X])[:, None]
        dataTrain["X_mm"] = np.concatenate([dataTrain["X_mm"], extra_X])
        dataTrain["Y_GHz"] = np.concatenate([dataTrain["Y_GHz"], extra_Y])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = args.output_dir / f"antenna_training_plus{args.extra_samples}_seed{args.sample_seed}.npz"
        np.savez(dataset_path, X_mm=np.concatenate([original_X, extra_X]),
                 Y_GHz=np.concatenate([original_Y, extra_Y]),
                 train_indices=np.concatenate([train_indices, np.arange(len(original_X), len(original_X) + len(extra_X))]),
                 validation_indices=validation_indices, test_indices=test_indices,
                 sample_seed=args.sample_seed, original_dataset="antennaSample_training.npz")
    generation_seconds = perf_counter() - generation_start
    print(f"Additional data generation/storage: {generation_seconds:.3f} s")
    if args.base_model:
        model = NeuralNetworkReLU.load(args.base_model, device=args.device)
    else:
        model = NeuralNetworkReLU(hiddenSize=args.hidden_size, seed=args.seed,
                                  device=args.device, dtype=args.dtype)
    print(f"Device: {args.device.upper()}, arithmetic: {args.dtype}")
    if args.device == "gpu":
        properties = model.xp.cuda.runtime.getDeviceProperties(model.xp.cuda.Device().id)
        print("GPU:", properties["name"].decode())
    print("Training shapes:", dataTrain["X_mm"].shape, dataTrain["Y_GHz"].shape)
    print("Validation shapes:", dataValidation["X_mm"].shape, dataValidation["Y_GHz"].shape)
    print("Weight shapes:", model.W_hidden.shape, model.W_output.shape)
    print("Bias shapes:", model.B_hidden.shape, model.B_output.shape)
    print("Raw units: X in mm, Y in GHz; no standardization")
    print("Internal inputs: X = X_mm.T, then W_hidden @ X + B_hidden")
    print(f"Architecture: 3 -> {model.hiddenSize} ReLU -> 1 linear output")
    
    if args.optimizer == "lm":
        if not args.base_model:
            lm_data = {key: values[:args.lm_samples] for key, values in dataTrain.items()}
            model.fit_lm(lm_data, dataValidation, epochs=args.epochs,
                         target_rmse_kHz=args.target_rmse_khz, progress_every=10)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        base_path = args.output_dir / f"nn_model_relu_lm_base_h{model.hiddenSize}_{args.device}.npz"
        model.save(base_path)
        if args.refinement_factor > 1:
            # CPU SVD avoids the large GPU memory workspace of the tall output
            # system. GPU remains available for LM and final predictions.
            model = NeuralNetworkReLU.load(base_path, device=args.output_fit_device)
            model = model.refine_hidden_layer(args.refinement_factor)
            print(f"Refined architecture: 3 -> {model.hiddenSize} ReLU -> 1; output fit on {args.output_fit_device.upper()}", flush=True)
            model.fit_output(dataTrain, dataValidation)
        history = model.history
    else:
        history = model.fit(
            dataTrain=dataTrain, dataValidation=dataValidation,
            learning_rate=args.learning_rate, epochs=args.epochs,
            batch_size=args.batch_size, tolerance_percent=args.tolerance_percent,
            patience=args.patience, progress_every=max(1, args.epochs // 10),
        )
    print(f"Data transfer/conversion time: {model.transfer_seconds:.6f} s")
    print(f"Training time (including any first-use kernel setup): {model.fit_seconds:.3f} s")
    print(f"Epochs completed: {len(history['training_loss'])}; stop reason: {model.stop_reason}")
    print(f"Best validation epoch: {model.best_epoch}")
    print(f"Best validation MSE (GHz^2): {min(history['validation_loss']):.6g}")
    validation_pred, _ = model.forward(dataValidation["X_mm"])
    validation_rmse_MHz = 1000 * np.sqrt(model.loss(validation_pred, dataValidation["Y_GHz"]))
    print(f"Validation RMSE: {validation_rmse_MHz:.3f} MHz")
    # Include the hidden size so an experiment preserves earlier model files.
    run_name = f"relu_{args.optimizer}_h{model.hiddenSize}_{args.device}_n{len(dataTrain['X_mm'])}"
    model_path = args.output_dir / f"nn_model_{run_name}.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved ReLU model: {model_path}")
    model = NeuralNetworkReLU.load(model_path, device=args.device)
    curve_path = args.output_dir / f"{run_name}_training.png"
    model.plot_history(curve_path, show=args.show)
    print(f"Saved training curve: {curve_path}")

    # Test data is evaluated only after training and best-validation selection.
    # Recompute the reference from PatchFormula.py, rather than trusting labels.
    analytical_start = perf_counter()
    analytical_GHz = np.array([RMA(*design).ResonanceF_GHz for design in X_test_mm])
    analytical_seconds = perf_counter() - analytical_start
    model.synchronize()
    prediction_start = perf_counter()
    predicted_GHz = model.to_numpy(model.forward(X_test_mm)[0]).ravel()
    prediction_seconds = perf_counter() - prediction_start
    error_MHz = (predicted_GHz - analytical_GHz) * 1000.0
    absolute_percentage_error = 100.0 * np.abs(predicted_GHz - analytical_GHz) / analytical_GHz
    rmse_MHz = float(np.sqrt(np.mean(error_MHz**2)))
    mae_MHz = float(np.mean(np.abs(error_MHz)))
    mape_percent = float(np.mean(absolute_percentage_error))
    max_absolute_error_MHz = float(np.max(np.abs(error_MHz)))
    print(f"Test comparison with PatchFormula.RMA: {len(X_test_mm)} designs")
    print(f"Test RMSE: {rmse_MHz:.3f} MHz; MAE: {mae_MHz:.3f} MHz")
    print(f"Mean absolute percentage error: {mape_percent:.3f}%")
    print(f"Maximum absolute error: {max_absolute_error_MHz:.3f} MHz")
    mse_GHz2 = float(np.mean((predicted_GHz - analytical_GHz)**2))
    print(f"Test MSE: {mse_GHz2:.9g} GHz^2; RMSE: {rmse_MHz * 1000:.3f} kHz")
    print(f"Test RMSE below 10 kHz: {rmse_MHz * 1000 < 10.0}")
    print(f"Analytical evaluation: {analytical_seconds:.6f} s; NN prediction including transfers: {prediction_seconds:.6f} s")
    comparison_path = args.output_dir / f"{run_name}_comparison.npz"
    np.savez(
        comparison_path, test_indices=test_indices, X_mm=X_test_mm,
        analytical_GHz=analytical_GHz, predicted_GHz=predicted_GHz,
        error_MHz=error_MHz, absolute_percentage_error=absolute_percentage_error,
        rmse_MHz=rmse_MHz, mae_MHz=mae_MHz, mape_percent=mape_percent,
        max_absolute_error_MHz=max_absolute_error_MHz,
        mse_GHz2=mse_GHz2, rmse_kHz=rmse_MHz * 1000,
        target_rmse_kHz=10.0, target_met=rmse_MHz * 1000 < 10.0,
        analytical_seconds=analytical_seconds, prediction_seconds=prediction_seconds,
        model_filename=model_path.name, dataset_filename=dataset_path.name,
    )
    print(f"Saved test comparison: {comparison_path}")
    summary = {
        "architecture": [3, model.hiddenSize, 1], "activation": "relu", "raw_mm_GHz": True,
        "model_file": model_path.name, "dataset_file": dataset_path.name,
        "training_samples": len(dataTrain["X_mm"]), "validation_samples": len(validation_indices),
        "test_samples": len(test_indices), "seed": args.seed, "sample_seed": args.sample_seed,
        "settings": model.training_settings, "best_epoch": model.best_epoch,
        "training_seconds": model.fit_seconds, "generation_seconds": generation_seconds,
        "validation_rmse_kHz": validation_rmse_MHz * 1000,
        "test_mse_GHz2": mse_GHz2, "test_rmse_kHz": rmse_MHz * 1000,
        "test_mae_kHz": mae_MHz * 1000, "test_max_absolute_error_kHz": max_absolute_error_MHz * 1000,
        "test_rmse_below_10_kHz": rmse_MHz * 1000 < 10.0,
    }
    (args.output_dir / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Show frequency agreement and the small remaining errors in kHz.
    order = np.argsort(analytical_GHz)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(np.arange(1, len(order) + 1), analytical_GHz[order], label="Analytical formula", color="#2563eb")
    axes[0].scatter(np.arange(1, len(order) + 1), predicted_GHz[order], label="ReLU prediction", color="#ea580c", s=18)
    axes[0].set(xlabel="Test designs (sorted by analytical frequency)", ylabel="Resonance frequency (GHz)")
    axes[1].scatter(analytical_GHz, error_MHz * 1000, color="#ea580c", s=18, label="Test prediction error")
    axes[1].axhline(0, linestyle="--", color="#64748b", label="Exact agreement")
    axes[1].axhline(10, linestyle=":", color="#15803d", label="+/-10 kHz reference")
    axes[1].axhline(-10, linestyle=":", color="#15803d")
    axes[1].set(xlabel="Analytical frequency (GHz)", ylabel="ReLU minus analytical (kHz)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(f"3 - {model.hiddenSize} - 1 ReLU | {len(X_test_mm)} held-out designs | RMSE {rmse_MHz * 1000:.2f} kHz")
    fig.tight_layout()
    comparison_plot_path = args.output_dir / f"{run_name}_comparison.png"
    fig.savefig(comparison_plot_path, dpi=180)
    print(f"Saved frequency plot: {comparison_plot_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    design = [38.0, 30.0, 1.6]
    restored = NeuralNetworkReLU.load(model_path)
    reference_GHz = RMA(*design).ResonanceF_GHz
    prediction_GHz = restored.predict(design)
    print(f"Reference design [W,L,h]={design} mm")
    print(f"Analytical: {reference_GHz:.9f} GHz; reloaded ReLU: {prediction_GHz:.9f} GHz")
    print(f"Absolute prediction error: {abs(prediction_GHz - reference_GHz) * 1000:.3f} MHz")
