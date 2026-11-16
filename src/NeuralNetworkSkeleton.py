"""Completed NumPy example: 3 inputs -> 8 tanh neurons -> 1 linear output.

Run from the project root: python src/NeuralNetworkSkeleton.py
Running this file trains on the prepared split, reports training/validation
errors and saves results/nn/nn_model_raw.npz. The test split stays unused so it can
be used for final evaluation after choosing the training settings.

Use the same names as NerualNetwork.py: W_hidden, B_hidden, W_output,
B_output, Z_hidden and A_hidden. The lecture calls the two weight matrices
W_input-hidden and W_hidden-output, respectively.
For a batch of N examples, the default shapes are:
    X = X_mm.T                              -> input columns (3, N)
    Z_hidden = W_hidden @ X + B_hidden      -> hidden inputs (8, N)
    A_hidden = tanh(Z_hidden)               -> hidden outputs (8, N)
    Z_output = W_output @ A_hidden + B_output -> output columns (1, N)
    Y_pred = Z_output.T                     -> returned predictions (N, 1)
The network has 3*8 + 8 + 8*1 + 1 = 41 trainable parameters.
Each hidden neuron receives all three dimensions; its weights form one ROW.
Use @ for matrix multiplication; * means elementwise multiplication in NumPy.

Read the implementation in this order:
    1 forward -> 2 loss -> 3 backward -> 4 update_parameters -> 5 fit
    6 save -> 7 load

fit() accepts dataTrain and dataValidation directly. Each is a dictionary (or an open .npz
archive for one split) with X_mm and Y_GHz entries, in raw mm/GHz units.
X and Y are NOT standardized: forward() uses mm and returns GHz, and loss()
returns MSE in GHz^2. Public batches keep one sample per row, matching the
dataset. Only forward's internal arrays and cache use one sample per COLUMN.
predict() accepts a single design in mm and returns GHz for later PSO.
"""

from pathlib import Path
from time import perf_counter

import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_size=8, seed=1):
        self.hidden_size = hidden_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize different weights so neurons can learn differently.
        # Raw dimensions are tens of mm. A small initial hidden-weight range
        # keeps tanh away from saturation at initialization. This is a training
        # setting, not a transformation of X: input values stay in mm.
        # Zero biases are fine: the random weights already distinguish neurons.
        limit_hidden = 0.01
        # One row per hidden neuron: three input weights and one bias each.
        self.W_hidden = self.rng.uniform(
            -limit_hidden, limit_hidden, size=(hidden_size, 3)
        )
        self.B_hidden = np.zeros((hidden_size, 1))

        # Hidden activations are bounded by tanh, so use Xavier for this layer.
        limit_output = np.sqrt(6.0 / (hidden_size + 1))
        # One output neuron receives all hidden outputs.
        self.W_output = self.rng.uniform(
            -limit_output, limit_output, size=(1, hidden_size)
        )
        self.B_output = np.zeros((1, 1))

        self.history = {"training_loss": [], "validation_loss": []}
        self.training_settings = {}
        self.best_epoch = 0  # Use 1-based epoch numbers once training begins.

    def forward(self, X_mm):
        """Return (Y_pred, cache) for raw mm inputs of shape (N, 3).

        Y_pred is in GHz and has shape (N, 1), matching the target rows.
        cache contains X (3, N), Z_hidden and A_hidden (hidden_size, N).
        backward() uses X and A_hidden. Shapes below assume hidden_size=8.
        """
        # Dataset rows become sample columns for W_hidden @ X.
        X = X_mm.T                                         # (3, N)
        # Each neuron has one bias, shared across its sample columns.
        Z_hidden = self.W_hidden @ X + self.B_hidden        # (8, N)
        A_hidden = np.tanh(Z_hidden)                        # (8, N)
        # A linear output predicts frequency in GHz without an inverse transform.
        Z_output = self.W_output @ A_hidden + self.B_output # (1, N)
        Y_pred = Z_output.T                                 # (N, 1)
        # Cache the same inputs and activations used for these predictions.
        cache = {"X": X, "Z_hidden": Z_hidden, "A_hidden": A_hidden}
        return Y_pred, cache

    def loss(self, Y_pred, Y_actual):
        """Return MSE in GHz^2; both frequency arrays have shape (N, 1)."""
        # Squaring makes positive/negative errors both contribute to the loss.
        # Keep BOTH arrays shaped (N, 1): an (N,) target would broadcast to (N, N).
        # sqrt(MSE) is RMSE in GHz; multiply by 1000 to report RMSE in MHz.
        return float(np.mean((Y_pred - Y_actual)**2))

    def backward(self, Y_pred, Y_actual, cache):
        """Return MSE gradients for W_hidden, B_hidden, W_output, B_output.

        Y_pred and Y_actual are frequencies in GHz, both shaped (N, 1).
        Use the cache from the same forward pass, before any parameter update.
        X has shape (3, N); A_hidden has shape (hidden_size, N).

        Lecture Example 3, pp. 52-55: E_output = dLoss/dZ_output and
        E_hidden = dLoss/dZ_hidden. The example uses half-squared error;
        our loss is mean squared error, so its derivative includes 2/N.
        Our hidden activation is tanh, so use 1 - A_hidden**2 instead of
        the example's ReLU derivative. The output activation remains linear.

        This method only calculates gradients. update_parameters() must
        subtract learning_rate * gradient from each matching parameter.
        """
        X = cache["X"]                    # (3, N)
        A_hidden = cache["A_hidden"]      # (8, N) for the default network
        N = X.shape[1]
        # X has shape (3, N): len(X) would count INPUTS, not batch samples.

        # 1. Error at output: dLoss/dZ_output = dLoss/dY_pred * 1.
        # The output is linear, so dY_pred/dZ_output = 1.
        # Transpose row-based errors into columns, matching Z_output.
        E_output = 2.0 * (Y_pred - Y_actual).T / N       # (1, N)

        # 2. Output weight and bias gradients (lecture p. 53).
        # Each weight gradient combines error with its incoming activation.
        dW_output = E_output @ A_hidden.T              # (1, 8)
        # Biases are shared across samples: sum over sample columns.
        dB_output = np.sum(E_output, axis=1, keepdims=True)  # (1, 1)

        # 3. Error at hidden layer (lecture p. 54, adapted to tanh).
        # W_output.T carries error back to each hidden neuron. Multiply by
        # tanh'(Z_hidden) = 1 - tanh(Z_hidden)**2 = 1 - A_hidden**2.
        # @ is matrix multiplication; * is an elementwise product.
        E_hidden = (self.W_output.T @ E_output) * (1.0 - A_hidden**2)  # (8, N)

        # 4. Hidden weight and bias gradients (lecture p. 55).
        dW_hidden = E_hidden @ X.T                     # (8, 3)
        dB_hidden = np.sum(E_hidden, axis=1, keepdims=True)  # (8, 1)

        # The 1/N in E_output already averages ALL four gradients.
        # keepdims=True keeps biases as columns, avoiding (8, 8) broadcasting.
        # Leave the parameters unchanged until ALL gradients are available.
        return {
            "W_hidden": dW_hidden,
            "B_hidden": dB_hidden,
            "W_output": dW_output,
            "B_output": dB_output,
        }

    def update_parameters(self, gradients, learning_rate):
        """Apply one gradient-descent update to both weights and both biases."""
        # The minus sign moves parameters towards decreasing loss.
        # Do not divide by batch size again; backward() already included 1/N.
        self.W_hidden -= learning_rate * gradients["W_hidden"]
        self.B_hidden -= learning_rate * gradients["B_hidden"]
        self.W_output -= learning_rate * gradients["W_output"]
        self.B_output -= learning_rate * gradients["B_output"]

    def fit(
        self, dataTrain, dataValidation,
        epochs=1000, learning_rate=0.0001, batch_size=32,
    ):
        """Train from two loaded datasets; return the loss history.

        Each dataset contains X_mm (N, 3) and Y_GHz (N, 1), with only its
        own split's samples. Pass dictionaries or open np.load(...) results
        from separate split archives. This function does not open file paths.

        The existing antennaSample_training.npz holds ALL samples plus split
        indices. Select its training/validation rows first, as in __main__.
        Both inputs and targets stay in their original units throughout training.
        Use positive integer epochs and batch_size, and nonempty data splits.
        These settings are starting choices to tune using validation data.
        The model finishes with the parameters from the best validation epoch.
        """
        # Read raw values directly: X in mm and Y in GHz, without standardization.
        X_train = dataTrain["X_mm"]
        Y_train = dataTrain["Y_GHz"]
        X_validation = dataValidation["X_mm"]
        Y_validation = dataValidation["Y_GHz"]

        self.history = {"training_loss": [], "validation_loss": []}
        self.best_epoch = 0
        self.training_settings = {
            "epochs": epochs, "learning_rate": learning_rate,
            "batch_size": batch_size,
        }

        best_validation_loss = np.inf
        # Start with independent copies; later in-place updates must not change
        # the checkpoint. Each improved validation result replaces this tuple.
        best_parameters = (
            self.W_hidden.copy(), self.B_hidden.copy(),
            self.W_output.copy(), self.B_output.copy(),
        )

        # One epoch visits every training sample once.
        for epoch in range(epochs):
            order = self.rng.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                # The same indices keep each design paired with its frequency.
                indices = order[start:start + batch_size]
                X_batch = X_train[indices]
                Y_batch = Y_train[indices]
                predicted, cache = self.forward(X_batch)
                gradients = self.backward(predicted, Y_batch, cache)
                self.update_parameters(gradients, learning_rate)
                # backward() uses the actual batch size, including a short
                # final batch (e.g. 4 samples for 420 samples / batch size 32).

            # Evaluate both splits using the SAME parameters after this epoch.
            # Validation samples never enter backward() or parameter updates.
            training_pred, _ = self.forward(X_train)
            validation_pred, _ = self.forward(X_validation)
            training_loss = self.loss(training_pred, Y_train)
            validation_loss = self.loss(validation_pred, Y_validation)
            self.history["training_loss"].append(training_loss)
            self.history["validation_loss"].append(validation_loss)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self.best_epoch = epoch + 1
                best_parameters = (
                    self.W_hidden.copy(), self.B_hidden.copy(),
                    self.W_output.copy(), self.B_output.copy(),
                )

        # Restore the best validation epoch, which may precede the final epoch.
        self.W_hidden, self.B_hidden, self.W_output, self.B_output = best_parameters
        return self.history

    def predict(self, design_mm):
        """Predict GHz for one design [W_mm, L_mm, h_mm]; used later by PSO."""
        # Make a one-row batch. The forward pass already returns GHz.
        # This uses the trained surrogate; it never evaluates the antenna formula.
        X_mm = np.asarray(design_mm, dtype=float).reshape(1, 3)
        Y_GHz, _ = self.forward(X_mm)
        return float(Y_GHz[0, 0])

    def save(self, filename):
        """After fit(), save raw-unit parameters, settings and loss history.

        Use an .npz filename. This saves an inference model, not the random
        generator state needed to resume the exact training/shuffle sequence.
        """
        np.savez(
            filename,
            W_hidden=self.W_hidden, B_hidden=self.B_hidden,
            W_output=self.W_output, B_output=self.B_output,
            data_representation="raw_mm_GHz",
            hidden_size=self.hidden_size, seed=self.seed,
            best_epoch=self.best_epoch,
            epochs=self.training_settings["epochs"],
            learning_rate=self.training_settings["learning_rate"],
            batch_size=self.training_settings["batch_size"],
            training_loss=self.history["training_loss"],
            validation_loss=self.history["validation_loss"],
        )

    @classmethod
    def load(cls, filename):
        """Return a model restored from the archive written by save()."""
        # cls is the class itself, so no existing model is needed to call load.
        with np.load(filename) as data:
            # Old standardized weights require scaling during prediction and
            # cannot be interpreted as raw-unit weights by this version.
            if ("data_representation" not in data
                    or data["data_representation"].item() != "raw_mm_GHz"):
                raise ValueError("Load nn_model_raw.npz or retrain this model using raw mm/GHz data.")
            model = cls(
                hidden_size=int(data["hidden_size"]), seed=int(data["seed"]),
            )
            # Replace initialization with the learned parameters.
            model.W_hidden = data["W_hidden"].copy()
            model.B_hidden = data["B_hidden"].copy()
            model.W_output = data["W_output"].copy()
            model.B_output = data["B_output"].copy()
            model.best_epoch = int(data["best_epoch"])
            model.training_settings = {
                "epochs": int(data["epochs"]),
                "learning_rate": float(data["learning_rate"]),
                "batch_size": int(data["batch_size"]),
            }
            model.history = {
                "training_loss": data["training_loss"].tolist(),
                "validation_loss": data["validation_loss"].tolist(),
            }
        return model


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    # This archive also retains the original split indices and data/split seeds.
    dataset_path = project_root / "antennaSample_training.npz"
    with np.load(dataset_path) as data:
        train_indices = data["train_indices"]
        validation_indices = data["validation_indices"]
        # The archive has all 600 samples. Select the saved split rows so
        # training cannot use validation or test samples by accident.
        dataTrain = {
            "X_mm": data["X_mm"][train_indices],
            "Y_GHz": data["Y_GHz"][train_indices],
        }
        dataValidation = {
            "X_mm": data["X_mm"][validation_indices],
            "Y_GHz": data["Y_GHz"][validation_indices],
        }

        model = NeuralNetwork(hidden_size=8, seed=1)

    print("Training shapes:", dataTrain["X_mm"].shape, dataTrain["Y_GHz"].shape)
    print("Validation shapes:", dataValidation["X_mm"].shape, dataValidation["Y_GHz"].shape)
    print("Weight shapes:", model.W_hidden.shape, model.W_output.shape)
    print("Bias shapes:", model.B_hidden.shape, model.B_output.shape)
    print("Raw units: X in mm, Y in GHz; no standardization")
    print("Internal inputs: X = X_mm.T, then W_hidden @ X + B_hidden")
    print("Architecture: 3 -> 8 -> 1 (41 trainable parameters)")

    # Raw mm inputs can produce larger gradients, so start with a smaller
    # learning rate. Tune these settings using validation data.
    fit_start = perf_counter()
    history = model.fit(
        dataTrain, dataValidation,
        epochs=1000, learning_rate=0.0001, batch_size=32,
    )
    fit_seconds = perf_counter() - fit_start
    print(f"Training time: {fit_seconds:.3f} s")
    print(f"Best validation epoch: {model.best_epoch}")
    print(f"Best validation MSE (GHz^2): {min(history['validation_loss']):.6g}")

    # These errors use the restored best parameters and physical frequency units.
    for name, dataset in (
        ("Training", dataTrain),
        ("Validation", dataValidation),
    ):
        X_mm = dataset["X_mm"]
        Y_GHz = dataset["Y_GHz"]
        Y_pred_GHz, _ = model.forward(X_mm)
        rmse_MHz = 1000.0 * np.sqrt(np.mean((Y_pred_GHz - Y_GHz)**2))
        print(f"{name} RMSE: {rmse_MHz:.3f} MHz")

    output_directory = project_root / "results" / "nn"
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / "nn_model_raw.npz"
    model.save(model_path)
    restored = NeuralNetwork.load(model_path)
    print(f"Saved model and loss history: {model_path}")
    print(
        "Reloaded model prediction for [38, 30, 1.6] mm:",
        f"{restored.predict([35.614, 29.6778, 1.0]):.9f} GHz",
    )
    print("Test split reserved for final evaluation after choosing settings.")
