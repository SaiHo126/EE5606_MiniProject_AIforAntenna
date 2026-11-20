import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

data = np.load("antennaSample_training.npz")
X = data["X_mm"]
Y = data["Y_GHz"].ravel()

split = np.load("antennaSample_training.npz")
train = split["train_indices"]
test = split["test_indices"]

# Learn scaling from training data only.
scaler = StandardScaler()
X_train = scaler.fit_transform(X[train])
X_test = scaler.transform(X[test])
# Create and train the GP.
gp = GaussianProcessRegressor(
    kernel=RBF(length_scale=[1.0, 1.0, 1.0]),
    alpha=1e-8,
    normalize_y=True,
    random_state=1,
)
gp.fit(X_train, Y[train])

predicted = gp.predict(X_test)
rmse = np.sqrt(np.mean((predicted - Y[test]) ** 2))
print("Test RMSE:", rmse * 1000, "MHz")
# Predict frequency for a new antenna.
design = [[35.614, 29.6778, 1.0]]
frequency = gp.predict(scaler.transform(design))
print("Predicted frequency:", frequency[0], "GHz")
