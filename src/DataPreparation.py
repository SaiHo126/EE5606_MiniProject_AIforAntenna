import numpy as np

def prepare_dataset(filename: str, split_seed: int =1):
    # load saved data set
    with np.load(filename) as data:
        X = data["X_mm"]
        Y = data["Y_GHz"]
        data_seed = data["seed"].item()
    # Stay X and Y paired
    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(len(X))
    train_end = int(0.7 * len(X))
    validation_end = train_end + int(0.15*len(X))
    # indices of the 3 parts of data
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]
    # extract the data for training
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    # calculate the statistic for the training data
    X_mean_train = X_train.mean(axis = 0, keepdims = True)
    X_std_train = X_train.std(axis = 0, keepdims = True)
    Y_mean_train = Y_train.mean(axis = 0, keepdims = True)
    Y_std_train = Y_train.std(axis = 0, keepdims = True)
    # save the data 
    np.savez(
        "antennaSample_training.npz",
        X_mm = X,
        Y_GHz = Y,
        train_indices = train_indices,
        validation_indices = validation_indices,
        test_indices = test_indices,
        X_mean_train = X_mean_train,
        X_std_train = X_std_train,
        Y_mean_train = Y_mean_train,
        Y_std_train = Y_std_train,
        data_seed = data_seed,
        split_seed = split_seed
    )
if __name__ == "__main__":
    prepare_dataset("antennaSample_dataset.npz", split_seed=1)