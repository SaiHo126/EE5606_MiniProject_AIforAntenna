import numpy as np

def prepare_dataset(filename: str, split_seed: int =1):
    # load saved data set
    with np.load(filename) as data:
        X = data["X_mm"]
        Y = data["Y_GHz"]
        data_seed = data["seed"].item()
        
    