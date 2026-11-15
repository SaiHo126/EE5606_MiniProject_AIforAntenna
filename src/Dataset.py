from PatchFormula import RMA, AntennaBounds_mm
import numpy as np
# setting config
seed = 1
number_of_sample = 600

def generate_dataset( number_of_sample: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = np.empty((number_of_sample,3))
    Y = np.empty((number_of_sample,1))
    cout = 0
    while cout < number_of_sample:
        # sample [W,L,H] &  resonance frequency within AntennaBounds
        design = rng.uniform(
            AntennaBounds_mm[:,0],
            AntennaBounds_mm[:,1]
        )
        W,L,h = design
        # Reject invalid geometry design
        if W/L <1.2:
            continue
        # creating RMA component 
        antenna = RMA(W,L,h)
        X[cout] = design
        Y[cout] = antenna.ResonanceF_GHz
        cout += 1
    return X, Y

if __name__ == "__main__":
    X, Y = generate_dataset(number_of_sample = number_of_sample ,seed = seed)
    print("Input shape", X.shape)
    print("Frequency shape", Y.shape)
    np.savez(
        "antennaSample_dataset",
        X_mm=X,
        Y_GHz = Y,
        seed = seed
             )