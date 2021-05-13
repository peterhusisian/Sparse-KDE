import numpy as np


def gaussian_kernel(sigma_squared):
    def out(X):
        return (1.0 / np.sqrt(2 * np.pi * sigma_squared)) *\
            np.exp(-np.linalg.norm(X, axis = X.ndim - 1)**2 / (2 * sigma_squared))
    return out
