import numpy as np

def random_permutation_matrix(n):
    out = np.eye(n)
    np.random.shuffle(out)
    return out.astype(np.int)
