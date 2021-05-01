import numpy as np

<<<<<<< HEAD
=======
'''
generates a random n x n permutation matrix (dtype is np.int)
'''
>>>>>>> c44b128b60d2c9a840e4ceafd6be30f8d5562de8
def random_permutation_matrix(n):
    out = np.eye(n)
    np.random.shuffle(out)
    return out.astype(np.int)
