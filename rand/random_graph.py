import numpy as np
import rand.random_matrix as random_matrix


'''
creates a random DAG as follows:

Creates a random upper-triangular matrix with 0 diagonal, where an upper-diagonal
element 1 is added with probability p. As all DAGs must have some permutation such
that permuting its rows and columns that way results in an upper triangular matrix
with zero diagonal. This process guarantees this is the case.

Then, a random permutation is applied to the rows and columns of this matrix.
'''
def random_dag(n, p):
    out = np.zeros((n, n), dtype = np.int)
    for i in range(out.shape[0]):
        for j in range(i + 1, out.shape[1]):
            if np.random.rand() <= p:
                out[i,j] = 1
    P = random_matrix.random_permutation_matrix(n)
    return P.T @ out @ P
