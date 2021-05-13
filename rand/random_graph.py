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

'''
Generates an adjacency matrix for a Directed Acyclic Graph over n nodes, where
each has out-degree of at most max_out_deg.

A minimum degree argument is not included, as all DAGs must have at least one
node with degree zero (such a node must exist to be placed at the end of the
topological sort of the DAG)
'''
def random_max_deg_dag(n, max_out_deg):
    U = np.zeros((n, n), dtype = np.int)
    for i in range(0, U.shape[0] - 1):
        i_out_deg = np.random.randint(0, min(U.shape[1] - i, max_out_deg))
        i_children = np.random.choice(np.arange(i + 1, U.shape[0]), size = i_out_deg, replace = False)
        U[i, i_children] = 1
    U = U.T
    P = random_matrix.random_permutation_matrix(n)
    return np.dot(np.dot(P.T, U), P)
