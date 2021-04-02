import numpy as np

#returns a topological sort of dag using Kahn's algorithm
def topological_sort(dag):
    dag = dag.copy()
    indegrees = np.sum(dag, axis = 0)
    L = np.empty(dag.shape[0], dtype = np.int)
    iters = 0
    while (indegrees != 0).any():
        #finds first occurrence of a node with no incoming edges
        n = np.argmax(indegrees == 0)
        if indegrees[n] != 0:
            raise ValueError("Cannot topoligically sort a non-DAG")

        L[iters] = n
        n_as_parent_nodes = np.argwhere(dag[n] != 0)
        dag[n,:] = 0
        indegrees[n_as_parent_nodes] -= 1
        iters += 1
    return L
