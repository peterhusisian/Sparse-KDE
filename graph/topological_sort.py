import numpy as np

#returns a topological sort of dag using Kahn's algorithm
def topological_sort(dag):
    indegrees = np.sum(dag, axis = 0)
    L = np.zeros(dag.shape[0], dtype = np.int)
    iters = 0
    zero_indegree_nodes = []
    for i in range(indegrees.shape[0]):
        if indegrees[i] == 0:
            zero_indegree_nodes.append(i)

    while len(zero_indegree_nodes) != 0:
        L[iters] = zero_indegree_nodes.pop()
        for i in range(indegrees.shape[0]):
            if dag[L[iters], i] != 0:
                indegrees[i] -= 1
                if indegrees[i] == 0:
                    zero_indegree_nodes.append(i)

        iters += 1

    if iters != dag.shape[0]:
        raise ValueError("Cannot topologically sort a non-DAG")
    return L
