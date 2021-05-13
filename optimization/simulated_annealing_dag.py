import numpy as np
import numpy_toolbox.adjugate as adjugate

'''
is j an ancestor of i in adjacency matrix dag A
'''
def isAncestor(A, j, i):
    checked = np.zeros(A.shape[0], dtype=bool)
    stack = [i]
    while(stack):
        current = stack.pop()
        if current == j:
            return True
        for parent in range(0, A.shape[0]):
            if A[parent, current]==1 and not checked[parent]:
                stack.append(parent)
                checked[parent]=True
    return False


'''
Get edge additions and deletion possibilities based on input matrix A
where A is an adjacency 2d numpy matrix of a DAG
'''
def neighbors_func(A):
    assert(A.shape[0] == A.shape[1])
    possibilities_list = []
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[0]):
            if A[i, j]!=0:
                #Produce neighbors with deleted edges
                B= np.empty_like(A)
                B[:] = A
                B[i, j]=0
                possibilities_list.append(B)
            else:
                #Checks and produces neighbors with added edges
                if not isAncestor(A, j, i):
                    B= np.empty_like(A)
                    B[:] = A
                    B[i, j]=1
                    possibilities_list.append(B)
    return possibilities_list
def degree_constrained_neighbors_func(max_deg):
    def out(A):

        neighbors = []
        for neighbor in neighbors_func(A):
            if np.sum(neighbor[0], axis = 0).max() <= max_deg:
                neighbors.append(neighbor)
        return neighbors
    return out

def exponentiation(A, exponent):
    '''
    if(exponent == 1):
        return A
    else:
        final_matrix = exponentiation(A, (exponent-(exponent % 2))/2)
        final_matrix = np.matmul(final_matrix, final_matrix)
        if(exponent % 2 == 1):
            final_matrix = np.matmul(final_matrix, A)
    return final_matrix
    '''
    return np.linalg.matrix_power(A, exponent)

def check_dag_properties(A):
    assert(A.shape[0] == A.shape[1])
    exponent = A.shape[0]
    final_matrix = exponentiation(A, exponent)
    if np.trace(final_matrix) == 0:
        return True
    else:
        return False

def single_degree_constrained_neighbor_func(max_deg):
    def out(A):
        r = np.random.rand()
        if np.sum(A) == 0 or r < 0.5:
            added_edge_dag = fast_add_random_dag_edge(A, max_deg)
            if added_edge_dag is not None:
                return [added_edge_dag]

        possible_edges_to_remove = np.argwhere(A != 0)
        out_dag = A.copy()
        remove_edge = possible_edges_to_remove[np.random.randint(0, possible_edges_to_remove.shape[0])]
        out_dag[remove_edge[0], remove_edge[1]] = 0
        return [(out_dag, (remove_edge[0], remove_edge[1]))]
    return out



def fast_add_random_dag_edge(A, max_deg):
    #sums[i,j] = number of parents of node[j]
    sums = np.outer(np.ones(A.shape[0]), np.sum(A, axis = 0))
    unfilled_edges = np.argwhere((A == 0) * (sums < max_deg) * (1 - np.eye(A.shape[0])))
    if (unfilled_edges.shape[0] == 0):
        return None
    np.random.shuffle(unfilled_edges)
    A = A.copy()
    for i in range(unfilled_edges.shape[0]):
        e = unfilled_edges[i]
        A[e[0], e[1]] = 1
        if check_dag_properties(A):
            assert((np.sum(A, axis = 0) <= max_deg).all())
            return (A, e)
        A[e[0], e[1]] = 0
    return None




'''
Get edge additions and deletion possibilities based on input matrix A
where A is an adjacency 2d numpy matrix of a DAG
'''
def neighbors_func(A):
    assert(A.shape[0] == A.shape[1])
    possibilities_list = []
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[0]):
            if A[i, j]!=0:
                #Produce neighbors with deleted edges
                B= np.empty_like(A)
                B[:] = A
                B[i, j]=0
                possibilities_list.append((B, (i, j)))
            else:
                #Checks and produces neighbors with added edges
                if not isAncestor(A, j, i):
                    B= np.empty_like(A)
                    B[:] = A
                    B[i, j]=1
                    possibilities_list.append((B, (i, j)))
    return possibilities_list
if __name__ == "__main__":
    a = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 0]])
    assert(check_dag_properties(a))
    print(neighbors_func(a))
