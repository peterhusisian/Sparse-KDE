import numpy as np

'''
Calculates cofactor matrix using SVD as described in here:

https://stackoverflow.com/questions/6527641/speed-up-python-code-for-computing-matrix-cofactors
'''
def cofactors(A):
    U,sigma,Vt = np.linalg.svd(A)
    N = len(sigma)
    g = np.tile(sigma,N)
    g[::(N+1)] = 1
    G = np.diag(-(-1)**N*np.product(np.reshape(g,(N,N)),1))
    return U @ G @ Vt


def adjugate(A):
    return cofactors(A).T
