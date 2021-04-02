from neighbors.neighbor_searcher import NeighborSearcher
import numpy as np

class BruteForceNeighborSearcher(NeighborSearcher):

    def __init__(self, data, norm_order):
        NeighborSearcher.__init__(self, data)
        self.__norm_order = norm_order

    def k_neighbors(self, X, k):
        if k == self._data.shape[0]:
            out = np.empty((X.shape[0], k), dtype = np.int)
            out += np.arange(0, k, dtype = np.int)
            return out

        #element [i,j] is X[i] - self._data[j]
        residuals = X[:,:,np.newaxis] - self._data

        #element [i,j] is self.__norm_func(X[i] - self._data[j])
        dists = np.linalg.norm(residuals, ord = self.__norm_order, axis = 2)

        return np.argpartition(dists, k, axis = -1)
