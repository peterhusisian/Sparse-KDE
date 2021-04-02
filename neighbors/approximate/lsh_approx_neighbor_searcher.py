from neighbors.neighbor_searcher import NeighborSearcher
import numpy as np
import sklearn.neighbors

class LSHApproxNeighborSearcher(NeighborSearcher):

    def __init__(self, data, n_estimators=10, n_candidates=50, min_hash_match=4, radius_cutoff_ratio=0.9, random_state=None):
        NeighborSearcher.__init__(self, data)
        self.__lsh = sklearn.neighbors.LSHForest(n_estimators = n_estimators,\
            n_candidates = n_candidates,\
            min_hash_match = min_hash_match,\
            radius_cutoff_ratio = radius_cutoff_ratio,\
            random_state = random_state)
        self.__lsh.fit(self._data)

    def k_neighbors(self, X, k):
        return self.__lsh.kneighbors(X, n_neighbors = k, return_distance = False)
