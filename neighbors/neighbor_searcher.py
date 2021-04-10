from abc import ABC, abstractmethod
import numpy as np

class NeighborSearcher(ABC):

    def __init__(self, data):
        self._data = data.copy()

    def query_points(self, inds):
        out_shape = inds.shape + (self._data.shape[1],)
        return np.reshape(self._data[inds.flatten()], out_shape).copy()

    @abstractmethod
    def k_neighbors(self, X, k):
        pass
