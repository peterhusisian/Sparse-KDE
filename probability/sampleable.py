from abc import ABC, abstractmethod

class Sampleable(ABC):

    @abstractmethod
    def sample(self, n_samples):
        pass
