from abc import ABC

class Sampleable(ABC):

    @abstractmethod
    def sample(self, n_samples):
        pass
