from abc import ABC

class JointDistribution(ABC):

    
    @abstractmethod
    def joint_prob(self, X):
        pass
