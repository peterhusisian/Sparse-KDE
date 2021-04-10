from abc import ABC, abstractmethod

class JointDistribution(ABC):

    
    @abstractmethod
    def joint_prob(self, X):
        pass
