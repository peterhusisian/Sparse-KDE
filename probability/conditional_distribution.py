from abc import ABC

class ConditionalDistribution(ABC):

    '''
    computes P(x|y) element-wise -- that is, out[i] = P(X[i] | Y[i])
    '''
    @abstractmethod
    def conditional_prob(self, X, Y):
        pass
