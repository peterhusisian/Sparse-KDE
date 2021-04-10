from abc import ABC, abstractmethod

class ConditionalSampleable(ABC):

    '''
    generates Y.shape[0] samples, X[0],...,X[Y.shape[0] - 1], such that
    X[i] ~ P(x | Y[i]).

    As such, this class is essentially a wrapper for functions that generate
    random samples as a function of some input. That is, it implements a random
    function f(y) (if that makes sense). It need not only be implemented
    by ConditionalDistributions only (although that was the main purpose of
    its creation)
    '''
    @abstractmethod
    def sample(self, Y):
        pass
