from probability.conditional_sampleable import ConditionalSampleable
from probability.conditional_distribution import ConditionalDistribution
import numpy as np

class LinearConditionedGaussian(ConditionalDistribution, ConditionalSampleable):

    '''
    Models the conditional distribution:
    P(x | y) ~ normal with mean w^T y + b and variance sigma_squared

    If y is of length d, then:
    - w is a d-length numpy array
    - b is a scalar
    '''
    def __init__(self, w, b, sigma_squared):
        ConditionalDistribution.__init__(self)
        ConditionalSampleable.__init__(self)
        self.__w = w
        self.__b = b
        self.__sigma_squared = sigma_squared

    def __get_means(self, Y):
        return np.dot(Y, self.__w) + self.__b

    def conditional_prob(self, X, Y):
        return (1.0 / (np.sqrt(2 * self.__sigma_squared * np.pi))) * np.exp(-0.5 * (X - self.__get_means(Y))**2 / sigma_squared)

    def sample(self, Y):
        mus = self.__get_means(Y)
        return np.random.normal(loc = mus, scale = np.sqrt(self.__sigma_squared))
