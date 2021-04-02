from probability.joint_distribution import JointDistribution
from sklearn.neighbors import KernelDensity
import numpy as np

class SklearnKDE(JointDistribution):

    def __init__(self, data, bandwidth, kernel):
        self.__kde = KernelDensity(bandwidth = bandwidth, kernel = kernel).fit(data)


    def joint_prob(self, X):
        return np.exp(self.__kde.score_samples(X))
