from probability.bayesian_net.sampleable_bayesian_net import SampleableBayesianNet
from probability.bayesian_net.bayesian_net import BayesianNet
from kde.kde import KDE
from probability.joint_formed_conditional_distribution import JointFormedConditionalDistribution
from probability.independent_conditional_distribution import IndependentConditionalDistribution
import numpy as np


class KDEBayesianNet(BayesianNet):

    def __init__(self, X, dag, kernel_func):
        BayesianNet.__init__(self, dag, self.__construct_conditional_dists(X, dag, kernel_func))

    def __construct_conditional_dists(self, X, dag, kernel_func):
        out = []
        for i in range(dag.shape[0]):
            i_parents = np.argwhere(dag[:,i] != 0)[:,0]
            if (i_parents.shape[0] == 0):
                out.append(IndependentConditionalDistribution(KDE(X[:,i,np.newaxis].copy(), kernel_func)))
            else:
                denominator_dataset = X[:, i_parents].copy()
                denominator_dist = KDE(denominator_dataset.copy(), kernel_func)
                numerator_dataset = np.insert(denominator_dataset, 0, X[:,i], axis = 1)
                numerator_dist = KDE(numerator_dataset, kernel_func)
                out.append(JointFormedConditionalDistribution(numerator_dist, denominator_dist))
        return out
