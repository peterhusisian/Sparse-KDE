from probability.bayesian_net.sampleable_bayesian_net import SampleableBayesianNet
from probability.bayesian_net.bayesian_net import BayesianNet
from probability.linear_conditioned_gaussian import LinearConditionedGaussian
import numpy as np


class LinearGaussianBayesianNet(SampleableBayesianNet):

    '''
    Constructs a bayesian network where each node is randomly distributed as follows:
    - p(x[i] | pa(x[i])) = Gaussian with mean W[:, i]^T x + b[i], variance sigma_squareds[i]

    Where inputs are d-dimensional, initialization terms have shape:
    - W: d x d
    - b: d
    - sigma_squareds: d

    Where lack of conditional dependence is indicated by the corresponding weight in W being zero.

    REQUIRES: The non-zero entries of W constitute a DAG
    '''
    def __init__(self, W, b, sigma_squareds):
        dag, dists = self.__construct_dag_and_conditional_dists(W, b, sigma_squareds)
        SampleableBayesianNet.__init__(self, dag, dists)




    def __construct_dag_and_conditional_dists(self, W, b, sigma_squareds):
        dag = W.copy()

        dag[np.where(W != 0)] = 1
        dag = dag.astype(np.int)
        dists = []
        for i in range(W.shape[0]):
            pa_i = np.argwhere(dag[:,i] != 0)[:,0]
            dists.append(LinearConditionedGaussian(W[pa_i, i], b[i], sigma_squareds[i]))
        return dag, dists
