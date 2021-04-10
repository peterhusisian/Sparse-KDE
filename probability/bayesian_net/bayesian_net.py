from probability.joint_distribution import JointDistribution
import numpy as np
import numpy_toolbox.autobox as autobox

class BayesianNet(JointDistribution):

    '''
    If there is a node with no parents, expects its conditional_dist to be
    conditioned upon nothing. That is, p(x[i] | pa(x[i])) = p(x[i])

    This conditional dist will always be FED nothing on the conditional side
    '''
    def __init__(self, dag, conditional_dists):
        self._dag = dag
        self._conditional_dists = conditional_dists

    def _parents_of(self, node):
        return np.argwhere(self._dag[:,node] != 0)[:,0]


    def node_prob(self, i, X):
        i_parents = self._parents_of(i)
        x_i_values = X[:,i,np.newaxis]
        parent_values = X[:,i_parents]
        return self._conditional_dists[i].conditional_prob(x_i_values, parent_values)


    def joint_prob(self, X):
        out = np.ones(X.shape[0], dtype = np.float64)
        for i in range(X.shape[1]):
            out *= self.node_prob(i, X)
        return out
