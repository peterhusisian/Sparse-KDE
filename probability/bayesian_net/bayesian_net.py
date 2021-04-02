from probability.joint_distribution import joint_distribution

class BayesianNet(JointDistribution):

    def __init__(self, dag, conditional_dists):
        self._dag = dag
        self._conditional_dists = conditional_dists

    def _parents_of(self, node):
        return np.argwhere(self._dag[:,node] != 0)


    def node_prob(self, i, X):
        i_parents = _parents_of(i)
        x_i_values = X[:,i]
        parent_values = X[:,i_parents]
        return self._conditional_dists[i].conditional_prob(x_i_values, parent_values)

    def joint_prob_from_node_probs(self, node_probs):
        return np.product(node_probs, axis = 1)

    def joint_prob(self, X):
        out = np.ones(X.shape[0], dtype = np.float64)
        for i in range(X.shape[1]):
            out *= node_prob(i, X)
        return out
