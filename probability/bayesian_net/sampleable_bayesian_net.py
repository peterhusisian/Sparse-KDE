from probability.bayesian_net.bayesian_net import BayesianNet
from probability.sampleable import Sampleable
from abc import ABC
import graph.topological_sort as topological_sort

class SampleableBayesianNet(BayesianNet, Sampleable):

    def __init__(self, dag, conditional_sampleable_dists):
        BayesianNet.__init__(self, dag, conditional_sampleable_dists)

    
    def sample(self, n_samples):
        #use a topological sort to determine sample order
        order = topological_sort.topological_sort(self._dag)
        out = np.empty((n_samples, self._dag.shape[0]), dtype = np.float64)
        for i in order:
            i_parent_values = out[:, self._parents_of(i)]
            out[:, i] = self._conditional_dists[i].sample(i_parent_values)
        return out
