from probability.bayesian_net.kde_bayesian_net import KDEBayesianNet
import numpy as np

def test_dud_kernel_func(X):
    return np.ones(X.shape[0], dtype = np.float64)




DAG = np.array([[0,1,1,1],[0,0,1,1],[0,0,0,1],[0,0,0,0]])
X = np.tile(np.arange(0,4), (10,1))
kde_bayesian_net = KDEBayesianNet(X, DAG, test_dud_kernel_func)


kde_bayesian_net.joint_prob(np.array([[1,2,3,4],[1,2,3,4]]))
