from probability.bayesian_net.kde_bayesian_net import KDEBayesianNet
import numpy as np
import rand.random_graph as random_graph
from probability.bayesian_net.linear_gaussian_bayesian_net import LinearGaussianBayesianNet

'''
def test_dud_kernel_func(X):
    return np.ones(X.shape[0], dtype = np.float64)




DAG = np.array([[0,1,1,1],[0,0,1,1],[0,0,0,1],[0,0,0,0]])
X = np.tile(np.arange(0,4), (10,1))
kde_bayesian_net = KDEBayesianNet(X, DAG, test_dud_kernel_func)


kde_bayesian_net.joint_prob(np.array([[1,2,3,4],[1,2,3,4]]))
'''

d = 10
p = 0.5
dag = random_graph.random_dag(d, p)
print("dag: ", dag)
W = np.random.rand(d,d) * dag.astype(np.float64)
b = np.random.rand(d)
sigma_squareds = 100*np.random.rand(d) + 1

LGBN = LinearGaussianBayesianNet(W, b, sigma_squareds)

print(LGBN.sample(10))
