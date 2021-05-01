from probability.bayesian_net.kde_bayesian_net import KDEBayesianNet
import numpy as np
import rand.random_graph as random_graph
from probability.bayesian_net.linear_gaussian_bayesian_net import LinearGaussianBayesianNet

N = 1000
d_range = (5, 50)
d_inc = 5
p = 0.5

for d in d_range(d_range[0], d_range[1] + 1, d_inc):
    dag = random_graph.random_dag(d, p)
    W = np.random.rand(d,d) * dag.astype(np.float64)
    b = np.random.rand(d)
    sigma_squareds = 10*np.random.rand(d) + 1
    LGBN = LinearGaussianBayesianNet(W, b, sigma_squareds)
    X = LGBN.sample(N)

    
