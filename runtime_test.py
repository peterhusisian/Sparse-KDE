from probability.bayesian_net.kde_bayesian_net import KDEBayesianNet
import numpy as np
import rand.random_graph as random_graph
from probability.bayesian_net.linear_gaussian_bayesian_net import LinearGaussianBayesianNet
import optimization.objectives.kde_bayesian_net_log_likelihood as kde_bayesian_net_log_likelihood
import kde.kernel as kernel
import optimization.graph_differential_simulated_annealing as simulated_annealing
import optimization.simulated_annealing_dag as simulated_annealing_dag
import timeit
from matplotlib import pyplot as plt



def silverman_scalar_bandwidth(training_data):
    n,d = training_data.shape
    out = 0
    const = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4))
    for i in range(d):
        out += const * np.std(training_data[:,i])
    return out / d

N = 100
d_range = (5, 50)
d_inc = 5
p = 0.5
max_deg = 5
percent_train = 0.8
#bandwidth = 10
n_iters = 500
runtimes = []
for d in range(d_range[0], d_range[1] + 1, d_inc):
    dag = random_graph.random_dag(d, p)

    W = np.random.rand(d,d) * dag.astype(np.float64)
    b = np.random.rand(d)
    sigma_squareds = np.random.rand(d) + 1
    LGBN = LinearGaussianBayesianNet(W, b, sigma_squareds)
    X = LGBN.sample(N)
    X_train = X[:int(percent_train * X.shape[0])]
    X_test = X[int(percent_train * X.shape[0]):]

    bandwidth = silverman_scalar_bandwidth(X)
    print("d: ", d)
    print("bandwidth: ", bandwidth)

    initial_dag = random_graph.random_max_deg_dag(X.shape[1], max_deg)
    print("initial_dag: ", np.sum(initial_dag, axis = 0))

    initial_temp = kde_bayesian_net_log_likelihood.bayesian_net_log_likelihood(X_train, X_test, kernel.gaussian_kernel(bandwidth), 0, 0, negative = True)(initial_dag)

    start_time = timeit.default_timer()
    final_temp = initial_temp / 4
    alpha = (initial_temp - final_temp) / float(n_iters)
    opt_dag = simulated_annealing.simulated_annealing(initial_dag, \
        initial_temp,\
        final_temp,\
        alpha, \
        initial_temp,\
        kde_bayesian_net_log_likelihood.bayesian_net_log_likelihood_differential(X_train, X_test, kernel.gaussian_kernel(bandwidth), 0, 0, negative = True),\
        simulated_annealing_dag.single_degree_constrained_neighbor_func(max_deg),\
        print_iters = n_iters)

    runtimes.append((d, timeit.default_timer() - start_time))



print("runtimes: \n", runtimes)

runtime_dimensions = [x[0] for x in runtimes]
runtime_times = [x[1] for x in runtimes]
plt.plot(runtime_dimensions, runtime_times)
plt.show()
