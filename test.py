from probability.bayesian_net.kde_bayesian_net import KDEBayesianNet
import numpy as np
import data_loader.load_data as load_data
import numpy as np
import optimization.objectives.kde_bayesian_net_log_likelihood as kde_bayesian_net_log_likelihood
import optimization.graph_differential_simulated_annealing as simulated_annealing
import optimization.simulated_annealing_dag as simulated_annealing_dag
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
from rand.sampling import bayesian_net_sampler
from rand import random_graph as random_graph
from scipy.ndimage import gaussian_filter1d
from kde.kernel import gaussian_kernel

def silverman_scalar_bandwidth(training_data):
    n,d = training_data.shape
    out = 0
    const = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4))
    for i in range(d):
        out += const * np.std(training_data[:,i])
    return out / d
max_deg = 2
X = load_data.load_clean_alfredo_saurez("../adolfosuarez20191008-20191027.csv", "Iberia").to_numpy()

X_train = X[:int(0.7 * X.shape[0]), :]
X_test = X[int(0.7 * X.shape[0]):, :]
bandwidth = silverman_scalar_bandwidth(X_train)

initial_dags = [random_graph.random_max_deg_dag(X.shape[1], max_deg) for i in range(0, 1)]


initial_temp = kde_bayesian_net_log_likelihood.bayesian_net_log_likelihood(X_train, X_test, gaussian_kernel(bandwidth), 0, 0, negative = True)(initial_dags[0])
final_temp = initial_temp / 4
alpha = (initial_temp - final_temp) / 5000
opt_dag = simulated_annealing.simulated_annealing(initial_dags[0], \
    initial_temp,\
    final_temp,\
    alpha, \
    initial_temp,\
    kde_bayesian_net_log_likelihood.bayesian_net_log_likelihood_differential(X_train, X_test, gaussian_kernel(bandwidth), 0, 0, negative = True),\
    simulated_annealing_dag.degree_constrained_neighbors_func(max_deg),\
    print_iters = 10)
