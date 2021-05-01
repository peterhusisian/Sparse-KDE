import random
import math
import numpy as np

'''
Performs simulated annealing on graph optimization problems when the objective can be
efficiently posed as an "update" of the previous objective evaluation. Simulated
adds or removes exactly one edge at a time -- in some instances, if the edge
added/removed is known, one can efficiently update the prior cost to receive
a new cost, and this version of simulated annealing is built around evaluating
such objectives in the scheme of the simulated annealing optimization algorithm.
'''
def simulated_annealing(initial_state, initial_temp, final_temp, alpha, initial_cost, get_cost_differential, get_neighbors_and_edge, print_iters = 50):
    T = initial_temp
    x = initial_state
    e = initial_cost
    iter_num = 0

    best_x = x
    best_e = e

    while T > final_temp:
        x_prime, (i, j) = random.choice(get_neighbors_and_edge(x))
        cost_differential = get_cost_differential(x, (i, j))
        e_prime = e+cost_differential
        if iter_num % print_iters == 0:
            print("best cost(" + str(iter_num) + "): ", best_e)
            print("best iterate(" + str(iter_num) + "): \n", best_x)
            print("---------------------------------------------------------")

        if e_prime < best_e:
            best_x = x_prime
            best_e = e_prime


        if e_prime < e:
            #always jump to x_prime when it is better than x
            x = x_prime
            e = e_prime
        else:
            if np.random.rand() <= np.exp(-(e_prime - e) / T):
                #jump to x_prime randomly even though it is worse than x
                x = x_prime
                e = e_prime

        T -= alpha
        iter_num += 1

    return best_x
