import random
import math
import numpy as np
'''
General simulated annealing algorithm
initial_state is the starting solution
- randomized appropriately through the domain space
initial_temp is the starting temperature
- best if matches closely with the general range of the function space
final_temp is the ending temperature
alpha is the step size of the temperature as it goes from initial_temp to final_temp
get_cost(x) is the cost function with which to evaluate state x
get_neighbors(x) gets the neighbors of state x
'''
def simulated_annealing_modified(initial_state, initial_temp, final_temp, alpha, initial_cost, get_cost_differential, get_neighbors_and_edge, print_iters = 50):
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
