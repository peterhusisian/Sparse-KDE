import numpy as np
from probability.bayesian_net.bayesian_net import KDEBayesianNetwork

'''
returns: a function which takes in a list of DAGs, dags, and returns the log-likelihoods
of the dataset X_test (with rows that are data points) according to the KDE
bayesian network initialized with training data X_train and kernel kernel.
'''
def bayesian_net_log_likelihood(X_train, X_test, kernel, lambd, regularizer_order, negative = False):
    #There is a possibility that a log-sum-exp trick may need to be used
    #in order to prevent numerical errors within the joint probability calculations
    #in the bayesian network. if this is the case, we should also add a "log_joint_prob"
    #function to the BayesianNetwork implementation.
    def out(dags):
        wasnt_list = False
        if not isinstance(dags, list):
            dags = [dags]
            wasnt_list = True
        likelihoods = np.zeros(len(dags), dtype = np.float64)
        for i in range(likelihoods.shape[0]):
            dag = dags[i]
            bayesian_net = KDEBayesianNetwork(dag, X_train, kernel)
            probs = bayesian_net.joint_prob(X_test)
            likelihoods[i] = np.sum(np.log(probs)) - lambd * np.sum(dag)**regularizer_order

        if negative:
            likelihoods *= -1

        if wasnt_list:
            return likelihoods[0]

        return likelihoods
        '''
        likelihoods = np.zeros(len(dags), dtype = np.float64)
        for i in range(likelihoods.shape[0]):
            dag = dags[i]
            bayesian_net = KDEBayesianNetwork(dag, X_train, kernel)
            probs = np.zeros(X_test.shape[0], dtype = np.float64)
            for j in range(X_test.shape[0]):
                probs[j] = bayesian_net.joint_prob(X_test[j])
            likelihoods[i] = np.sum(np.log(probs)) - lambd * np.sum(dag)**regularizer_order
        return likelihoods
        '''

    return out
'''
returns: a function which takes in a list of DAGs, dags, and returns the log-likelihoods
of the dataset X_test (with rows that are data points) according to the KDE
bayesian network initialized with training data X_train and kernel kernel.
'''
def bayesian_net_log_likelihood_differential(X_train, X_test, kernel, lambd, regularizer_order, negative = False):
    #There is a possibility that a log-sum-exp trick may need to be used
    #in order to prevent numerical errors within the joint probability calculations
    #in the bayesian network. if this is the case, we should also add a "log_joint_prob"
    #function to the BayesianNetwork implementation.
    def out(dags, edges):
        wasnt_list = False
        if not isinstance(dags, list):
            dags = [dags]
            wasnt_list = True
        likelihood_differential = np.zeros(len(dags), dtype = np.float64)
        for i in range(likelihood_differential.shape[0]):
            dag = dags[i]
            bayesian_net = KDEBayesianNetwork(dag, X_train, kernel)
            if dag[edge[0], edge[1]]==1:
                dag[edge[0], edge[1]] = 0
            else:
                dag[edge[0], edge[1]] = 1
            bayesian_net_new = KDEBayesianNetwork(dag, X_train, kernel)
            prob_differential = bayesian_net_new.node_prob(X_test, edge[1]))/bayesian_net.node_prob(X_test,edge[1]))
            likelihood_differential[i] = np.sum(np.log(prob_differential)) + lambd * np.sum(dag)**regularizer_order - (lambd * np.sum(new_dag)**regularizer_order)

        if negative:
            likelihood_differential *= -1

        if wasnt_list:
            return likelihood_differential[0]

        return likelihood_differential
        '''
        likelihoods = np.zeros(len(dags), dtype = np.float64)
        for i in range(likelihoods.shape[0]):
            dag = dags[i]
            bayesian_net = KDEBayesianNetwork(dag, X_train, kernel)
            probs = np.zeros(X_test.shape[0], dtype = np.float64)
            for j in range(X_test.shape[0]):
                probs[j] = bayesian_net.joint_prob(X_test[j])
            likelihoods[i] = np.sum(np.log(probs)) - lambd * np.sum(dag)**regularizer_order
        return likelihoods
        '''

    return out
