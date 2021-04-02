from probability.conditional_distribution import ConditionalDistribution

class JointFormedConditionalDistribution(ConditionalDistribution):
    '''
    models p(x|y) using the fact that:
        p(x|y) = p(x,y) / p(y)

    Where x and y are vectors (of type compatible with argument dtype). Let x be of length m and y of length n.
    Expects:
        - numerator_dist: a JointDistribution that expects p([x[0],...,x[m-1],y[0],...,y[n-1]])
        - denominator_dist: a JointDistribution that expects p(y)
        - dtype: the numpy array type such that both x and y could be stored in an array of this type
    '''
    def __init__(self, numerator_dist, denominator_dist, dtype = np.float64):
        self.__numerator_dist = numerator_dist
        self.__denominator_dist = denominator_dist
        self.__dtype = dtype

    def conditional_prob(self, X, Y):
        X_Y_merged = np.zeros((X.shape[1] + Y.shape[1]), dtype = self.__dtype)
        X_Y_merged[:X.shape[1]] = X
        X_Y_merged[X.shape[1]:] = Y
        out = self.__numerator_dist(X_Y_merged)
        out /= self.__denominator_dist(Y)
        return out
