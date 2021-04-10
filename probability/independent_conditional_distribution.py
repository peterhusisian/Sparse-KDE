from probability.conditional_distribution import ConditionalDistribution

'''
Models a conditional distribution, p(x|y), where x is INDEPENDENT of y, meaning
p(x|y) = p(x). This is useful in niche cases -- such as defining the conditional
probabilities of nodes in a bayesian network that have no parents. Rather
than clunkily forcing those nodes to have JointDistributions instead of
ConditionalDistributions, you can simply ignore the conditioned upon factor,
since such nodes have no conditional dependencies anyway
'''

class IndependentConditionalDistribution(ConditionalDistribution):

    def __init__(self, joint_dist):
        self.__joint_dist = joint_dist

    def conditional_prob(self, X, Y):
        return self.__joint_dist.joint_prob(X)
