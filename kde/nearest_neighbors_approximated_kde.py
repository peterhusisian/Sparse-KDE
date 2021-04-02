from probability.joint_distribution import JointDistribution

'''
TODO: if neccessary. Problem is there is no way to guarantee the KDE integrates to 1.

Further, even when kernel has finite support, you can't vectorize nearest-neighbor-within-radius
search as each point has a different number of neighbors. Because python is slow, the cost of
iteration will almost certainly dominate computation time unless evaluating the joint
probability on small sets of points
'''
SAMPLE_AVERAGE = 'sample_avg'
EXPONENTIAL_FIT = 'exp_fit'

class NearestNeighborsApproximatedKDE(JointDistribution):



    '''
    Is almost certainly guaranteed to not integrate to 1 due to neighbors left out of the summation.

    Assumes the dataset_neighbor_searcher makes sense to use with the chosen kernel_func. e.g.
    if kernel_func is a parzen window (L1 distance threshold), it would be odd to use a
    dataset_neighbor_searcher that finds the nearest points by L2 distance (probably not that big of a
    deal, though). So long as dataset_neighbor_searcher does an adequate job of finding points
    that will yield high kernel outputs from kernel_func, it should be fine.

    kernel_func is a function that types in an arbitrary-shaped numpy array and element-wise
    outputs the kernel function applied to each vector in the array, where the last axis
    is the axis along which the values of each point are stored. That is, the input is
    assumed to have a final index corresponding to the dimensions of a point.
    '''
    def __init__(self, dataset_neighbor_searcher, num_neighbors, kernel_func, non_neighbor_approximation_method = SAMPLE_AVERAGE):
        self.__dataset_neighbor_searcher = dataset_neighbor_searcher
        self.__num_neighbors = num_neighbors
        self.__kernel_func = kernel_func

    def joint_prob(self, X):
        X_neighbor_inds = self.__dataset_neighbor_searcher.k_neighbors(X, self.__num_neighbors)
        #NOT FINISHED
        return None
