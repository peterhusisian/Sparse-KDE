from probability.joint_distribution import JointDistribution

class KDE(JointDistribution):

    #kernel_func applies the kernel function along the last axis of the input
    def __init__(self, data, kernel_func):
        self.__data = data
        self.__kernel_func = kernel_func

    def joint_prob(self, X):
        #entry [i,j] is X[i] - self.__data[j]
        X_residuals = X[:,:,np.newaxis] - self.__data
        kernel_outputs = self.__kernel_func(X_residuals)
        return np.average(kernel_outputs, axis = -1)
