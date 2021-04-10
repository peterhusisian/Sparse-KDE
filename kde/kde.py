from probability.joint_distribution import JointDistribution
import numpy as np



class KDE(JointDistribution):

    #kernel_func applies the kernel function along the last axis of the input.
    #assumes the data is of shape (n, d), where n is the number of training examples,
    #and d is the number of features of the data
    def __init__(self, data, kernel_func):
        self.__data = data
        self.__kernel_func = kernel_func


    #expects X to be an (m,d) array, where m is the number of points upon which the joint
    #probability is to be evaluated
    def joint_prob(self, X):
        #entry [i,j] is X[i] - self.__data[j]
        #(m,n,d) =_broadcast (m,1,d), (1,n,d)
        X_residuals = X[:,np.newaxis,:] - self.__data[np.newaxis,:,:]
        kernel_outputs = self.__kernel_func(X_residuals)
        out = np.average(kernel_outputs, axis = -1)
        return out
