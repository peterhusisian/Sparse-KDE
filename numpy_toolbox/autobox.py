import numpy as np

'''
Adds or removes dud (size 1) dimensions to X such that the length of its shape
equals the integer shape_size. That is, if shape_size = 2 and X is a vector,
it'll return [X].

Useful for when you have functions defined element-wise, but you still want
them to handle evaluation on a single element as well
'''
def autobox(X, shape_size):
    if (len(X.shape) == shape_size):
        return X

    if (len(X.shape) < shape_size):
        return __prepend_dud_dims(X, shape_size)

    return __remove_prepended_dud_dims(shape_size, X)


def autobox_like(X, Y):
    return autobox(X, len(Y.shape))

def __prepend_dud_dims(X, shape_size):
    if len(X.shape) == shape_size:
        return X
    single_dud_prepend = np.zeros((1,) + X.shape, dtype = X.dtype)
    single_dud_prepend[0] = X
    return __prepend_dud_dims(single_dud_prepend, shape_size)

def __remove_prepended_dud_dims(shape_size, X):
    if len(X.shape) == shape_size:
        return X

    if X.shape[0] != 1:
        raise ValueError("__remove_prepended_dud_dims defined only when removing dud (size 1) dimensions")

    return __remove_prepended_dud_dims(shape_size, X[0])
