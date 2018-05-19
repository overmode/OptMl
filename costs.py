import numpy as np
from hw_helpers import sigmoid


def compute_mse(y, tx, w):
    """
    Compute the mean square error.
    :param y: labels
    :param tx: features
    :param w: weights
    :return: the mean square error
    """
    e = y - tx.dot(w)
    mse = e.T.dot(e) / (2 * len(e))
    return mse


def calculate_loss(y, tx, w, lambda_=0):
    """
    Compute the negative log likelihood
    :param y: labels
    :param tx: features
    :param w: weights
    :param lambda_: regularization
    :return: the negative log likelihood
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    return (((-1)*(y*np.log(sigmoid(tx@w))+(1-y)*np.log(1-sigmoid(tx@w)))) + lambda_/2*w.T@w).mean()


