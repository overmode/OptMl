import numpy as np
from hw_helpers import sigmoid


def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient of the mean square error cost function
    :param y: labels
    :param tx: features
    :param w: weights
    :return: the gradient of the mse cost function
    """
    N = y.shape[0]
    e = y-(tx@w)
    return -1/N*(tx.T@e)


def compute_gradient_likelihood(y, tx, w, lamdba_=0):
    """
    Compute the gradient of the negative log likelihood cost function
    :param y: labels
    :param tx: features
    :param w: weights
    :param lamdba_: regularization
    :return: the gradient of the negative log likelihood cost function
    """
    return tx.T@(sigmoid(tx@w)-y) + lamdba_*w


def calculate_hessian(y, tx, w):
    """
    Compute the hessian of the negative log likelihood cost function
    :param y: labels
    :param tx: features
    :param w: weights
    :return: the hessian of the negative log likelihood cost function
    """

    S = np.diag((sigmoid(tx@w)*(1-sigmoid(tx@w))).T[0])

    return tx.T@S@tx


def double_pen_gradient_likelihood(y, tx, w, lambda_):
    """
    Compute the gradient of the negative log likelihood function with a double penalization when we are not able to
    predict a "1". Because the labels are unbalanced with 2 times more -1s than 1s in the training data we double
    the weights of those errors
    :param y: labels
    :param tx: features
    :param w: weights
    :param lambda_: regularization
    :return: the modified gradient
    """
    error = sigmoid(tx@w) - y
    #if error < -0.5 it means that y = 1 and we predict a "0". The error is doubled.
    error[error < -0.5] = error[error < -0.5] * 2
    gradient = tx.T@error + lambda_*w
    return gradient
