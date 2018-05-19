import numpy as np
from costs import *
from gradients import *
from hw_helpers import *



def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Perform a gradient descent using the least-squares method(MSE cost function)
    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: maximum number of iteration
    :param gamma:  step-size
    :return: the optimized weights
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    loss = float('inf')
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y,tx,w)
        print(gradient)
        loss = compute_mse(y,tx,w)
        print(loss)
        w = w - gamma*gradient
        ws.append(w)
        print(w)
        losses.append(loss)

    return w, loss


def least_squares_SDG(y, tx, initial_w, max_iters, gamma):
    """
    Perfom a stochastic gradient descent using the least squares method(MSE cost function)
    :param y: labels
    :param tx: features
    :param initial_w: weights
    :param max_iters: maximum number of iteration
    :param gamma: step-size
    :return: the optimized weights
    """
    losses= []
    w = initial_w
    ws = [initial_w]
    losses = []
    loss = float('inf')
    batch_size = 1

    for i in range(max_iters):
        for minbatch_y, minbatch_x in batch_iter(y, tx, batch_size):
            gradient = compute_gradient_mse(minbatch_y, minbatch_x, w)
            #SGD using mse
            loss = compute_mse(minbatch_y, minbatch_x, w)
            w = w-gamma*gradient
            ws.append(w)
            losses.append(loss)
    #returns only the final choice
    return w, loss


def least_squares(y,tx):
    """
    Return the optimized w using the normal equation of the least-squares method(MSE cost function)
    :param y: labels
    :param tx: features
    :return: the optimized weights
    """
    if np.linalg.matrix_rank(tx) == tx.shape[1]:
        w = np.linalg.inv(tx.T@tx)@tx.T@y
        ls = compute_mse(y, tx, w)
        return w, ls
    else:
        w = np.linalg.pinv(tx)@y
        ls = compute_mse(y, tx, w)
        return w, ls


def ridge_regression(y, tx, lambda_):
    """
    Perform the ridge regression

    :param y: labels
    :param tx: features
    :param lambda_: regularization
    :return: the optimized w + the result of the minimum square error cost function for the optimized w
    """
    w = np.linalg.inv(tx.T@tx + (lambda_*2.0*float(tx.shape[0]))*np.identity(tx.shape[1]))@(tx.T@y)
    #w = np.linalg.inv(tx@tx.T + lambda_*np.identity(tx.shape[1]))@tx@y
    e = compute_mse(y, tx, w)
    return w, e


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform the logistic regression using a gradient descent
    :param y: labels
    :param tx: features
    :param initial_w: initial weight vectors
    :param max_iters: maximum number of iteration
    :param gamma: step-size
    :return: the optimized weights and the result of the negative log likelihood cost function
    """
    w = initial_w
    losses = []
    threshold = 1e-8
    for i in range(max_iters):
        g = compute_gradient_likelihood(y, tx, w)
        w = w-gamma*g
        loss = calculate_loss(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2] < threshold):
            break
    loss = calculate_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform the regularized logistic regression using a gradient descent
    :param y: labels
    :param tx: features
    :param lambda_: regularization hyper parameter
    :param initial_w: initial weights
    :param max_iters: maximum number of iteration
    :param gamma: step-size
    :return:
    """
    w = initial_w
    losses = []
    threshold = 1e-8
    for i in range(max_iters):
        #loss = ((-1)*(2*y*np.log(sigmoid(tx@w))+(1-y)*np.log(1-sigmoid(tx@w)))).sum()
        #loss, g = penalized_logistic_regression(y, tx, w, lambda_)
        loss = calculate_loss(y, tx, w, lambda_)
        g = compute_gradient_likelihood(y, tx, w, lambda_)
        #loss = calculate_loss(y, tx, w)
        losses.append(loss)
        #g = compute_gradient_likelihood(y, tx, w) - lambda_ * w
        #print(loss)
        w = w - gamma*g
        if len(losses) > 1 and np.abs(losses[-1]- losses[-2] < threshold):
            break

    return w, loss


def new_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform the regularized logistic regression using the newton's method
    :param y: labels
    :param tx: features
    :param lambda_: regularization
    :param initial_w: initial weights
    :param max_iters: maximum number of iteration
    :param gamma: step-size
    :return:
    """
    losses = []
    w = initial_w
    threshold = 1e-8
    for i in range(max_iters):
        loss = calculate_loss(y, tx, w)
        losses.append(loss)
        g = compute_gradient_likelihood(y, tx, w) + lambda_ * w
        hes = calculate_hessian(y, tx, w)
        w = w-gamma*np.linalg.inv(hes)@g
        if len(losses) > 1 and np.abs(losses[-1]- losses[-2] < threshold):
            break
    loss = calculate_loss(y, tx, w)
    return w,loss


def double_pen_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    """
    Perform the regularized logistic regression using gradient descent and double error for false detection of 0
    :param y: labels
    :param tx: features
    :param lambda_: regularization
    :param initial_w: initial weights
    :param max_iters: maximum number of iteration
    :param gamma: step-size
    :return:
    """
    losses = []
    w = initial_w
    threshold = 1e-8
    for i in range(max_iters):
        loss = calculate_loss(y, tx, w, lambda_)
        losses.append(loss)
        g = double_pen_gradient_likelihood(y, tx, w, lambda_)
        w = w-gamma*g
        #if len(losses) > 1 and np.abs(losses[-1]- losses[-2] < threshold):
        #    break
    loss = calculate_loss(y, tx, w, lambda_)
    return w, loss


