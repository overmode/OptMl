import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def sigmoid(t):
    """
    Logistic fonction that returns value between 0 and 1

    :param t: is the prediction
    :return: the probability
    """
    return 1/(1 + np.exp(-t))


def build_poly(x, degree):
    """
    Augment the vector x to the polynomial basis of degree "degree"

    :param x: the vector to augment
    :param degree: the degree of the polynomial basis
    :return: the augmented vector
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    #poly_matrix = np.zeros((x.shape[0],degree+1))
    #deg = np.arange(degree+1)
    #for i in range(degree+1):
    #    poly_matrix[:,i] = x

    return np.vander(x, degree, increasing=True)


def absolute_error(y, y_pd):
    """
    Calculate the accuracy score
    :param y: true labels
    :param y_pd: predicted labels
    :return: the percentage of good prediction
    """
    n=len(y)
    return (np.equal(y, y_pd).astype(int).sum())/n
