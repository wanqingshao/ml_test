
import numpy as np


class Perceptron(object):

    """
    testing the implementation of perceptron

    input parameters:
    X feature values, each row is a sample, each col is a feature
    y available values as 0 or 1
    """
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.niter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # create random state
        rgen = np.random.RandomState(self.random_state)
        # initiate random weights
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.niter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[0]  += update
                self.w_[1:] += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
