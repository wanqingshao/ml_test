import numpy as np

class AdalineSGD(object):

    def __init__(self, eta, n_iter, random_state, shuffle = True):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >=0, 1, -1)

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, scale= 0.01, size = m +1)
        self.w_initialized = True

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _update_weight(self, X, y):
        net_input = self.net_input(X)
        prediction = self.activation(net_input)
        self.w_[1:] += self.eta * X.T.dot(y - prediction)
        self.w_[0] += self.eta * ((y - prediction).sum())
        cost = ((y - prediction) ** 2).sum() / 2
        return cost

    def partital_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if (y.ravel().shape[0]) >1:
            for xi, target in zip(X, y):
                self._update_weight(xi, target)
        else:
            self._update_weight(X, y)
        return self

    def fit(self, X, y):
        self.cost_ = []
        self._initialize_weights(X.shape[1])
        for i in range(self.n_iter):
            if self.shuffle:
                self._shuffle(X, y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weight(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self