import math
import numpy as np
from scipy.special import softmax
from typing import List


def BatchNorm():
    # From https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
    # TODO: Add doctring for variable names. Add momentum to init.
    def __init__(self):
        pass

    def forward(self, X, gamma, beta):
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_norm = (X - mu) / np.sqrt(var + 1e-8)
        out = gamma * X_norm + beta

        cache = (X, X_norm, mu, var, gamma, beta)

        return out, cache, mu, var

    def backward(self, dout, cache):
        X, X_norm, mu, var, gamma, beta = cache

        N, D = X.shape

        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + 1e-8)

        dX_norm = dout * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        return dX, dgamma, dbeta


class RNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, batch_size=1) -> None:
        """

        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = output_dim
        self.batch_size = batch_size
        # initialization
        self.params = self._init_params()
        self.hidden_state = self._init_hidden_state()

    def _init_params(self) -> List[np.array]:
        """
        """
        scale = 0.01
        Waa = np.random.normal(scale=scale, size=[self.hidden_dim, self.hidden_dim])
        Wax = np.random.normal(scale=scale, size=[self.hidden_dim, self.input_dim])
        Wy = np.random.normal(scale=scale, size=[self.out_dim, self.hidden_dim])
        ba = np.zeros(shape=self.hidden_dim)
        by = np.zeros(shape=self.out_dim)
        return [Waa, Wax, Wy, ba, by]

    def _init_hidden_state(self) -> np.array:
        return np.zeros(shape=[self.hidden_dim, self.batch_size])

    def forward(self, input_vector: List[np.array]) -> List[np.array]:
        """
        Input vector:
            dimension: [num_steps, self.input_dim, self.batch_size]
        """
        Waa, Wax, Wy, ba, by = self.params
        output_vector = []
        for vector in input_vector:
            self.hidden_state = np.tanh(
                np.dot(Waa, self.hidden_state) + np.dot(Wax, vector) + ba
            )
            y = softmax(
                np.dot(Wy, self.hidden_state) + by
            )
            output_vector.append(y)
        return output_vector


def Adagrad(data):
    pass


def Adam(data):
    pass


def LBFGS(data):
    pass


def RMSProp(data):
    pass


# def SGD(data, batch_size, lr):
#     N = len(data)
#     np.random.shuffle(data)
#     mini_batches = np.array([data[i:i+batch_size]
#      for i in range(0, N, batch_size)])
#     for X,y in mini_batches:
#         backprop(X, y, lr)


def SGD_Momentum():
    pass


if __name__ == "__main__":
    input_data = np.array([
        [
            [1, 3]
            , [2, 4]
            , [3, 6]
        ]
        , [
            [4, 3]
            , [3, 4]
            , [1, 5]
        ]
    ])
    batch_size = 2
    input_dim = 3
    output_dim = 4
    hidden_dim = 5
    time_step = 2
    rnn = RNN(input_dim=input_dim, batch_size=batch_size, output_dim=output_dim, hidden_dim=hidden_dim)
    output_vector = rnn.forward(input_vector=input_data)
    print(output_vector)
