import numpy as np


class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backprop(self):
        raise NotImplementedError

    def _activation(self, vector):
        raise NotImplementedError


class SigmoidLayer(Layer):
    def __init__(self, input_dim, layer_next_dim):
        self.weights = np.random.randn(input_dim, layer_next_dim) * \
                       1 / np.sqrt(input_dim)
        self.bias = np.random.randn(1, layer_next_dim)

    def forward(self, input):
        linear_comb = np.dot(input, self.weights) + self.bias
        return self._activation(linear_comb)

    def backprop(self):
        pass

    def _activation(self, vector):
        """
        Applies sigmoid function element wise
        :param vector: Arbitrary matrix or a vector
        :return: Matrix/Vector of sigmoid activations
        """
        return 1 / (1 + np.exp(-vector))


class SoftmaxLayer(Layer):
    def __init__(self, input_dim, layer_next_dim):
        self.weights = np.random.randn(input_dim, layer_next_dim) * \
                       1 / np.sqrt(input_dim)
        self.bias = np.random.randn(1, layer_next_dim)

    def forward(self, input):
        linear_comb = np.dot(input, self.weights) + self.bias
        return self._activation(linear_comb)

    def backprop(self):
        pass

    def _activation(self, vector):
        """
        Applies softmax function to an arbitrary matrix or vector
        https://en.wikipedia.org/wiki/Softmax_function
        :param vector: Matrix or a vector
        :return: Softmaxed Matrix/vector
        """
        return np.exp(vector) / np.sum(np.exp(vector), axis=1, keepdims=True)
