import numpy as np


class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backprop(self, layer_next):
        raise NotImplementedError

    def _activation(self, vector):
        raise NotImplementedError


class SigmoidLayer(Layer):
    def __init__(self, input_dim, layer_next_dim, weights_opt, bias_opt):
        self.weights = np.random.randn(input_dim, layer_next_dim) * \
                       1 / np.sqrt(input_dim)
        self.bias = np.random.randn(1, layer_next_dim)
        self.weights_opt = weights_opt
        self.bias_opt = bias_opt

    def forward(self, input):
        self.input = input
        linear_comb = np.dot(input, self.weights) + self.bias
        return self._activation(linear_comb)

    def backprop(self, layer_next):
        self.dLdWinput = np.dot(self.input.T, layer_next)
        return self.dLdWinput

    def _activation(self, vector):
        """
        Applies sigmoid function element wise
        :param vector: Arbitrary matrix or a vector
        :return: Matrix/Vector of sigmoid activations
        """
        return 1 / (1 + np.exp(-vector))

    def update_weights(self, prev_activ):
        self.weights = self.weights_opt.update_weights(self.weights, self.dLdWinput)
        self.bias = self.bias_opt.update_weights(self.bias,
                                             np.sum(prev_activ, axis=0,
                                                    keepdims=True))


class SoftmaxLayer(Layer):
    def __init__(self, input_dim, layer_next_dim, weights_opt, bias_opt):
        self.weights = np.random.randn(input_dim, layer_next_dim) * \
                       1 / np.sqrt(input_dim)
        self.bias = np.random.randn(1, layer_next_dim)
        self.weights_opt = weights_opt
        self.bias_opt = bias_opt

    def forward(self, input):
        self.prev_output = input
        linear_comb = np.dot(input, self.weights) + self.bias
        self.output = self._activation(linear_comb)
        return self.output

    def backprop(self, layer_next):
        self.dLdy_ = (self.output - layer_next) / layer_next.shape[1]
        # Derivative of the error with respect to the hidden layer activations
        self.dLdhiddenActiv = np.dot(np.transpose(self.dLdy_),
                                     self.prev_output)
        delta_h = np.multiply(
            np.multiply(self.prev_output, (1 - self.prev_output)),
            (np.dot(self.dLdy_, self.weights.T)))
        return delta_h

    def _activation(self, vector):
        """
        Applies softmax function to an arbitrary matrix or vector
        https://en.wikipedia.org/wiki/Softmax_function
        :param vector: Matrix or a vector
        :return: Softmaxed Matrix/vector
        """
        return np.exp(vector) / np.sum(np.exp(vector), axis=1,
                                       keepdims=True)

    def update_weights(self):
        self.weights = self.weights_opt.update_weights(self.weights,
                                                self.dLdhiddenActiv.T)
        self.bias = self.bias_opt.update_weights(self.bias,
                                             np.sum(self.dLdy_, axis=0,
                                                    keepdims=True))
