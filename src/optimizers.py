import numpy as np


class Optimizer:
    def update_weights(self, weight_matrix, gradient):
        """
        Updates weights using the implemented optimizer
        :param: weight_matrix: Weights Matrix
        :param: gradient: Gradients matrix
        """
        raise NotImplementedError



class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update_weights(self, weight_matrix, gradient):
        weight_matrix -= self.learning_rate * gradient
        return weight_matrix


class RMSProp(Optimizer):
    """
    RMSProp optimizer WITHOUT momentum
    """
    def __init__(self, learning_rate = 0.05,
                 gamma=0.9, alpha=0.01, eps=0.00000001):
        self.lr = learning_rate
        self.r = 1
        self.gamma = gamma
        self.velocity = 0
        self.alpha = alpha
        self.eps = eps

    def update_weights(self, weight_matrix, gradient):
        self.r = self.gamma * np.square(gradient) + (1 - self.gamma) * self.r
        self.velocity = np.multiply((self.alpha / (np.sqrt(self.r) + self.eps)),
                                    gradient)
        weight_matrix -= self.lr * self.velocity

        return weight_matrix
