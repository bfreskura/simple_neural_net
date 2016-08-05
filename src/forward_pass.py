import numpy as np

picture_dimension = 28
hidden_layer_size = 300
no_labels = 10


def sigmoid(vector):
    """
    Applies sigmoid function elementwise
    :param vector: Arbitrary matrix or a vector
    :return:
    """
    return 1 / (1 + np.exp(-vector))


def softmax(vector):
    """
    Applies softmax function to a arbitrary matrix or vector
    :param vector: Matrix or a vector
    :return:
    """
    return np.exp(vector) / np.sum(np.exp(vector))


def log_loss(true_output, net_output):
    """
    Calculates logarithmic loss across all output classes
    https://en.wikipedia.org/wiki/Cross_entropy
    :param true_output: Real outputs
    :param net_output: Network outputs
    :return:
    """
    return -1 / net_output.shape[1] * np.sum(true_output * np.log(net_output))


# initalize all matrices

input_vector = np.ones([1, picture_dimension ** 2])

real_outputs = np.zeros([1, no_labels])
# Real outputs are one-hot encoded
# [0,0,0,1,0,0,0,0]
real_outputs[0][2] = 1

# Matrix between input and hidden layer
# initalize weights with standard distribution
input_hidden_weights = np.random.randn(picture_dimension ** 2,
                                       hidden_layer_size)

hidden_output_weights = np.random.randn(hidden_layer_size, no_labels)

# Feedforward
hidden_layer = sigmoid(np.dot(input_vector, input_hidden_weights))
output_layer = softmax(np.dot(hidden_layer, hidden_output_weights))

loss = log_loss(true_output=real_outputs, net_output=output_layer)

print('Logarithmic loss is: {:.5f}'.format(loss))
