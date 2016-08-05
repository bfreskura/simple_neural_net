import numpy as np
from constants import *
import mnist_loader


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
    return -1 / net_output.shape[0] * np.sum(true_output * np.log(net_output))


def main():
    hidden_layer_size = 100
    no_examples_train = 2000

    # Load data
    images, labels = mnist_loader.load(TRAIN_INPUT, TRAIN_OUTPUT,
                                       no_examples_train)

    # Matrix between input and hidden layer
    # initalize weights with standard distribution
    input_hidden_weights = np.random.randn(IMAGE_SIZE ** 2,
                                           hidden_layer_size)
    hidden_output_weights = np.random.randn(hidden_layer_size, CLASSES)

    correct_predictions = 0
    step = 0
    # Start training
    for image, label in zip(images, labels):
        # Reshape inputs so they fit the net architecture
        image = np.transpose(image)

        # Feedforward
        hidden_layer = sigmoid(np.dot(image, input_hidden_weights))
        output_layer = softmax(np.dot(hidden_layer, hidden_output_weights))

        loss = log_loss(true_output=label, net_output=output_layer)

        # Measure correct predicitons
        if np.argmax(output_layer) == np.argmax(label):
            correct_predictions += 1

        # Print loss
        if step % 500 == 0:
            print('Logarithmic loss is: {:.5f}'.format(loss))
        step += 1

    print("\nAccuracy: {:.3f}".format(correct_predictions / no_examples_train))


if __name__ == "__main__":
    main()
