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
    return - np.sum(true_output * np.log(net_output)) / net_output.shape[0]


def backprop(y, y_, hidden_output_activations, hidden_weights,
             input_weights, bias_hidden, bias_input, input_x):
    # Derivative with respect to output (cross entropy + softmax)
    # https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function#945918
    dLdy_ = (y_ - y) / y.shape[0]

    # Derivative of the error with respect to the hidden layer activations
    dLdhiddenActiv = np.dot(np.transpose(dLdy_), (hidden_output_activations))

    delta_h = np.multiply(
        np.multiply(hidden_output_activations, (1 - hidden_output_activations)),
        (np.dot(dLdy_, hidden_weights.T)))

    dLdWinput = np.dot(input_x.T, delta_h)

    learning_rate = 0.001
    # Update hidden layers weights
    hidden_weights -= learning_rate * np.transpose(dLdhiddenActiv)
    bias_hidden -= learning_rate * dLdy_

    input_weights -= learning_rate * dLdWinput
    bias_input -= learning_rate * delta_h

    return hidden_weights, bias_hidden, input_weights, bias_input


def forward_pass(input, input_hidden_weight, bias_input, hidden_output_weight,
                 bias_hidden):
    # Feedforward
    hidden_layer = np.dot(input, input_hidden_weight) + bias_input
    hidden_activations = sigmoid(hidden_layer)
    return softmax(
        np.dot(hidden_activations,
               hidden_output_weight) + bias_hidden), hidden_activations


def main():
    hidden_layer_size = 100
    no_examples_train = 60000
    no_examples_test = 10000

    # Load training data
    images, labels = mnist_loader.load(TRAIN_INPUT, TRAIN_OUTPUT,
                                       no_examples_train)
    images_eval, labels_eval = mnist_loader.load(EVAL_INPUT, EVAL_OUTPUT,
                                                 no_examples_test)

    # Matrix between input and hidden layer
    # initialize weights with standard distribution / number of inputs
    # Input -> hidden layer
    input_hidden_weights = np.random.randn(IMAGE_SIZE ** 2,
                                           hidden_layer_size) / np.sqrt(
        IMAGE_SIZE ** 2)
    bias_input_hidden = np.random.randn(1, hidden_layer_size)

    # Hidden layer -> output layer
    hidden_output_weights = np.random.randn(hidden_layer_size,
                                            CLASSES) / np.sqrt(
        hidden_layer_size)
    bias_output_hidden = np.random.randn(1, CLASSES)

    # Initialize helper variables
    correct_predictions = 0
    step = 0
    # Start training
    for image, label in zip(images, labels):
        # Reshape inputs so they fit the net architecture
        image = np.transpose(image)
        image.resize(1, IMAGE_SIZE ** 2)

        output_layer, hidden_activations = forward_pass(input=image,
                                                        input_hidden_weight=input_hidden_weights,
                                                        bias_input=bias_input_hidden,
                                                        hidden_output_weight=hidden_output_weights,
                                                        bias_hidden=bias_output_hidden)

        loss = log_loss(true_output=label, net_output=output_layer)

        # Do the backprop
        hidden_output_weights, bias_output_hidden, input_hidden_weights, bias_input_hidden = backprop(
            y=label,
            y_=output_layer,
            hidden_output_activations=hidden_activations,
            hidden_weights=hidden_output_weights,
            input_weights=input_hidden_weights,
            bias_hidden=bias_output_hidden,
            bias_input=bias_input_hidden,
            input_x=image)

        # Measure correct predicitons
        if np.argmax(output_layer) == np.argmax(label):
            correct_predictions += 1

        # Print loss
        if step % 500 == 0:
            print('Step {}: Logarithmic loss is: {:.5f}'.format(step, loss))
        # Train set evaluation
        if step % 5000 == 0:
            print("\nAccuracy on a training set: {:.3f}".format(
                correct_predictions / no_examples_train))
        step += 1

    # Evaluation on the test set
    correct_predictions = 0
    for image, label in zip(images_eval, labels_eval):
        # Reshape inputs so they fit the net architecture
        image = np.transpose(image)
        image.resize(1, 784)

        output_layer, hidden_activations = forward_pass(input=image,
                                                        input_hidden_weight=input_hidden_weights,
                                                        bias_input=bias_input_hidden,
                                                        hidden_output_weight=hidden_output_weights,
                                                        bias_hidden=bias_output_hidden)
        # Measure correct predicitons
        if np.argmax(output_layer) == np.argmax(label):
            correct_predictions += 1

    print("\nAccuracy on the test set: {:.3f}".format(
        correct_predictions / no_examples_test))


if __name__ == "__main__":
    main()
