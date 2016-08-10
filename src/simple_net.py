import batcher
import mnist_loader
from Layer import *
from constants import *
from optimizers import *


def sigmoid(vector):
    """
    Applies sigmoid function element wise
    :param vector: Arbitrary matrix or a vector
    :return: Matrix/Vector of sigmoid activations
    """
    return 1 / (1 + np.exp(-vector))


def softmax(vector):
    """
    Applies softmax function to an arbitrary matrix or vector
    https://en.wikipedia.org/wiki/Softmax_function
    :param vector: Matrix or a vector
    :return: Softmaxed Matrix/vector
    """
    return np.exp(vector) / np.sum(np.exp(vector), axis=1, keepdims=True)


def log_loss(true_output, net_output):
    """
    Calculates logarithmic loss across all output classes.
    Takes the mean loss across all batch elements.
    https://en.wikipedia.org/wiki/Cross_entropy

    :param true_output: Real outputs
    :param net_output: Network outputs
    :return: Batch loss mean
    """
    sum = - np.sum((np.multiply(true_output, np.log(net_output))) / (
        net_output.shape[1]), axis=1)
    return np.mean(sum)


def backprop(y, y_, hidden_output_activations, hidden_weights,
             input_weights, bias_hidden, bias_input, input_x, optimizer):
    """
    Runs the backpropagation algorithm.
    :param y: True outputs
    :param y_: Network outputs after the softmax transformation
    :param hidden_output_activations: Output of hidden layer after the applied activation
    :param hidden_weights: Weight matrix between the hidden and the output layer
    :param input_weights: Weight matrix between the input and the hidden layer
    :param bias_hidden: Bias Weight matrix between the hidden and the output layer
    :param bias_input: Bias Weight matrix between the input and the hidden layer
    :param input_x: Network input (images)
    :return: Updated weight matrices
    """
    # Derivative with respect to output (cross entropy + softmax)
    # https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function#945918
    dLdy_ = (y_ - y) / y.shape[1]

    # Derivative of the error with respect to the hidden layer activations
    dLdhiddenActiv = np.dot(np.transpose(dLdy_), hidden_output_activations)

    delta_h = np.multiply(
        np.multiply(hidden_output_activations, (1 - hidden_output_activations)),
        (np.dot(dLdy_, hidden_weights.T)))

    dLdWinput = np.dot(input_x.T, delta_h)

    # Update weights
    hidden_weights = optimizer.update_weights(hidden_weights, dLdhiddenActiv.T)
    bias_hidden = optimizer.update_weights(bias_hidden,
                                           np.sum(dLdy_, axis=0, keepdims=True))
    input_weights = optimizer.update_weights(input_weights, dLdWinput)
    bias_input = optimizer.update_weights(bias_input,
                                          np.sum(delta_h, axis=0,
                                                 keepdims=True))

    return hidden_weights, bias_hidden, input_weights, bias_input


def forward_pass(input, input_hidden_weight, bias_input, hidden_output_weight,
                 bias_hidden):
    """
    Calculates the forward pass of the network
    :param input: Network input (images)
    :param input_hidden_weight: Weight matrix between the input and the hidden layer
    :param bias_input: Bias matrix between the input and the hidden layer
    :param hidden_output_weight: Weight matrix between the hidden and the output layer
    :param bias_hidden: Bias matrix between the hidden and the output layer
    :return: Softmax outputs for the current batch
    """
    hidden_layer = np.dot(input, input_hidden_weight) + bias_input
    hidden_activations = sigmoid(hidden_layer)
    return softmax(np.dot(hidden_activations,
                          hidden_output_weight) + bias_hidden), hidden_activations


def forward_pass2(first_layer_input, layer_list):
    assert len(layer_list) > 0, "You can't have zero layers!"

    output_prev = layer_list[0].forward(first_layer_input)

    # Do forward pass on the rest of the layers
    for layer in layer_list[1:]:
        output_prev = layer.forward(output_prev)
    return output_prev


def backprop2(sigmoid, softmax, output, optimizer):
    delta_h = softmax.backprop(output)
    dLdWinput = sigmoid.backprop(delta_h)

    sigmoid.update_weights(optimizer, delta_h)
    softmax.update_weights(optimizer)


def main():
    # Assert constants
    assert BATCH_SIZE % 2 == 0, "Must be devisable by 2"
    assert 0 <= NO_EXAMPLES_TEST <= 10000, "Must be in range [0, 10000]"
    assert 0 <= NO_EXAMPLES_TRAIN <= 60000, "Must be in range [0,60000]"

    mnist_loader.download_mnist_files()

    # Load training data
    train_data = mnist_loader.load(TRAIN_INPUT, TRAIN_OUTPUT,
                                   NO_EXAMPLES_TRAIN)
    eval_data = mnist_loader.load(EVAL_INPUT, EVAL_OUTPUT,
                                  NO_EXAMPLES_TEST)

    # Create batchers for data batching
    batcher_train = batcher.Batcher(train_data, BATCH_SIZE)
    eval_batcher = batcher.Batcher(eval_data, 1)

    # Create network layers
    sigmoid_layer = SigmoidLayer(IMAGE_SIZE ** 2, HIDDEN_LAYER_SIZE)
    softmax_layer = SoftmaxLayer(HIDDEN_LAYER_SIZE, CLASSES)

    # The order in the layers list is VERY IMPORTANT!
    layers = [sigmoid_layer, softmax_layer]

    # Create an optimizer
    gradient_optimizer = GradientDescent(LEARNING_RATE)

    # Initialize helper variables
    correct_predictions = 0
    step = 0
    # Start training
    print("Training started")
    for epoch in range(EPOCHS):
        print("######### Starting epoch: ", epoch, "#########")

        for image, label in batcher_train.next_batch():
            # Reshape inputs so they fit the net architecture
            network_output = forward_pass2(first_layer_input=image,
                                           layer_list=layers)

            loss = log_loss(true_output=label, net_output=network_output)
            backprop2(sigmoid_layer, softmax_layer, label, gradient_optimizer)
            # Measure correct predicitons

            if np.argmax(network_output) == np.argmax(label):
                correct_predictions += 1

            # Print loss
            if step % 10000 == 0:
                print('Iteration {}: Batch Cross entropy loss: {:.5f}'.format(
                    step,
                    loss))
            # Train set evaluation
            step += BATCH_SIZE

        # # Evaluation on the test set
        correct_predictions = 0
        # TODO create batching
        for image_eval, label_eval in eval_batcher.next_batch():
            # Reshape inputs so they fit the net architecture
            network_output = forward_pass2(first_layer_input=image_eval,
                                           layer_list=layers)
            # Count correct predictions
            if np.argmax(network_output) == np.argmax(label_eval):
                correct_predictions += 1

        print("\nAccuracy on the test set: {:.3f}".format(
            correct_predictions / NO_EXAMPLES_TEST))


if __name__ == "__main__":
    main()
