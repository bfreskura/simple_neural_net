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



def forward_pass(first_layer_input, layer_list):
    assert len(layer_list) > 0, "You can't have zero layers!"

    output_prev = layer_list[0].forward(first_layer_input)

    # Do forward pass on the rest of the layers
    for layer in layer_list[1:]:
        output_prev = layer.forward(output_prev)
    return output_prev


def backprop(sigmoid, softmax, output):
    delta_h = softmax.backprop(output)
    sigmoid.backprop(delta_h)
    sigmoid.update_weights(delta_h)
    softmax.update_weights()


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
    eval_batcher = batcher.Batcher(eval_data, BATCH_SIZE)

    # Create optimizers for each layer (necessary because every weight
    # optimizer must have unique optimizer parameters, e.g RMSProp)
    gradient_optimizer_w1 = GradientDescent()
    gradient_optimizer_b1 = GradientDescent()
    gradient_optimizer_w2 = GradientDescent()
    gradient_optimizer_b2 = GradientDescent()

    # Create network layers
    sigmoid_layer = SigmoidLayer(IMAGE_SIZE ** 2, HIDDEN_LAYER_SIZE,
                                 gradient_optimizer_w1, gradient_optimizer_b1)
    softmax_layer = SoftmaxLayer(HIDDEN_LAYER_SIZE, CLASSES,
                                 gradient_optimizer_w2,
                                 gradient_optimizer_b2)

    # The order in the layers list is VERY IMPORTANT!
    layers = [sigmoid_layer, softmax_layer]


    # Initialize helper variables
    correct_predictions = 0
    step = 0
    # Start training
    print("Training started")
    for epoch in range(EPOCHS):
        print("######### Starting epoch: ", epoch, "#########")

        for image, label in batcher_train.next_batch():
            # Reshape inputs so they fit the net architecture
            network_output = forward_pass(first_layer_input=image,
                                           layer_list=layers)

            loss = log_loss(true_output=label, net_output=network_output)
            backprop(sigmoid_layer, softmax_layer, label)

            # Measure correct predictions
            if np.argmax(network_output) == np.argmax(label):
                correct_predictions += 1

            # Print loss
            if step % 10000 == 0:
                print('Iteration {}: Batch Cross entropy loss: {:.5f}'.format(
                    step,
                    loss))
            step += BATCH_SIZE

        # # Evaluation on the test set
        correct_predictions = 0
        for image_eval, label_eval in eval_batcher.next_batch():
            # Reshape inputs so they fit the net architecture
            network_output = forward_pass(first_layer_input=image_eval,
                                           layer_list=layers)
            # Count correct predictions
            expr = np.argmax(network_output, axis=1) == np.argmax(label_eval,
                                                                 axis=1)
            correct_predictions += np.sum(expr)

        print("\nAccuracy on the test set: {:.3f}".format(
            correct_predictions / NO_EXAMPLES_TEST))


if __name__ == "__main__":
    main()
