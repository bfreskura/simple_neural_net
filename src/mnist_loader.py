import numpy as np
from constants import *


def load(imgf, labelf, n):
    """
    Converts MNIST binary data to a list of numpy arrays containing image
    and one-hot encoded label data.

    Output list is in the following format:
        [[123, 255, 0, ..., 210, 0,0,0,0,1,0,0,0,0,0],
         [9, 255, 3, ..., 10, 0,0,1,0,0,0,0,0,0,0],
         [153, 235, 32, ..., 110, 1,0,0,0,0,0,0,0,0,0],
         [...]]

    Array positions [0:784] contain image data and positions [785:794]
         contain one hot encoded image labels.

    :param imgf: Images train
    :param labelf: Labels train
    :param n: Number of examples to load
    :return: Array of image + label data
    """
    f = open(imgf, "rb")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = []
        label = to_one_hot(ord(l.read(1)), CLASSES)
        for j in range(IMAGE_SIZE ** 2):
            image.append(ord(f.read(1)))

        # Concatinate label and image
        data = np.concatenate([np.array(image), label])
        images.append(data)
    return images


def to_one_hot(label, no_classes):
    """
    One-hot encodes the labels
    :param label: Label
    :param no_classes: Number of existing classes
    :return: One-hot encoded label
    """
    array = np.zeros(no_classes)
    array[label] = 1
    return array
