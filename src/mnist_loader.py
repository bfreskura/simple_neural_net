import numpy as np
from constants import *


def load(imgf, labelf, n):
    """
    :param imgf: Images train
    :param labelf: Labels train
    :param n: Number of examples to load
    :return: Image inputs
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
    :return: One hot encoded label
    """
    array = np.zeros(no_classes)
    array[label] = 1
    return array
