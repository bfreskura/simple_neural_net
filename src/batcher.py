import numpy as np
from constants import *


class Batcher:
    def __init__(self, input, batch_size):
        """
        Constructor
        :param input: Array containing images and labels data

        [[123, 255, 0, ..., 210, 0,0,0,0,1,0,0,0,0,0],
         [...]
         [...]]

         Array positions [0:784] contain image data and positions [785:794]
         contain one hot encoded image labels.
        :param batch_size: Batch size
        """
        self.data = np.array(input)
        self.batch_size = batch_size

    def next_batch(self):
        """
        Generate batch of images and labels
        :return: Images and labels batch tuple
        """
        np.random.shuffle(self.data)
        for i in range(0, len(self.data), self.batch_size):
            # Extract one batch of images and labels from the data array
            batch_data = self.data[i:i + self.batch_size]
            data_reshaped = np.split(batch_data, [IMAGE_SIZE ** 2], axis=1)

            yield data_reshaped[0], data_reshaped[1]
