import numpy as np
from constants import *


class Batcher:
    def __init__(self, input, batch_size):
        self.data = np.array(input)
        self.batch_size = batch_size

    def next_batch(self):
        """
        Generate batch of items
        :return: Images and labels batch tuple
        """
        np.random.shuffle(self.data)
        for i in range(0, len(self.data), self.batch_size):
            # Extract image data and label from the data array
            batch_data = self.data[i:i + self.batch_size]
            data_reshaped = np.split(batch_data, [IMAGE_SIZE ** 2], axis=1)

            yield data_reshaped[0], data_reshaped[1]
