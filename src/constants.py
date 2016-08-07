import os

PROJ_ROOT = os.path.dirname(__file__)
RSRC = os.path.join(PROJ_ROOT, "resources")

TRAIN_INPUT = os.path.join(RSRC, "train-images.idx3-ubyte")
TRAIN_OUTPUT = os.path.join(RSRC, "train-labels.idx1-ubyte")
EVAL_INPUT = os.path.join(RSRC, "t10k-images.idx3-ubyte")
EVAL_OUTPUT = os.path.join(RSRC, "t10k-labels.idx1-ubyte")

CLASSES = 10
IMAGE_SIZE = 28

EPOCHS = 100
BATCH_SIZE = 300  # Must be an even number
LEARNING_RATE = 0.0009

# Network architecture

HIDDEN_LAYER_SIZE = 100
NO_EXAMPLES_TRAIN = 60000  # Can be in range [0,60000]
NO_EXAMPLES_TEST = 10000  # Can be in range [0,10000]
