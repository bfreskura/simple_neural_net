import os

PROJ_ROOT = os.path.dirname(__file__)
RSRC = os.path.join(PROJ_ROOT, "resources")

TRAIN_INPUT = os.path.join(RSRC, "train-images.idx3-ubyte")
TRAIN_OUTPUT = os.path.join(RSRC, "train-labels.idx1-ubyte")
EVAL_INPUT = os.path.join(RSRC, "t10k-images.idx3-ubyte")
EVAL_OUTPUT = os.path.join(RSRC, "t10k-labels.idx1-ubyte")

# Number of classes (digits)
CLASSES = 10
# Image width and height (Images are 28x28)
IMAGE_SIZE = 28

EPOCHS = 100
BATCH_SIZE = 2  # Must be an even number
LEARNING_RATE = 0.0009

# Network architecture

HIDDEN_LAYER_SIZE = 100
NO_EXAMPLES_TRAIN = 100  # MUST be in range [0,60000]
NO_EXAMPLES_TEST = 100  # MUST be in range [0,10000]
