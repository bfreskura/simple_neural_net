import os

PROJ_ROOT = os.path.dirname(__file__)
RSRC = os.path.join(PROJ_ROOT, "resources")

TRAIN_INPUT = os.path.join(RSRC, "train-images.idx3-ubyte")
TRAIN_OUTPUT = os.path.join(RSRC, "train-labels.idx1-ubyte")
EVAL_INPUT = os.path.join(RSRC, "t10k-images.idx3-ubyte")
EVAL_OUTPUT = os.path.join(RSRC, "t10k-labels.idx1-ubyte")

CLASSES = 10
IMAGE_SIZE = 28
