# Simple neural network
Implementation of a simple feedforward neural net with one hidden layer.


## Dataset
I've used the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

## Results
### Results with batch size = 1, no SGD

Epochs | Hidden Layers | Precision on the Test set [%] | Learning Rate
------ | ------------- | ----------------------------- | -------------
8      | 200           | 94.3                          | 0.001

### Results with SGD

Epochs | Hidden Layers | Precision on the Test set [%] | Learning Rate | Minibatch size
------ | ------------- | ----------------------------- | ------------- | --------------
30     | 100           | 95.3                          | 0.002         | 200
50     | 100           | 95.8                          | 0.002         | 200
100    | 100           | 96.4                          | 0.0009        | 200
100    | 200           | 96.5                          | 0.0009        | 300

## How to use
