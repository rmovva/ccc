# mnist-numpy
A basic fully connected network implemented purely in NumPy and trained on the MNIST dataset.

## Mean-only BatchNorm implementation

We implement mean-only batch normalization, in which the batch-wise mean of an input x is subtracted to compute the output y = BN(x). This computation is implemented as a layer module in layers.py, and layers are added to the defined network in main.py. We demonstrate correctness of the forward and backward pass of this layer in our two test files.

We make a couple design choices, notably keeping track of our dataset-wide "running mean" by using a momentum of 0.1. This running mean is used at inference time in place of our batch-wise means. We also include trainable parameters "beta" for each channel in our input x, allowing for a constant shift of our output y (and thereby allowing the BatchNorm layer to e.g. learn the identity function).

We test our BatchNorm layer using pytest. We write two forward pass test cases with zero and nonzero beta. We test the layer's ability to correct propagate gradients backward by comparing the analytic gradients dL/dX and dL/dbeta to the expected, manually calculated gradients. We also compare the analytic gradient dL/dX to the numerical gradient in test_numericalgrad.py.

## Experiments
The MNIST dataset is split into 50000 train, 10000 validation and 10000 test samples. All splits are normalized using the statistics of the training split (using the global mean and standard deviation, not per pixel).

The network has 2 fully connected layers with ReLU activations. The first hidden layer has 256 units and the second 128 units. The network is initialized with Xavier-He initialization.

The network is trained for 250 epochs with vanilla minibatch SGD and learning rate 1e-3. The final accuracy on the test set is about 0.97.


## Code structure:
### layers.py
Contains classes that represent layers for different transformations. Each class has a forward and a backward method that define a transformation and its gradient. The class keeps track of the variables defining the transformation and the variables needed to calculate the gradient. The file also contains a class that defines the softmax cross entropy loss.

### network.py
Defines Network, a configurable class representing a sequential neural network with any combination of layers. Network has a train function that performs minibatch SGD.

### main.py
Data loading, training and validation scripts. Running it trains the networks described in experiments. For loading the data it expects two files "data/mnist_train.csv" and "data/mnist_test.csv". These can be downloaded from https://pjreddie.com/projects/mnist-in-csv/. To run use "python3 main.py".
