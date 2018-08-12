# EigenNet
C++ Implementation of a Neural Network with Eigen3

A modular implementation of a vanilla feedforward neural network implemented in C++ with the Eigen3 library for matrix operations. 

### Installing Eigen3
MacOS: `$ brew install eigen`
Windows: 
Linux: `$ sudo apt-get install libeigen3-dev`.

### Usage 
The `main()` function in neuralnet.cpp is a walkthrough of how to build a neural network: 
1. Obtain and parse the training and test data. Both the samples and the labels should have the type MatrixXd. If the data has n training features, k label features, and m samples, then the dimensions of X and Y are `m x n` and `m x k` respectively.
2. Specify the hidden layer sizes in the form of a `vector <int>`. The activation functions for each layer also need to be specified. If you wish to create an activation function then you have to create two functions satisfying the dFunc typedef: the function itself and the derivative of the function. You can then use the ActivationFunc wrapper around them to create an object that the neural network can use. Four activation functions: Relu, LeakyRelu, Sigmoid and Tanh have already been created in the `main()` function for your convenience. 
3. Create the network object by specifying the training data, labels, layer sizes, activation functions and the learning rate.  
4. Use the `run` method to train the network and specify the number of epochs and minibatch size. 
5. Use the `predict` method to calculate the classification accuracy.

### Miscellany
Currently, the loss function cannot be changed and the network simply uses MSE. Also, the `predict` function is only made for logistic regression and only outputs classification error. 




