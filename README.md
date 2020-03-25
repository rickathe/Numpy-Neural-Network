# NumPy-Based Neural Network


This is a feed-forward neural network written with Python and NumPy. The goal of this project was to learn the theory behind neural networks and to build a network from scratch using only basic modules.

**Features:**
- Trains the network on a given dataset and runs validation, plotting error over epochs.
- Allows for the use of 1 to 5 hidden layers. 
- Normalizes data.
- Automatically breaks as over-training is detected.
- Includes options to test hyper-parameters, running up to 5 variations of a particular hyper-parameter and displaying the final error of each in a histogram with error bars.
- Runs multiple trials for each hyper-parameter and takes the mean and standard deviation of the results.

&nbsp;
&nbsp;

## Visualizing Data
&nbsp;

**Training/Validation Plot From a 5-Layer Network:**
<p align="center">
  <img width="560" height="400" src="https://github.com/rickathe/Numpy_Neural_Network/blob/master/Plots/multiply_10k_1k_50h_001lr_5layer_test5.png">
</p>

&nbsp;

**Histogram to Test Error vs. Hyper-Parameter Chosen:**
<p align="center">
  <img width="560" height="400" src="https://github.com/rickathe/Numpy_Neural_Network/blob/master/Plots/bar_10k1k_var_01lr_5layer_test1.png">
</p>

&nbsp;
&nbsp;

## How to Use
Apart from NumPy, all that is required to run this neural network is the two files mentioned below:
- neural_net: module containing the neural network; supports 1-5 hidden layers.
- neural_net_test: script which sets the hyper-parameters of the neural_net (epochs, number of nodes, learning rate, etc), runs and stores the results of the network, and creates visualizations of the data. 

&nbsp;
&nbsp;


## Acknowledgements
The original inspiration for this project was based on Tariq Rashid's excellent book, __Make Your Own Neural Network__, the code of which can be found [here](https://github.com/makeyourownneuralnetwork).
