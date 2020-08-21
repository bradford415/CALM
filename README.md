# Multiclassification
A multilayer perceptron neural network used to classify three or more labels through gene expression implemented in PyTorch. 

This project takes two input files, a gene expression matrix (GEM) and text file which lists the correct labels in one column. The GEM is transposed to allign to with the label file, 80% of the samples are used to train the neural network, and then the network is tested with the last 20% of samples.

## Requirements
- PyTorch
- 

## Setup
Because this project is meant to be run on the Palmetto Cluster, an Anaconda Environment will need to be set up with the required dependencies installed.

### Installation
Naviagte to the desired directory and clone the repository as shown belown. On the palmetto cluster, you will probably need to add the git module first.

```module add git/2.27.0-gcc/8.3.1```
```git clone 

### Anaconda Environment


