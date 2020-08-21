# Multiclassification
A multilayer perceptron neural network used to classify three or more labels through gene expression implemented in PyTorch. 

This project takes two input files, a gene expression matrix (GEM) and text file which lists the correct labels in one column. The GEM is transposed to allign with the label file, 80% of the samples are used to train the neural network, and then the network is tested with the last 20% of samples.

## Requirements
- pytorch
- pandas

## Setup
Because this project is meant to be run on the Palmetto Cluster, an Anaconda Environment will need to be set up with the required dependencies installed.

### Installation
Naviagte to the desired directory and clone the repository as shown belown. On the palmetto cluster, the git module will probably need to be added first.

```
module add git/2.27.0-gcc/8.3.1
git clone https://github.com/bradford415/multiclassification.git
```

### Anaconda Environment
Create a virtual environment with the required dependecies. If a virtual envrionment already exists and meets these requirements, this step can be skipped.
```
module add anaconda3/5.10-gcc/8.3.1
```
```
conda create -n myenv python=3.7 pytorch pandas
```

