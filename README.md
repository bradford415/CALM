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

conda create -n myenv python=3.7 pytorch pandas
```

### PBS Script
In the lung_GEM_NN.pbs script, the virtual env name will need to match virtual env being used. The default name is 'myenv'. Change the line below if neccessary.

```
source activate myenv
```
### Input Files
Move the GEM file and labels file into the 'input' directory. Currently, the GEM file must be named 'lung.emx.txt' and the label file must be named 'lung_sample_condition_no_sample_names.txt'

## Running
To run the code, navigate to the root directory of the project and schedule the job with the following command
```
qsub lung_GEM_NN.pbs
```
While the job is running, several files will be created in the input directory but these can be ignored. When the job finishes, pbs will create an output file which just shows genreal print and error statements, this file begins with 'lungGTEx'. The important output is the results file which is saved in the output directory and has the extention '.emx_results_test'. This file lists the training loss, accuracy of the model, number of predictions correct, and number of predictions per epoch. There is an example of the result file in the output directory, this file will be overwritten each time the project finishes.

