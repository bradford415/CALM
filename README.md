# Classification Algorithm for Labeled Matrices
A multilayer perceptron neural network used to classify three or more labels through gene expression implemented in PyTorch. 

This project takes two input files, a gene expression matrix (GEM) and text file which lists the correct labels in one column. The GEM is transposed to allign with the label file, 80% of the samples are used to train the neural network, and then the network is tested with the last 20% of samples.

## Requirements
- torch
- pandas
- matplotlib
- sklearn
- seaborn

## Palmetto Setup
Because this project is meant to be run on the Palmetto Cluster, an Anaconda Environment will need to be set up with the required dependencies installed.

### Installation
Naviagte to the desired scratch directory and clone the repository as shown belown. On the palmetto cluster, the git module will probably need to be added first.

```bash
module add git/2.27.0-gcc/8.3.1

git clone https://github.com/bradford415/CALM.git
```

### Anaconda Environment
Create a virtual environment with the required dependecies. If a virtual envrionment already exists and meets these requirements, this step can be skipped.
```bash
module add anaconda3/5.10-gcc/8.3.1

conda create -n myenv python=3.7 pytorch pandas matplotlib sklearn seaborn
```
If you wish to use an existing environment, use the pip command to install each missing dependency inside your environment. Example:
```bash
source activate myenv

pip install seaborn
```

### PBS Script
The pbs file contains the necessary script to run CALM on the palmetto. The node resources, virtual environment, and command-line arguments are set in this script.

First, Open the classifier.pbs script and verify the resources of the node are correct. Show below are the default resources. 
```
select=1:ncpus=4:mem=32gb:interconnect=fdr,walltime=2:00:00
```
To use a gpu, replace the line above with the line below. For some reason the only gpu that will work with PyTorch is the p100 model. When requesting a gpu, the mem must be 16gb or greater. At least this was the case when testing the lung dataset.
```
select=1:ncpus=4:mem=16gb:ngpus=1:gpu_model=p100:interconnect=fdr,walltime=2:00:00
```

Next, change the virtual environment being used. The default environement is 'myenv'. Change the line below to use the virtual env you created or modified.
```
source activate myenv
```
The rest of the pbs file calls the main python script, along with specifying the command line arguments. The most important command line arguments are '--sample_file', '--label_file', and '--output_name', these MUST be changed. The command line arguments are shown below, followed by a description of each.
```
python src/main.py --sample_file lung.emx.txt \
                   --label_file sample_condiiton.txt \
                   --output_name tissue-run-1 \ 
                   --max_epoch 75 \
                   --batch_size 16 \
                   --learning_rate 0.001 \
                   --test_split 0.3 \
                   --continuous_discrete continuous \
                   --plot_results True \
                   --use_gpu False
```
- sample_file
- label_file
- output_name
- max_epoch
- batch_size
- learning_rate
- test_split
- continous_discrete
- plot_results
- use_gpu

### Input Files
Move the GEM file and labels file into the 'input' directory. Currently, the GEM file must be named 'lung.emx.txt' and the label file must be named 'lung_sample_condition_no_sample_names.txt'

## Running
To run the code, navigate to the root directory of the project and schedule the job with the following command
```
qsub lung_GEM_NN.pbs
```
While the job is running, several files will be created in the input directory but these can be ignored. 

To check the job status at any point, use this command
```
qstat -u <user_name>
```

## Output
When the job finishes, pbs will create an output file which just shows genreal print and error statements, this file begins with 'lungGTEx'. The important output is the results file which is saved in the output directory and has the extention '.emx_results_test'. This file lists the training loss, accuracy of the model, number of predictions correct, and number of predictions per epoch. There is an example of the result file in the output directory, this file will be overwritten each time the project finishes.

