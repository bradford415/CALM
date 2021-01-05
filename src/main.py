"""
 main.py
"""
import pandas as pd
import sys
import os
import warnings
import logging
import argparse
import utils
import dataset
import torch
import preprocessing
import numpy as np 
from datetime import datetime
from sklearn.metrics import f1_score
from plots import Plotter
from torch.utils import data
from definitions import INPUT_DIR
from definitions import OUTPUT_DIR


def train(model, device, is_binary, train_generator, optimizer, loss_fn, batch_size, loss_meter, train_stats):
    """
     Train the network and collect accuracy and loss in dataframes. Different loss functions will be 
     used if it is a binary prediction or multiclass prediction.
    """
    model.train() # Set model to train mode (default mode)

    total_items = 0
    acc = 0.0
    loss= 0.0
    for data, target in train_generator:
        data = data.unsqueeze(1).float()
        data, target = data.to(device), target.to(device)
        total_items += target.shape[0] 
        optimizer.zero_grad() # Zero out the gradients
        prediction = model(data)
        if is_binary:
            target = target.unsqueeze(1).float()
            acc += utils.bin_accuracy(target, prediction)
            loss = loss_fn(prediction, target.float())
        else:
            acc += utils.multi_accuracy(target, prediction)
            loss = loss_fn(prediction, target.long())
        loss.backward() # Compute gradients
        optimizer.step() # Upate weights

    # Calculate loss per epoch
    loss_meter.update(loss.item(), batch_size)
    acc_avg = acc/total_items
    train_stats = train_stats.append(pd.DataFrame([[acc_avg, loss_meter.avg]], columns=['accuracy', 'loss']), ignore_index=True)

    return train_stats

def test(model, device, is_binary, test_generator, loss_fn, epoch, batch_size, loss_meter, test_stats, train_stats, logger):
    """
     Test the model with the test dataset. Only doing forward passes, backpropagrations should not be applied
    """
    model.eval() # Set model to eval mode - required for dropout and norm layers
    
    total_items = 0
    acc = 0.0
    loss= 0.0
    loss_meter.reset()
    with torch.no_grad():
        for data, target in test_generator:
            data = data.unsqueeze(1).float()
            data, target = data.to(device), target.to(device)
            total_items += target.shape[0]
            prediction = model(data)
            if is_binary:
                target = target.unsqueeze(1).float()
                acc += utils.bin_accuracy(target, prediction)
                loss = loss_fn(prediction, target.float())
            else:
                acc += utils.multi_accuracy(target, prediction)
                loss = loss_fn(prediction, target.long())
            loss_meter.update(loss.item(), batch_size)

    loss_meter.update(loss.item(), batch_size)
    acc_avg = acc/total_items
    test_stats = test_stats.append(pd.DataFrame([[acc_avg, loss_meter.avg]], columns=['accuracy', 'loss']), ignore_index=True)

    # write training log to the log file
    logger.info('Epoch: %d Training Loss: %2.5f Test Accuracy : %2.3f Accurate Count: %d Total Items :%d '% (epoch, train_stats.iloc[epoch]['loss'], acc_avg, acc, total_items))
    loss_meter.reset()

    return test_stats


def forward(model, device, is_binary, test_generator, predict_list, target_list):
    """
     One forward pass through the model. mostly used to get confusion matrix values
    """
    with torch.no_grad():
        for data, target in test_generator:
            data = data.unsqueeze(1).float()
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            if is_binary:
                actual_labels = actual_labels.unsqueeze(1).float()
                pred_labels_sigmoid = torch.nn.Sigmoid(prediction)
                prediction_tags = (pred_labels_sigmoid >= 0.5).eq(actual_labels)
            else:
                prediction_softmax = torch.softmax(prediction, dim=1)
                _, prediction_tags = torch.max(prediction_softmax, dim=1)
            
            predict_list.append(prediction_tags.to('cpu'))
            target_list.append(target.to('cpu'))
            
    predict_list = [j for val in predict_list for j in val]
    target_list = [j for val in target_list for j in val]

    return predict_list, target_list

def main():
    # Maybe delete this ?
    group = 'lung'

    parser = argparse.ArgumentParser(description='classifier')
    parser.add_argument('--sample_file', type=str, default='lung.emx.txt', help="the name of the GEM organized by samples (columns) by genes (rows)")
    parser.add_argument('--label_file', type=str, default='sample_condition.txt', help="name of the label file: two columns that maps the sample to the label")
    parser.add_argument('--output_name', type=str, default='tissue-run-1', help="name of the output directory to store the output files")
    #parser.add_argument('--overwrite_output', type=bool, default=False, help="overwrite the output directory file if it already exists")
    parser.add_argument('--batch_size', type=int, default=16, help="size of batches to split data")
    parser.add_argument('--max_epoch', type=int, default=100, help="number of passes through a dataset")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="controls the rate at which the weights of the model update")
    parser.add_argument('--test_split', type=float, default=0.3, help="percentage of test data, the train data will be the remaining data. 30% -> 0.3")
    parser.add_argument('--continuous_discrete', type=str, default='continuous', help="type of data in the sample file, typically RNA will be continous and DNA will be discrete")
    parser.add_argument('--plot_results', type=bool, default=True, help="plots the sample distribution, training/test accuracy/loss, and confusion matrix")
    parser.add_argument('--use_gpu', type=bool, default=False, help="true to use a gpu, false to use the cpu - if the node does not have a gpu then it will use the cpu")
    args = parser.parse_args()

    #If data is discrete, data should only range between 0-3
    #if args.continuous_discrete == "discrete":
        #args.input_num_classes = 4

    # Initialize file paths and create output folder
    LABEL_FILE = os.path.join(INPUT_DIR, args.label_file)
    SAMPLE_FILE = os.path.join(INPUT_DIR, args.sample_file)
    OUTPUT_DIR_FINAL = os.path.join(OUTPUT_DIR, args.output_name + "-" + str(datetime.today().strftime('%Y-%m-%d-%H:%M')))
    if not os.path.exists(OUTPUT_DIR_FINAL):
        os.makedirs(OUTPUT_DIR_FINAL)

    # Create log file to keep track of model parameters
    logging.basicConfig(filename=os.path.join(OUTPUT_DIR_FINAL,'classifier.log'),
                        filemode='w',
                        format='%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Classifer log file for ' + args.sample_file + ' - Started on ' + str(datetime.today().strftime('%Y-%m-%d-%H:%M')) + '\n')
    logger.info('Batch size: %d', args.batch_size)
    logger.info('Number of epochs: %d', args.max_epoch)
    logger.info('Learning Rate: %f', args.learning_rate)
    logger.info('Sample filename: ' + args.sample_file)
    logger.info('Output directory: ' + args.output_name)

    if args.continuous_discrete != 'continuous' and args.continuous_discrete != 'discrete':
        logger.error("ERROR: check that the continuous_discrete argument is spelled correctly.")
        logger.error("       only continuous or discrete data can be processed.")
        sys.exit("\nCommand line argument error. Please check the log file.\n")

    # Intialize gpu usage if desired
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and args.use_gpu else "cpu")
    train_kwargs = {'batch_size': 16}
    test_kwargs = {'batch_size': 16}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Load matrix, labels/weights, and number of samples
    column_names = ("sample", "label")
    matrix_df = pd.read_csv(SAMPLE_FILE, sep='\t', index_col=[0])
    labels_df = pd.read_csv(LABEL_FILE, names=column_names, delim_whitespace=True, header=None)


    # Error checking for same number of samples in both files and samples are unique
    samples_unique = set(labels_df.iloc[:,0])
    assert len(labels_df) == len(matrix_df.columns)
    assert len(labels_df) == len(samples_unique)

    
    labels, class_weights = preprocessing.labels_and_weights(labels_df)
    args.output_num_classes = len(labels)
    is_binary = False
    if len(labels) == 2:
        is_binary = True
        args.output_num_classess = 1

    # Define model paramters
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    learning_rate = args.learning_rate #5e-4
    num_features = len(matrix_df.index)

    # Setup model
    model = utils.Net(input_seq_length=num_features,
                  output_num_classes=args.output_num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    if is_binary:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()#(weight=class_weights)

    logger.info('Number of samples: %d\n', len(labels_df))
    logger.info('Labels: ')
    for i in range(len(labels)):
        logger.info('       %d - %s', i, labels[i])
    
    # Replace missing data with the global minimum of the dataset
    val_min, val_max = np.nanmin(matrix_df), np.nanmax(matrix_df)
    matrix_df.fillna(val_min, inplace=True)

    # Transposing matrix to align with label file
    matrix_transposed_df = matrix_df.T

    # Create density and tsne plot
    graphs = Plotter(OUTPUT_DIR_FINAL)
    graphs.density(matrix_df)
    graphs.tsne(matrix_transposed_df, labels_df, labels, title=args.sample_file)

    train_data, test_data = preprocessing.split_data(matrix_transposed_df, labels_df, args.test_split, args.output_num_classes)

    # Convert tuple of df's to tuple of np's
    # Allows the dataset class to access w/ data[][] instead of data[].iloc[]
    train_data_np = (train_data[0].values, train_data[1].values)
    test_data_np = (test_data[0].values, test_data[1].values)

    train_dataset = dataset.Dataset(train_data_np)
    test_dataset = dataset.Dataset(test_data_np)
    train_generator = data.DataLoader(train_dataset, **train_kwargs, drop_last=False)
    test_generator = data.DataLoader(test_dataset, **test_kwargs, drop_last=False)
    # drop_last=True would drop the last batch if the sample size is not divisible by the batch size

    logger.info('\nTraining size: %d \nTesting size: %d\n', len(train_dataset), len(test_dataset))

    # Create variables to store accuracy and loss
    loss_meter = utils.AverageMeter()
    loss_meter.reset()
    summary_file = pd.DataFrame([], columns=['Epoch', 'Training Loss', 'Accuracy', 'Accurate Count', 'Total Items'])
    train_stats = pd.DataFrame([], columns=['accuracy', 'loss'])
    test_stats = pd.DataFrame([], columns=['accuracy', 'loss'])

    # Train and test the model
    for epoch in range(args.max_epoch):
        train_stats = train(model, device, is_binary, train_generator, optimizer, loss_fn, batch_size, loss_meter, train_stats)
        test_stats = test(model, device, is_binary, test_generator, loss_fn, epoch, batch_size, loss_meter, test_stats, train_stats, logger)
        scheduler.step()

    # Training finished - Below is used for testing the network, plots and saving results
    if(args.plot_results):
        y_predict_list = []
        y_target_list = []
        y_predict_list, y_target_list = forward(model, device, is_binary, test_generator, y_predict_list, y_target_list)

        graphs.accuracy(train_stats, test_stats, graphs_title=args.sample_file)
        graphs.confusion(y_predict_list, y_target_list, labels, cm_title=args.sample_file)
        logger.info("\n\nf1 score: %0.2f" % (f1_score(y_target_list, y_predict_list, average="weighted")))

    #summary_file.to_csv(RESULTS_FILE, sep='\t', index=False)
    logger.info('\nFinal Accuracy: %2.3f', test_stats.iloc[epoch]['accuracy'])
    logger.info('\nFinished at  ' + str(datetime.today().strftime('%Y-%m-%d-%H:%M')))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
    print("\nRUN COMPLETED\n\n")