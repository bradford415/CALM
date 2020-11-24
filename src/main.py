"""
 main.py
"""
import pandas as pd
import sys
import os
import logging
import argparse
import utils
import dataset
import plots
import torch
import preprocessing
import numpy as np 
from datetime import datetime
from plots import plot
from torch.utils import data
from definitions import INPUT_DIR
from definitions import OUTPUT_DIR


#def train():

def main():
    # Maybe delete this ?
    group = 'lung'

    parser = argparse.ArgumentParser(description='classifier')
    parser.add_argument('--sample_file', type=str, default='lung.emx.txt', help="the name of the GEM organized by samples (columns) by genes (rows)")
    parser.add_argument('--label_file', type=str, default='sample_condition.txt', help="name of the label file: two columns that maps the sample to the label")
    parser.add_argument('--output_name', type=str, default='tissue-1', help="name of the output directory to store the output files")
    #parser.add_argument('--overwrite_output', type=bool, default=False, help="overwrite the output directory file if it already exists")
    parser.add_argument('--batch_size', type=int, default=16, help="size of batches to split data")
    parser.add_argument('--max_epoch', type=int, default=100, help="number of passes through a dataset")
    parser.add_argument('--learning_rate', type=int, default=0.001, help="controls the rate at which the weights of the model update")
    parser.add_argument('--test_split', type=int, default=0.3, help="percentage of test data, the train data will be the remaining data")
    #parser.add_argument('--input_num_classes', type=int, default=10) # binning value, will come back to later when working with discrete data
    parser.add_argument('--continuous_discrete', type=str, default='continuous', help="type of data in the sample file, typically RNA will be continous and DNA will be discrete")
    parser.add_argument('--plot_results', type=bool, default=True, help="plots the sample distribution, training/test accuracy/loss, and confusion matrix")

    args = parser.parse_args()

    #If data is discrete, data should only range between 0-3
    #if args.continuous_discrete == "discrete":
        #args.input_num_classes = 4

    # Initialize file paths and create output folder
    LABEL_FILE = os.path.join(INPUT_DIR, args.label_file)
    SAMPLE_FILE = os.path.join(INPUT_DIR, args.sample_file)
    OUTPUT_DIR_FINAL = os.path.join(OUTPUT_DIR, "-" + args.output_name + "-" + str(datetime.today().strftime('%Y-%m-%d-%H:%M')))
    if not os.path.exists(OUTPUT_DIR_FINAL):
        os.mkdirs(OUTPUT_DIR_FINAL)

    # Create log file to keep track of model parameters
    logging.basicConfig(filename=os.path.join(OUTPUT_DIR_FINAL,'classifier.log'),
                        filemode='w',
                        format='%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Classifer log file for ' + args.sample_file + ' - Started on ' + str(datetime.today().strftime('%Y-%m-%d-%H:%M')) + '\n')
    logger.info('Batch size: %d', args.batch_size)
    logger.info('Number of epochs: %d', args.max_epoch)
    logger.info('Learning Rate: %d', args.learning_rate)
    logger.info('Sample filename: ' + args.sample_file)
    logger.info('Output directory: ' + args.output_name)

    if args.continuous_discrete != 'continuous' and args.continuous_discrete != 'discrete':
        logger.error("ERROR: check that the continuous_discrete argument is spelled correctly.")
        logger.error("       only continuous or discrete data can be processed.")
        sys.exit("\nCommand line argument error. Please check the log file.\n")

    # Load matrix
    matrix_df = pd.read_csv(SAMPLE_FILE, sep='\t', index_col=[0])

    # Get number of samples and list of labels - log this information
    column_names = ("sample", "label")
    labels_df = pd.read_csv(LABEL_FILE, names=column_names, delim_whitespace=True, header=None)
    labels, class_weights = preprocessing.labels_and_weights(labels_df)
    args.output_num_classes = len(labels)
    #is_binary = False
    #if len(labels) == 2:
    #    is_binary = True
    #    args.output_num_classess = 1

    # Define paramters
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    learning_rate = args.learning_rate #5e-4
    num_features = len(matrix_df.index)
    num_classes = len(labels)

    # Setup model
    model = utils.Net(input_seq_length=num_features,
                  output_num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=50)
    loss_fn = torch.nn.CrossEntropyLoss()#(weight=class_weights)

    logger.info('Number of samples: %d\n', args.seq_length)
    logger.info('Labels: ')
    for i in range(len(labels)):
        logger.info('       %d - %s', i, labels[i])
    
    # Replace missing data with the global minimum of the dataset
    val_min, val_max = np.nanmin(matrix_df), np.nanmax(matrix_df)
    matrix_df.fillna(val_min, inplace=True)

    graphs = plots.Plotter(OUTPUT_DIR_FINAL)
    graphs.density(matrix_df)

    # Transposing matrix to align with label file
    matrix_transposed_df = matrix_df.T
    train_data, test_data = preprocessing.split_data(matrix_transposed_df, labels_df, args.test_split, num_classes)

    # Convert tuple of df's to tuple of np's
    # Allows the dataset class to access w/ data[][] instead of data[].iloc[]
    train_data_np = (train_data[0].values, train_data[1].values)
    test_data_np = (test_data[0].values, test_data[1].values)

    train_dataset = dataset.Dataset(train_data_np)
    test_dataset = dataset.Dataset(test_data_np)
    train_generator = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
    test_generator = data.DataLoader(test_dataset, batch_size=batch_size, drop_last=False)
    # drop_last=True would drop the last batch if the sample size is not divisible by the batch size

    ## PICK UP HERE TOMORROW - IM TIRED ##################################################################3
    preprocessing.process_inputs(matrix_df, args)
    train_data, val_data = preprocessing.parse_data(sample_file=NORMALIZED_SAMPLE,
                                                        label_file=LABEL_FILE,
                                                        input_seq_length=args.seq_length,
                                                        input_num_classes=args.input_num_classes,
                                                        output_num_classes=args.output_num_classes)
    train_dataset = dataset.Dataset(train_data, input_num_classes=args.input_num_classes,
                                        output_num_classes=args.output_num_classes)
    val_dataset = dataset.Dataset(val_data, input_num_classes=args.input_num_classes,
                                    output_num_classes=args.output_num_classes)
    logger.info('\nTraining size: %d \nValidation size: %d', len(train_dataset), len(val_dataset))
    net = utils.Net(input_seq_length=args.seq_length,
                   input_num_classes=args.input_num_classes,
                   output_num_classes=args.output_num_classes)
    # Characterize dataset
    # drop_last adjusts the last batch size when the given batch size is not divisible by the number of samples
    batch_size = args.batch_size
    training_generator = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
    val_generator = data.DataLoader(val_dataset, batch_size=batch_size, drop_last=False)
    if is_binary:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 5e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True, patience=50)
    loss_avgmeter = utils.AverageMeter()

    summary_file = pd.DataFrame([], columns=['Epoch', 'Training Loss', 'Accuracy', 'Accurate Count', 'Total Items'])
    train_stats = pd.DataFrame([], columns=['accuracy', 'loss'])
    val_stats = pd.DataFrame([], columns=['accuracy', 'loss'])

    for epoch in range(args.max_epoch):
        total_items = 0
        acc = 0.0
        # Training 
        print("Step 1 - Training the network")
        for local_batch, local_labels in training_generator:
            total_items += local_labels.shape[0] 
            local_batch = local_batch.unsqueeze(1).float()
            # Predict in-sample labels and get training accuracy
            prediction = net(local_batch)
            if is_binary:
                local_labels = local_labels.unsqueeze(1).float()
                acc += utils.multi_accuracy(local_labels, prediction)
                loss = loss_fn(prediction, local_labels.float())
            else:
                acc += utils.multi_accuracy(local_labels, prediction)
                loss = loss_fn(prediction, local_labels.long())
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Calculate loss per epoch - loss_avgmeter.avg stores this value I think
            loss_avgmeter.update(loss.item(), batch_size)
        acc_avg = acc/total_items
        temp_stats = pd.DataFrame([[acc_avg, loss_avgmeter.avg]], columns=['accuracy', 'loss'])
        train_stats = train_stats.append(temp_stats, ignore_index=True)
        total_items = 0
        correct = 0
        acc = 0.0
        acc_avg = 0.0
        loss_avgmeter.reset()
        # Validation
        print("Step 2 - Testing the network")
        for local_batch, local_labels in val_generator:
            total_items += local_labels.shape[0]
            local_batch = local_batch.unsqueeze(1).float()
            # Predict out-sample labels (samples network hasn't seen) and get validation accuracy
            prediction = net(local_batch)
            if is_binary:
                local_labels = local_labels.unsqueeze(1).float()
                acc += utils.multi_accuracy(local_labels, prediction)
                loss = loss_fn(prediction, local_labels.float())
            else:
                acc += utils.multi_accuracy(local_labels, prediction)
                loss = loss_fn(prediction, local_labels.long())
            loss_avgmeter.update(loss.item(), batch_size)
        print("Step 3 - Saving loss/accuracy")
        acc_avg = acc/total_items
        temp_stats = pd.DataFrame([[acc_avg, loss_avgmeter.avg]], columns=['accuracy', 'loss'])
        val_stats = val_stats.append(temp_stats, ignore_index=True)
        run_file = pd.DataFrame([['%d' %epoch, '%2.5f' %train_stats.iloc[epoch]['loss'], '%2.3f' %acc_avg, '%d' % acc, '%d' % total_items]], columns=['Epoch', 'Training Loss', 'Accuracy', 'Accurate Count', 'Total Items'])
        summary_file = summary_file.append(run_file, ignore_index=True)
        print('\nEpoch: %d Training Loss: %2.5f Accuracy : %2.3f Accurate Count: %d Total Items :%d \n'% (epoch, train_stats.iloc[epoch]['loss'], acc_avg, acc, total_items))
        scheduler.step(acc)
        loss_avgmeter.reset()
    # All epochs finished - Below is used for testing the network, plots and saving results
    # Deleting unnecessary files 
    os.remove(SAMPLE_STRINGS)
    if(args.plot_results):
        # List to store predictions and actual labels for confusion matrix
        y_pred_list = []
        y_target_list = []
        # Test validation set to get confusion matrix values
        for local_batch, local_labels in val_generator:
            total_items += local_labels.shape[0]
            local_batch = local_batch.unsqueeze(1).float()
            pred_labels = net(local_batch)
            if is_binary:
                actual_labels = actual_labels.unsqueeze(1).float()
                pred_labels_sigmoid = torch.nn.Sigmoid(pred_labels)
                pred_labels_tags = (pred_labels_sigmoid >= 0.5).eq(actual_labels)
            else:
                pred_labels_softmax = torch.softmax(pred_labels, dim=1)
                _, pred_labels_tags = torch.max(pred_labels_softmax, dim=1)
            y_pred_list.append(pred_labels_tags)
            y_target_list.append(local_labels)
        y_pred_list = [j for val in y_pred_list for j in val]
        y_target_list = [j for val in y_target_list for j in val]
        plots.plot(train_stats, val_stats, y_target_list, y_pred_list, labels,
                    graphs_title=args.sample_file, cm_title=args.sample_file)
    summary_file.to_csv(RESULTS_FILE, sep='\t', index=False)
    logger.info('\nAccuracy: %2.3f', val_stats.iloc[epoch]['accuracy'])
    logger.info('\nFinished at  ' + str(datetime.today().strftime('%Y-%m-%d-%H:%M')))

if __name__ == '__main__':
    main()