import multiprocessing
import random
import pandas as pd
import csv
import glob
import os
import logging
import utils
import argparse
import dataset
import torch
import numpy as np
import math
from datetime import datetime
from torch.utils import data
from definitions import INPUT_DIR
from definitions import OUTPUT_DIR

# Maybe delete this ?
group = 'lung'
loci_number = 19648

parser = argparse.ArgumentParser(description='NetSeq.')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--input_num_classes', type=int, default=10) # Normalization range - ex: 0-9 => 10
parser.add_argument('--output_num_classes', type=int, default=4)
parser.add_argument('--seq_length', type=int, default=loci_number)
parser.add_argument('--sample_file', type=str, default='lung.emx.txt')
parser.add_argument('--label_file', type=str, default='lung_sample_condition_no_sample_names.txt')
parser.add_argument('--continuous_discrete', type=str, default='continuous')
parser.add_argument('--plot_results', type=bool, default=True)

args = parser.parse_args()

# Create log file to keep track of network statistics
logging.basicConfig(filename=os.path.join(OUTPUT_DIR,'classifier_' + str(datetime.today().strftime('%Y-%m-%d-%H')) + '.log'),
                    filemode='w',
                    format='%(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('Network log file for ' + args.sample_file + ' - ' + str(datetime.today().strftime('%Y-%m-%d-%H:%M')) + '\n')
logger.info('Batch size: %d', args.batch_size)
logger.info('Number of epochs: %d', args.max_epoch)
logger.info('Sample filename: ' + args.sample_file)

if args.continuous_discrete != 'continuous' or args.continuous_discrete != 'discrete':
    logger.error("ERROR: check that the continuous_discrete argument is spelled correctly.")
    logger.error("       only continuous or discrete data can be processed.")
    sys.exit("\nCommand line argument error. Please check the log file.\n")

# If data is discrete, data should only range between 0-3
if args.continuous_discrete == "discrete":
    args.input_num_classes = 4

# Initialize file paths
LABEL_FILE = os.path.join(INPUT_DIR, args.label_file)
SAMPLE_FILE = os.path.join(INPUT_DIR, args.sample_file)
SAMPLE_STRINGS = os.path.join(INPUT_DIR, args.sample_file.split('.', 1)[0] + '_strings.emx')
NORMALIZED_SAMPLE = os.path.join(INPUT_DIR, args.sample_file.split('.', 1)[0] + '_strings_normalized.emx')
RESULTS_FILE = os.path.join(OUTPUT_DIR, "%s_results_test" %(args.sample_file))

print('Batch Size: %d Epochs: %d \n'% (args.batch_size, args.max_epoch))

# Load matrix
print("Loading Matrix")
RNA_matrix = pd.read_csv(SAMPLE_FILE, sep='\t', index_col=[0])

# Get number of samples and list of labels - log this information
args.seq_length = len(RNA_matrix.index)
labels = utils.get_labels(LABEL_FILE)
args.output_num_classes = len(labels)
if len(labels) == 2:
    is_binary = True
    args.output_num_classess = 1
logger.info('Number of samples: %d\n', args.seq_length)
logger.info('Labels: ')
for i in range(len(labels)):
    logger.info('       %d - %s', i, labels[i])

# Puts genes that have any nan values at the end
is_NaN = RNA_matrix.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = RNA_matrix[row_has_NaN]
rows_without_NaN = RNA_matrix[~row_has_NaN]
RNA_matrix_T = pd.concat([rows_without_NaN, rows_with_NaN], axis=0)

#transpose so that samples are along y axis
#sort df by index so that sample names match label names
print("transposing matrix")
RNA_matrix_TR = RNA_matrix_T.T
RNA_matrix_TR = RNA_matrix_TR.sort_index()

#get rid of nans
#RNA_matrix_TR = RNA_matrix_TR.replace(np.nan, '', regex=True)

#save matrix to double check it looks like what I think it should look like
#sample names down y axis, genes as column names
#RNA_matrix_TR.to_csv("lung_format_check.emx", sep='\t')

# Convert data to strings
print("turning matrix into strings")
strings = pd.DataFrame([])
strings['string'] = RNA_matrix_TR.iloc[:,:].astype(str).apply(lambda y: ','.join(y), axis = 1)
strings = strings.sort_index()

print("saving string file")
strings.to_csv(SAMPLE_STRINGS, sep='\t', index=False, header=False)
print("done saving string file")

##normalize
f = open(SAMPLE_STRINGS, 'r')
f1 = open(NORMALIZED_SAMPLE, 'w+')

f = f.readlines()

for j in f:
    j = j.rstrip('\n')
    j = j.split(',')
    sample_data = ([float(i) for i in j])
    amin, amax = min(sample_data), max(sample_data)
    sample_data = [amin if math.isnan(x) else x for x in sample_data]
    if args.continuous_discrete == "continuous":
        sample_data = utils.NormalizeData(sample_data, 0, 9)
    sample_data = sample_data.astype(int)
    for h in sample_data:
        f1.write(str(h))
    f1.write('\n')
f1.close()



for i in range(1):

    train_data, val_data = utils.parse_data(sample_file=NORMALIZED_SAMPLE,
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
            pred_labels = net(local_batch)
            if is_binary:
                local_labels = local_labels.unsqueeze(1).float()
                acc += utils.multi_accuracy(local_labels, prediction)
                loss = loss_fn(prediction, local_labels.float())
            else:
                acc += utils.multi_accuracy(local_labels, prediction)
                loss = loss_fn(prediction, local_labels.long())

            loss_avgmeter.update(loss.item(), batch_size)

        print("Step 3 - Calcuating loss/accuracy")

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

        utils.plots(train_stats, val_stats, y_target_list, y_pred_list, labels,
                    graphs_title=args.sample_file, cm_title=args.sample_file)

    summary_file.to_csv(RESULTS_FILE, sep='\t', index=False)
    logger.info('\nAccuracy: %2.3f', val_stats.iloc[epoch]['accuracy'])
    logger.info('\nFinished at  ' + str(datetime.today().strftime('%Y-%m-%d-%H:%M')))
    
