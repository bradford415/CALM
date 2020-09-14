import multiprocessing
import random
import pandas as pd
from pandas import DataFrame
from pandas import read_csv
import csv
import glob
import os
import util_lung
import argparse
import dataset
import torch
import numpy as np
#import matplotlib.pyplot as plt
import math
from torch.utils import data
from definitions import INPUT_DIR
from definitions import OUTPUT_DIR

group = 'lung'
loci_number = 19648

parser = argparse.ArgumentParser(description='NetSeq.')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--input_num_classes', type=int, default=10)
parser.add_argument('--output_num_classes', type=int, default=4)
parser.add_argument('--seq_length', type=int, default=loci_number)
parser.add_argument('--sample_file', type=str, default='lung.emx.txt')
parser.add_argument('--label_file', type=str, default='lung_sample_condition_no_sample_names.txt')

args = parser.parse_args()

#normalize function
def NormalizeData(data, x, y):
    array = (data - np.min(data)) / (np.max(data) - np.min(data))
    range2 = y - x;
    normalized = (array*range2) + x
    return normalized

# Create file paths
LABEL_FILE = os.path.join(INPUT_DIR, args.label_file)
SAMPLE_FILE = os.path.join(INPUT_DIR, args.sample_file)
SAMPLE_STRINGS = os.path.join(INPUT_DIR, args.sample_file.split('.', 1)[0] + '_strings.emx')
NORMALIZED_SAMPLE = os.path.join(INPUT_DIR, args.sample_file.split('.', 1)[0] + '_strings_normalized.emx')
RESULTS_FILE = os.path.join(OUTPUT_DIR, "%s_results_test" %(args.sample_file))

print('Batch Size: %d Epochs: %d \n'% (args.batch_size, args.max_epoch))

#load matrix
print("Loading Matrix")
RNA_matrix = pd.read_csv(SAMPLE_FILE, sep='\t', index_col=[0])

# Get sequence length (num rows) - input to the nn
args.seq_length = len(RNA_matrix.index)

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

#convert to strings
print("turning matrix into strings")
strings = pd.DataFrame([])
strings['string'] = RNA_matrix_TR.iloc[:,:].astype(str).apply(lambda y: ','.join(y), axis = 1)
strings = strings.sort_index()

#save it
print("saving string file")
strings.to_csv(SAMPLE_STRINGS, sep='\t', index=False, header=False)
print("done saving string file")
#print(strings)

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
    sample_data = NormalizeData(sample_data, 0, 9)
    sample_data = sample_data.astype(int)
    for h in sample_data:
        f1.write(str(h))
    f1.write('\n')
f1.close()

# Deleting unnecessary files 
os.remove(SAMPLE_STRINGS)

for i in range(1):

    #args.filename = ('lung_strings_normalized.emx')
    #normalized_filename = file_name + '_strings_normalized.emx' 

    train_data, val_data = util_lung.parse_data(sample_file=NORMALIZED_SAMPLE,
                                           label_file=LABEL_FILE,
                                           input_seq_length=args.seq_length,
                                           input_num_classes=args.input_num_classes,
                                           output_num_classes=args.output_num_classes)
    train_dataset = dataset.Dataset(train_data, input_num_classes=args.input_num_classes,
                                    output_num_classes=args.output_num_classes)
    val_dataset = dataset.Dataset(val_data, input_num_classes=args.input_num_classes,
                                   output_num_classes=args.output_num_classes)

    net = util_lung.Net(input_seq_length=args.seq_length,
                   input_num_classes=args.input_num_classes,
                   output_num_classes=args.output_num_classes)

    #Training Code Generator
    batch_size = args.batch_size
    training_generator = data.DataLoader(train_dataset, batch_size=batch_size)
    val_generator = data.DataLoader(val_dataset, batch_size=batch_size)

    #loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 5e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True, patience=50)

    loss_avgmeter = util_lung.AverageMeter()
    
    summary_file = pd.DataFrame([], columns=['Epoch', 'Training Loss', 'Accuracy', 'Accurate Count', 'Total Items'])
    #train_stats = pd.DataFrame([], columns=['accuracy', 'loss'])
    #val_stats = pd.DataFrame([], columns=['accuracy', 'loss'])

    for epoch in range(args.max_epoch):

        total_items = 0
        acc = 0.0

        # Training 
        print("Step 1 - Training the network")
        for local_batch, local_labels in training_generator:

            total_items += local_labels.shape[0] 
            local_batch = local_batch.unsqueeze(1).float()

            # Predict in-sample labels
            prediction = net(local_batch)
            #pred_labels_softmax = torch.softmax(prediction, dim=1)
            #_, pred_labels_tags = torch.max(pred_labels_softmax, dim=1)

            #correct = (pred_labels_tags == local_labels).float()
            #acc += correct.sum()

            # Calculate loss and update weights
            loss = loss_fn(prediction, local_labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate loss per epoch - loss_avgmeter.sum stores this value
            loss_avgmeter.update(loss.item(), batch_size)
        
        acc_avg = acc/total_items
        #print(acc_avg)
        #print(loss_avgmeter.sum)
        #temp_stats = pd.DataFrame([acc_avg, loss_avgmeter.sum]) # might need to changed to .val instead of .sum
        #train_stats.append(temp_stats)

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

            # Predict out-sample labels
            pred_labels = net(local_batch)
            pred_labels_softmax = torch.softmax(pred_labels, dim=1)
            _, pred_labels_tags = torch.max(pred_labels_softmax, dim=1) 
            
            correct = (pred_labels_tags == local_labels).float()
            acc += correct.sum()

            #loss = loss_fn(pred_labels, local_labels.logn())
            #loss_avgmeter.update(loss.item(), batch_size)

        print("Step 3 - Calcuating loss/accuracy")

        acc_avg = acc/total_items
        #temp_stats = pd.DataFrame([acc_avg, loss_avgmeter.sum]) # might need to changed to .val instead of .sum
        #val_stats.append(temp_stats)



        run_file = pd.DataFrame([['%d' %epoch, '%2.5f' %loss_avgmeter.val, '%2.3f' %acc_avg, '%d' % acc, '%d' % total_items]], columns=['Epoch', 'Training Loss', 'Accuracy', 'Accurate Count', 'Total Items'])
        summary_file = summary_file.append(run_file)
        print(epoch)
        print('\nEpoch: %d Training Loss: %2.5f Accuracy : %2.3f Accurate Count: %d Total Items :%d \n'% (epoch, loss_avgmeter.val, acc_avg, acc, total_items))
        scheduler.step(acc)
        loss_avgmeter.reset()

    """# Plot Accuracy 
    plt.plot(train_stats['accuracy'])
    plt.plot(val_stats['accuracy'])
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper left")
    #plt.show()

    plt.savefig('accuracy.pdf')

    # Plot Loss 
    plt.plot(train_stats['loss'])
    plt.plot(val_stats['loss'])
    plt.title("Loss")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper left")
    #plt.show()

    plt.savefig('loss.pdf')"""

    summary_file.to_csv(RESULTS_FILE, sep='\t', index=False)
