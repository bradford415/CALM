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
import math
from torch.utils import data

loci_number = 19648
group = 'lung'

#normalize function
def NormalizeData(data, x, y):
    array = (data - np.min(data)) / (np.max(data) - np.min(data))
    range2 = y - x;
    normalized = (array*range2) + x
    return normalized

#load matrix
print("loading matrix")
RNA_matrix = pd.read_csv("lung.emx.txt", sep='\t', index_col=[0])

is_NaN = RNA_matrix.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = RNA_matrix[row_has_NaN]
rows_without_NaN = RNA_matrix[~row_has_NaN]
#print(len(rows_without_NaN))
RNA_matrix_T = pd.concat([rows_without_NaN, rows_with_NaN], axis=0)
#print(len(RNA_matrix_final_T))

#transpose so that samples are along y axis
print("transposing matrix")
RNA_matrix_TR = RNA_matrix_T.T
#sort df by index so that sample names match label names
RNA_matrix_TR = RNA_matrix_TR.sort_index()
#get rid of nans
#RNA_matrix_TR = RNA_matrix_TR.replace(np.nan, '', regex=True)

#save matrix to double check it looks like what I think it should look like
#sample names down y axis, genes as column names
print("saving copy of matrix")
RNA_matrix_TR.to_csv("lung_format_check.emx", sep='\t')

#convert to strings
print("turning matrix into strings")
strings = pd.DataFrame([])
strings['string'] = RNA_matrix_TR.iloc[:,:].astype(str).apply(lambda y: ','.join(y), axis = 1)
strings = strings.sort_index()
#print(strings)

#save it
print("saving string file")
strings.to_csv("lung_strings.emx", sep='\t', index=False, header=False)
print("done saving string file")
#print(strings)

##normalize

f = open('lung_strings.emx', 'r')
f1 = open('lung_strings_normalized.emx', 'w+')

f = f.readlines()
#print(len(f))

for j in f:
    j = j.rstrip('\n')
    j = j.split(',')
   #print(j)
    sample_data = ([float(i) for i in j])
    amin, amax = min(sample_data), max(sample_data)
    sample_data = [amin if math.isnan(x) else x for x in sample_data]
    sample_data = NormalizeData(sample_data, 0, 9)
    sample_data = sample_data.astype(int)
    #print(sample_data)
    for h in sample_data:
        #print(h)
        f1.write(str(h))
    f1.write('\n')
f1.close()

parser = argparse.ArgumentParser(description='NetSeq.')
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--input_num_classes', type=int, default=10)
parser.add_argument('--output_num_classes', type=int, default=4)
parser.add_argument('--seq_length', type=int, default=loci_number)
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=16)

for i in range(1):

    args = parser.parse_args()
    #print(args)
    args.filename = ('lung_strings_normalized.emx')


    train_data, val_data = util_lung.parse_data(filename=args.filename,
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
    val_generator = data.DataLoader(val_dataset, batch_size=1)

    #loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 5e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True, patience=50)

    loss_avgmeter = util_lung.AverageMeter()

    summary_file = pd.DataFrame([], columns=['Epoch', 'Training Loss', 'Accuracy', 'Accurate Count', 'Total Items'])

    for epoch in range(args.max_epoch):
        # Training
        print(epoch)
        print("step 1")
        for local_batch, local_labels in training_generator:
            local_batch=local_batch.unsqueeze(1).float()
            #local_labels = local_labels.unsqueeze(1).float()
            prediction = net(local_batch)
            #loss = loss_fn(prediction, local_labels.float())
            loss = loss_fn(prediction, local_labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avgmeter.update(loss.item(), batch_size)

        total_val_items= 0
        acc = 0.0
        print("step 2")

        for local_batch, local_labels in val_generator:

            total_val_items += local_labels.shape[0]
            local_batch = local_batch.unsqueeze(1).float()
            #local_labels = local_labels.unsqueeze(1).float()

            #pred_labels = torch.nn.Sigmoid()(net(local_batch))
            pred_labels = net(local_batch)
            #softmax = torch.nn.Softmax(1)
            pred_labels_softmax = torch.softmax(pred_labels, dim=1)
            _, pred_labels_tags = torch.max(pred_labels_softmax, dim=1) 
            
            #acc  += (pred_labels >= 0.5).eq(local_labels)
            correct = (pred_labels_tags == local_labels).float()
            acc += correct.sum()

        print("step 3")

        acc_avg = acc/total_val_items
        #print("Accuracy average")
        #print(acc_avg)

        run_file = pd.DataFrame([['%d' %epoch, '%2.5f' %loss_avgmeter.val, '%2.3f' %acc_avg, '%d' % acc, '%d' % total_val_items]], columns=['Epoch', 'Training Loss', 'Accuracy', 'Accurate Count', 'Total Items'])
        summary_file = summary_file.append(run_file)
        print(epoch)
        print('Epoch: %d Training Loss: %2.5f Accuracy : %2.3f Accurate Count: %d Total Items :%d '% (epoch, loss_avgmeter.val, acc_avg, acc, total_val_items))
        scheduler.step(acc)
        loss_avgmeter.reset()

    summary_file.to_csv("%s_results_test" %(args.filename), sep='\t', index=False)
