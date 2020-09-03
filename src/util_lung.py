"""
Utils.py
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from definitions import INPUT_DIR

def parse_data(sample_file=None, label_file=None, input_seq_length=19648, input_num_classes=10, output_num_classes=4, samples=10):
    if sample_file != None:
        SAMPLES_FILENAME = os.path.join(INPUT_DIR, sample_file)
        LABEL_FILENAME = os.path.join(INPUT_DIR, label_file)
        sample_lines = open(SAMPLES_FILENAME).readlines()
        label_lines = open(LABEL_FILENAME).readlines()
        assert len(sample_lines) == len(label_lines)

        # Map label index to sample data in a dictionary 
        data = {}
        for line_idx in range(len(sample_lines)):
            sample_data = sample_lines[line_idx]
            sample_name = str(line_idx)
            if sample_name not in data:
                data[sample_name] = {}
            data[sample_name]['data'] = sample_data.replace("-1","3")

        # Get unique all unique labels, sort them, create a dictionary to assign label values
        labels = [s.strip('\n') for s in set(label_lines)]
        labels.sort()
        labels_dict = {k: v for v, k in enumerate(labels)}

        all_sample_indexes = [[] for i in range(len(labels))]

        # Append the index for each label to the list
        # and add a 2nd value (the label) to the index key in the dictionary
        # A dictionary entry will look like: '1400': {'data': '123456', 'label': 2.0}
        for line_idx in range(len(label_lines)):
            label_data = label_lines[line_idx].replace("\n","")
            name = str(line_idx)

            if label_data in labels_dict:
                label_data = labels_dict[label_data]
                all_sample_indexes[label_data].append(name)
                label_data = float(label_data)
            else:
                raise ('Issue')

            if name in data:
                data[name]['label'] = label_data
            else:
                print('Skipping', name)
                
        # Find the number of occurences for each label and only take 80% of it
        label_weightings = [int(0.8 * len(length)) for length in all_sample_indexes]

        # Create training list of the sample indexes with proper label weighting, then flatten to 1d
        sample_train = [random.sample(samples, k=label_weightings[index]) \
                        for index, samples in enumerate(all_sample_indexes)]
        sample_train = [j for sub in sample_train for j in sub]

        # Create validation list (output for sample_val is actually a set), then flatten to 1d
        sample_val = [ set(samples).difference(set(sample_train)) for samples in all_sample_indexes]
        sample_val = [ j for sub in sample_val for j in sub]

        data_train_input, data_train_output = [],[]
        data_val_input, data_val_output = [], []

        # Match the index from the randomized list to the dictionary key 
        # Store the data in one list and the label in another
        # Finally, return these lists
        for sample_train_item in sample_train:
            tmp = []
            data_string = data[sample_train_item]['data'].replace("\n","")
            for string_index in range(len(data_string)):
                tmp.append(int(data_string[string_index]))
            data_train_input.append(tmp)
            data_train_output.append(data[sample_train_item]['label'])

        for sample_val_item in sample_val:
            tmp = []
            data_string = data[sample_val_item]['data'].replace("\n","")
            for string_index in range(len(data_string)):
                tmp.append(int(data_string[string_index]))
            data_val_input.append(tmp)
            data_val_output.append(data[sample_val_item]['label'])
        print('\nTraining Size: %d\n Val Size : %d \n\n' %(len(data_train_input), len(data_val_input)))
        return (data_train_input, data_train_output), (data_val_input, data_val_output)

    else: #Create Data
        input_data = []
        data_train_input = np.random.randint(input_num_classes, size=(samples,input_seq_length))
        data_train_output = np.random.randint(output_num_classes, size=(samples))

        data_val_input = np.random.randint(input_num_classes, size=(samples,input_seq_length))
        data_val_output = np.random.randint(output_num_classes,size=(samples))
        return (data_train_input, data_train_output), (data_val_input, data_val_output)


class Net(nn.Module):
    def __init__(self,  input_seq_length, input_num_classes, output_num_classes):
        super(Net, self).__init__()
        self.input_seq_length = input_seq_length
        self.input_num_classes = input_num_classes
        self.output_num_classes = output_num_classes

        # self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(self.input_num_classes*self.input_seq_length, 800)
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, 400)
        #self.fc4 = nn.Linear(10, self.output_num_classes)
        self.fc4 = nn.Linear(400, 220)
        self.fc5 = nn.Linear(220, 120)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, 10)
        self.fc8 = nn.Linear(10, output_num_classes)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_num_classes*self.input_seq_length* 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        return x



def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum() / target.numel()
    return acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
