"""
Utils.py
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random


def parse_data(filename=None, input_seq_length=19648, input_num_classes=10, output_num_classes=4, samples=10):
    if filename != None:
        samples_filename = filename
        label_filename = '../input/lung_sample_condition_no_sample_names.txt'
        sample_lines = open("../input/" + samples_filename).readlines()
        label_lines = open("../input/" + label_filename).readlines()
        assert len(sample_lines) == len(label_lines)
        data = {}
        for line_idx in range(len(sample_lines)):
            #sample_name, sample_data = sample_lines[line_idx].replace("\n","").split("\t")
            sample_data = sample_lines[line_idx]
            sample_name = str(line_idx)
            if sample_name not in data:
                data[sample_name] = {}
            data[sample_name]['data'] = sample_data.replace("-1","3")

        GN_ctr = []
        TLUAD_ctr = []
        TLUSC_ctr = []
        TN_ctr = []

        for line_idx in range(len(label_lines)):
            label_data = label_lines[line_idx].replace("\n","")
            #print(label_data)
            name = str(line_idx)
            if label_data == 'GTEx_normal':
                label_data = 0.0
                GN_ctr.append(name)
            elif label_data == 'TCGA-LUAD':
                label_data = 1.0
                TLUAD_ctr.append(name)
            elif label_data == 'TCGA-LUSC':
                label_data = 2.0
                TLUSC_ctr.append(name)
            elif label_data == 'TCGA-NORMAL':
                label_data = 3.0
                TN_ctr.append(name)
            else:
                print(label_data)
                raise ('Issue')

            if name in data:
                data[name]['prediction'] = label_data
            else:
                print('Skipping', name)

        #print('--------------------------')
        #print('Total Dataset : %d ' % ( len ( data)))
        #print('Pos : %d Negative %d' % (len(true_ctr), len(false_ctr)))
        #print('--------------------------')

        ##sample_true=int(0.8 * len(true_ctr))
        ##sample_false = int(0.8 * len(false_ctr))
        sample_GN = int(0.8 * len(GN_ctr))
        sample_TLUAD = int(0.8 * len(TLUAD_ctr))
        sample_TLUSC = int(0.8 * len(TLUSC_ctr))
        sample_TN = int(0.8 * len(TN_ctr))

        sample_train = list(random.sample(GN_ctr, k=sample_GN)) + \
		list(random.sample(TLUAD_ctr, k=sample_TLUAD)) + \
                list(random.sample(TLUSC_ctr, k=sample_TLUSC)) + \
                list(random.sample(TN_ctr, k=sample_TN))

        sample_val = set(list(GN_ctr) + \
		list(TLUAD_ctr) + \
                list(TLUSC_ctr) + \
                list(TN_ctr)).difference(set(sample_train))

        #print(len(sample_train), len(sample_val))
        data_train_input, data_train_output = [],[]
        data_val_input, data_val_output = [], []

        for sample_train_item in sample_train:
            tmp = []
            data_string = data[sample_train_item]['data'].replace("\n","")
            #print(data_string)
            for string_index in range(len(data_string)):
                #print(string_index)
                tmp.append(int(data_string[string_index]))
            #print(len(tmp))
            #assert(len(tmp)) == 230
            data_train_input.append(tmp)
            data_train_output.append(data[sample_train_item]['prediction'])

        for sample_val_item in sample_val:
            tmp = []
            data_string = data[sample_val_item]['data'].replace("\n","")
            for string_index in range(len(data_string)):
                tmp.append(int(data_string[string_index]))
            #print(len(tmp))
            #assert(len(tmp)) == input_seq_length
            data_val_input.append(tmp)
            data_val_output.append(data[sample_val_item]['prediction'])
        #print('Training Size: %d \n Val Size : %d' %(len(data_train_input), len(data_val_input)))
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
