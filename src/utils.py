"""
Utils.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from definitions import INPUT_DIR
from definitions import OUTPUT_DIR

class Net(nn.Module):
    def __init__(self,  input_seq_length, input_num_classes, output_num_classes):
        """Initialize model layers"""
        super(Net, self).__init__()
        self.input_seq_length = input_seq_length
        self.output_num_classes = output_num_classes

        self.fc1 = nn.Linear(self.input_seq_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_num_classes)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        """Forward pass through the model"""
        x = x.view(x.shape[0], self.input_seq_length)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        x = self.fc4(x)
        # Note:
        #   Softmax activation for output layer is not used because the nn.CrossEntropyLoss
        #   automatically applies it, so we just send it the raw output. The most likely
        #   class will be the index with the highest value. If probability is needed, the
        #   softmax function can be called when calculating accuracy, this is shown in the
        #   multi_acc function. Ultimately, the softmax as thelast activation function won't 
        #   change the classification result.
        return x


def multi_accuracy(actual_labels, predicted_labels):
    """Computes the accuracy for multiclass predictions"""
    pred_labels_softmax = torch.softmax(predicted_labels, dim=1)
    _, pred_labels_tags = torch.max(pred_labels_softmax, dim=1)

    correct = (pred_labels_tags == actual_labels).float()
    return correct.sum()
    

def bin_accuracy(actual_labels, predicted_labels):
    """Computes the accuracy for multiple binary predictions"""
    #actual_labels = actual_labels.unsqueeze(1).float()
    sig = torch.nn.Sigmoid()
    pred_labels_sigmoid = sig(predicted_labels)

    correct = (pred_labels_sigmoid >= 0.5).eq(actual_labels) 
    return correct.sum()


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
