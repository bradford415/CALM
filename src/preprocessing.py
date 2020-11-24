import pandas as pd
import numpy as np
import torch
from sklearn import model_selection

def labels_and_weights(label_file_df):
    """ 
     Get list of unique sample labels and weights of the samples using
     the inverse of the count. Weights is a tensor to be compatible with
     CrossEntropyLoss.
    """
    labels_all = label_file_df.iloc[:,-1].astype(str).values.tolist()
    labels_unique = set(labels_all)
    labels = sorted(labels_unique)
    
    labels_count = [labels_all.count(label) for label in labels]
    weights = 1. / torch.tensor(labels_count, dtype=torch.float) 
    
    return labels, weights

def split_data(matrix_transposed_df, label_file_df, test_split, num_classes=4):
    """ Merge sample with label file and create train/test set """
    
    # Create dictionary of labels - key:labels, value:indices
    labels = label_file_df.iloc[:,-1].astype(str).values.tolist()
    labels_unique = set(labels)
    labels = sorted(labels_unique)
    labels_dict = {k:v for v, k in enumerate(labels)}
    
    merged_df = pd.merge(matrix_transposed_df, label_file_df, left_index=True, right_on='sample')
    del merged_df['sample']
    merged_df['label'].replace(labels_dict, inplace=True)
    
    X = merged_df.iloc[:, 0:-1]
    y = merged_df.iloc[:, -1]
    
    # stratify=y -> weights the train and test labels
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, test_size=test_split)
    
    return (X_train, y_train), (X_test, y_test)


def NormalizeData(data, x, y):
    """Normalize (bin) between two values x and y"""
    array = (data - np.min(data)) / (np.max(data) - np.min(data))
    range2 = y - x
    normalized = (array*range2) + x
    return normalized
