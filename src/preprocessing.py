import pandas as pd
import numpy as np
import os
import math
import random
import torch
from sklearn import model_selection
from definitions import INPUT_DIR
from definitions import OUTPUT_DIR

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

def process_inputs(matrix_df, args):

    # Initialize file paths
    LABEL_FILE = os.path.join(INPUT_DIR, args.label_file)
    SAMPLE_FILE = os.path.join(INPUT_DIR, args.sample_file)
    SAMPLE_STRINGS = os.path.join(INPUT_DIR, args.sample_file.split('.', 1)[0] + '_strings.emx')
    NORMALIZED_SAMPLE = os.path.join(INPUT_DIR, args.sample_file.split('.', 1)[0] + '_strings_normalized.emx')
    RESULTS_FILE = os.path.join(OUTPUT_DIR, "%s_results_test" %(args.sample_file))

    # Puts samples that have any nan values at the end
    is_NaN = data_matrix.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = data_matrix[row_has_NaN]
    rows_without_NaN = data_matrix[~row_has_NaN]
    data_matrix_T = pd.concat([rows_without_NaN, rows_with_NaN], axis=0)

    #transpose so that samples are along y axis
    #sort df by index so that sample names match label names
    print("transposing matrix")
    data_matrix_TR = data_matrix_T.T
    data_matrix_TR = data_matrix_TR.sort_index()

    #get rid of nans
    #data_matrix_TR = data_matrix_TR.replace(np.nan, '', regex=True)

    # Convert data to strings
    print("turning matrix into strings")
    strings = pd.DataFrame([])
    strings['string'] = data_matrix_TR.iloc[:,:].astype(str).apply(lambda y: ','.join(y), axis = 1)
    strings = strings.sort_index()
    samples_list = strings.index.tolist()

    # Removes column/row labels when saving to csv
    strings.to_csv(SAMPLE_STRINGS, sep='\t', index=False, header=False)
    print("done saving string file")

    ##normalize
    f = open(SAMPLE_STRINGS, 'r')
    f1 = open(NORMALIZED_SAMPLE, 'w+')

    f_lines = f.readlines()
    f.close()

    for index, j in enumerate(f_lines):
        j = j.rstrip('\n')
        j = j.split(',')
        sample_data = ([float(i) for i in j])
        amin, amax = min(sample_data), max(sample_data)
        sample_data = [amin if math.isnan(x) else x for x in sample_data]
        if args.continuous_discrete == "continuous":
            sample_data = NormalizeData(sample_data, 0, 9)
        sample_data = sample_data.astype(int)
        f1.write(str(samples_list[index]) + " ")
        for h in sample_data:
            f1.write(str(h))
        f1.write('\n')
    f1.close()


def parse_data(sample_file=None, label_file=None, input_seq_length=19648, input_num_classes=10, output_num_classes=4, samples=10):
    if sample_file != None:
        SAMPLES_FILENAME = os.path.join(INPUT_DIR, sample_file)
        LABEL_FILENAME = os.path.join(INPUT_DIR, label_file)

        # Create ordered dictionary of unique labels
        with open(LABEL_FILENAME) as label_lines:
            label_lines = label_lines.readlines()
            labels = [s.split()[1].strip('\n') for s in label_lines]
            labels_unique = set(labels)
            labels = sorted(labels_unique)
            labels_dict = {k: v for v, k in enumerate(labels)}

        # Load each file into a data frame and merge these dataframes
        column_names_1 = ("sample", "label")
        column_names_2 = ("sample", "data")
        labels_df = pd.read_csv(LABEL_FILENAME, names=column_names_1, delim_whitespace=True, header=None)
        samples_df = pd.read_csv(SAMPLES_FILENAME, names=column_names_2, delim_whitespace=True, header=None)
        data_df = pd.merge(samples_df, labels_df, on='sample', how='left')

        # A 2D list to store all samples organized by their label
        # Example with 4 labels:
        #      [[sample15, sample10, sample8, sample0], <-- label 0
        #       [sample14, sample2,  sample3],          <-- label 1
        #       [sample4, sample6, sample7, sample1],   <-- label 2
        #       [sample5, sample12, sample13]]          <-- label 3
        all_samples = [[] for i in range(len(labels))]

        # Create a data dictionary to store details about every sample
        # Example entry: {'sample27':{'label' : 2.0, 'data' : 123456}}
        data_dict = {}
        for index, row in data_df.iterrows():
            sample_val = row['sample']
            label_val = row['label']
            data_val = row['data']
            if sample_val not in data_dict:
                data_dict[sample_val] = {}
            if label_val in labels_dict:
                label_index = labels_dict[label_val]
                all_samples[label_index].append(sample_val)
                data_dict[sample_val]['label'] = float(label_index)
            else:
                raise("Label mismatch")
            data_dict[row['sample']]['data'] = row['data'].replace("-1","3")
            # For DNA samples, -1 denotes a 'T', 'C' or undetected genotype
            # The -1 is then converted to a 3 by convention

        assert len(labels_df) == len(data_dict) and len(samples_df) == len(data_dict)

        # Find the number of occurences for each label and only take 80% of it
        label_weightings = [int(0.8 * len(length)) for length in all_samples]
 
        # Create training and validation lists
        sample_train = [random.sample(samples, k=label_weightings[index]) \
                        for index, samples in enumerate(all_samples)]
        sample_train = [j for sub in sample_train for j in sub]
 
        sample_val = [ set(samples).difference(set(sample_train)) for samples in all_samples]
        sample_val = [ j for sub in sample_val for j in sub]
 
        data_train_input, data_train_output = [], []
        data_val_input, data_val_output = [], []

        # Match the index from the randomized list to the dictionary key
        # Store the data in one list and the label in another - input and output
        # Finally, return these lists
        for sample_train_item in sample_train:
            tmp = []
            data_string = data_dict[sample_train_item]['data'].replace("\n","")
            for string_index in range(len(data_string)):
                tmp.append(int(data_string[string_index]))
            data_train_input.append(tmp)
            data_train_output.append(data_dict[sample_train_item]['label'])

        for sample_val_item in sample_val:
            tmp = []
            data_string = data_dict[sample_val_item]['data'].replace("\n","")
            for string_index in range(len(data_string)):
                tmp.append(int(data_string[string_index]))
            data_val_input.append(tmp)
            data_val_output.append(data_dict[sample_val_item]['label'])
        print('\nTraining Size: %d\nVal Size : %d \n\n' %(len(data_train_input), len(data_val_input)))
        
        return (data_train_input, data_train_output), (data_val_input, data_val_output)
