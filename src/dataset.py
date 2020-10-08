
from torch.utils import data
import torch

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, input_num_classes, output_num_classes):
        """
        Initialization

        data - normalized sample data 2 indexes wide
               the first index is the sample data and
               the second index is the label
        input_num_classes - represents the normalization range to 
                            create one-hot labels. Example before this funciton is called 
                            the data is normalized between 0-9 => 10 values, so if the 
                            data value is 5 a one-hot tensor of value 
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] will be created, this variable has 
                            NOTHING to do with the number of labels. If you choose to not return 
                            one-hot labels the accuracy substantially decreases and im not sure why.
                            I should probably ask Ben.
        output_num_classes - number of labels that are being classified 
        """
        self.input_num_classes  = input_num_classes
        self.output_num_classes = output_num_classes
        self.data_input = data[0]
        self.data_output = data[1]
        assert len(self.data_input) == len(self.data_output)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_input)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        '''
        return (torch.nn.functional.one_hot(torch.tensor(self.data_input[index]),
                                            num_classes=self.input_num_classes),
                torch.nn.functional.one_hot(torch.tensor(self.data_output[index]),
                                            num_classes=self.output_num_classes),)
        '''
        return (torch.nn.functional.one_hot(torch.tensor(self.data_input[index]),
                                            num_classes=self.input_num_classes),
                torch.tensor(self.data_output[index]))
