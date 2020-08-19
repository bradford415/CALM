
from torch.utils import data
import torch

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, input_num_classes, output_num_classes):
        'Initialization'
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
