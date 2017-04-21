import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RNNModel(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, layers_num):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.rnn = nn.LSTM(input_dim, hidden_dim, layers_num)
        self.ffd = nn.Linear(hidden_dim, hidden_dim)
        self.fcn = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def forward(self, input):
        # print('sequence_length = {}, feature_dim = {}'.format(input.size(0),
        #                                                       input.size(1)))
        # print('type of input: {}'.format(type(input.data)))
        #  print("input type: {}, hidden type: {}".format(type(input.data),
        #                                                 type(self.hidden[0].data)))
        output, self.hidden = self.rnn(input.view(input.size(0), 1,
                                                  input.size(1)), self.hidden)

        # print('size of LSTM-output = {}'.format(output.size()))

        output = self.fcn(F.relu(self.ffd(output.view(input.size(0), -1))))
        return output

    def init_hidden(self):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.layers_num, 1, self.hidden_dim).zero_()),
                Variable(weight.new(self.layers_num, 1, self.hidden_dim).zero_()))
