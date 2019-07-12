import numpy as np
import torch
from torch.autograd import Variable

class EncoderLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        tt = torch.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        c0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        encoded_input, hidden = self.lstm(input, (h0, c0))
        encoded_input = self.relu(encoded_input)
        return encoded_input

class DecoderLSTM(torch.nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = torch.nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        c0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        decoded_output, hidden = self.lstm(encoded_input, (h0, c0))
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output


class LSTMAE(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderLSTM(hidden_size, input_size, num_layers, isCuda)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output
