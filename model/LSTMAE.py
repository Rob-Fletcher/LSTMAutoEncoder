import numpy as np
import torch
from torch.autograd import Variable

class EncoderLSTM(torch.nn.Module):
    def __init__(self, input_size, lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lin_hidden_size = lin_hidden_size
        self.lin_output_size = lin_output_size
        self.num_layers = num_layers

        self.relu = torch.nn.ReLU()
        self.lin_1 = torch.nn.Linear(input_size, lin_hidden_size)
        self.lin_2 = torch.nn.Linear(lin_hidden_size, lin_output_size)
        self.lstm = torch.nn.LSTM(lin_output_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input):
        hidden = self.lin_1(input)
        hidden = self.relu(hidden)
        hidden = self.lin_2(hidden)
        encoded_input, hidden = self.lstm(hidden)
        return encoded_input

class DecoderLSTM(torch.nn.Module):
    def __init__(self, output_size, lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.relu = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(hidden_size, lin_output_size, num_layers, batch_first=True)
        self.lin_1 = torch.nn.Linear(lin_output_size, lin_hidden_size)
        self.lin_2 = torch.nn.Linear(lin_hidden_size, output_size)

    def forward(self, encoded_input):
        decoded_output, hidden = self.lstm(encoded_input)
        hidden = self.lin_1(decoded_output)
        hidden = self.relu(hidden)
        output = self.lin_2(hidden)
        return output


class LSTMAE(torch.nn.Module):
    def __init__(self, input_size, lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderLSTM(input_size, lin_hidden_size, hidden_size, lin_output_size, num_layers)
        self.decoder = DecoderLSTM(input_size, lin_hidden_size, hidden_size, lin_output_size, num_layers)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output
