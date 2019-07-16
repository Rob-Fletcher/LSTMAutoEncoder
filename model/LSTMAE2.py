import numpy as np
import torch

class LSTMAE(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAE, self).__init__()
        self.encoder = torch.nn.LSTM(input_size, hidden_size, num_layers)
        self.relu = torch.nn.ReLU()
        self.decoder = torch.nn.LSTM(hidden_size, input_size, num_layers)
        # self.tanh = torch.nn.Tanh()

    def forward(self, input):
        encoded_input, hidden_state_en = self.encoder(input)
        encoded_input = self.relu(encoded_input)
        decoded_output, hidden_state_de = self.decoder(encoded_input)
        decoded_output = self.relu(decoded_output)
        #decoded_output = self.tanh(decoded_output)
        return decoded_output
