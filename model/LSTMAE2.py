import numpy as np
import torch

class LSTMAE(torch.nn.Module):
    def __init__(self, input_size ,lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(LSTMAE, self).__init__()
        self.linear_in_1 = torch.nn.linear(input_size, lin_hidden_size)
        self.linear_in_2 = torch.nn.linear(lin_hidden_size, lin_output_size)
        self.encoder = torch.nn.LSTM(lin_output_size, hidden_size, num_layers)
        # self.relu = torch.nn.ReLU()
        self.decoder = torch.nn.LSTM(hidden_size, lin_output_size, num_layers)
        self.linear_out_1 = torch.nn.linear(lin_output_size, lin_hidden_size)
        self.linear_out_2 = torch.nn.linear(lin_hidden_size, input_size)
        # self.tanh = torch.nn.Tanh()

    def forward(self, input):
        hidden_state = self.linear_in_1(input)
        hidden_state = self.linear_in_2(hidden_state)
        encoded_input, hidden_state_en = self.encoder(hidden_state)
        #encoded_input = self.relu(encoded_input)
        decoded_output, hidden_state_de = self.decoder(encoded_input)
        hidden_state = self.linear_out_1(decoded_output)
        final_out = self.linear_out_2(hidden_state)
        #decoded_output = self.relu(decoded_output)
        #decoded_output = self.tanh(decoded_output)
        return final_out
