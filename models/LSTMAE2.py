import numpy as np
import torch

class LSTMAE(torch.nn.Module):
    def __init__(self, input_size ,lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(LSTMAE, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

        self.linear_in_1 = torch.nn.Linear(input_size, lin_hidden_size)
        self.linear_in_2 = torch.nn.Linear(lin_hidden_size, lin_output_size)
        self.encoder = torch.nn.LSTM(lin_output_size, hidden_size, num_layers)

        self.decoder = torch.nn.LSTM(hidden_size, lin_output_size, num_layers)
        self.linear_out_1 = torch.nn.Linear(lin_output_size, lin_hidden_size)
        self.linear_out_2 = torch.nn.Linear(lin_hidden_size, input_size)

    def forward(self, input):
        hidden_state = self.linear_in_1(input)
        hidden_state = self.relu1(hidden_state)
        hidden_state = self.linear_in_2(hidden_state)
        encoded_input, hidden_state_en = self.encoder(hidden_state)
        decoded_output, hidden_state_de = self.decoder(encoded_input)
        hidden_state = self.linear_out_1(decoded_output)
        hidden_state = self.relu2(hidden_state)
        final_out = self.linear_out_2(hidden_state)
        return final_out
