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
        self.lstm = torch.nn.LSTM(lin_output_size, hidden_size, num_layers)

    def forward(self, input):
        hidden = self.lin_1(input)
        hidden = self.relu(hidden)
        hidden = self.lin_2(hidden)
        encoded_input, (hidden, cell) = self.lstm(hidden)
        return encoded_input

class DecoderLSTM(torch.nn.Module):
    def __init__(self, output_size, lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.lstm = torch.nn.LSTM(hidden_size, lin_output_size, num_layers)
        self.lin_1 = torch.nn.Linear(lin_output_size, lin_hidden_size)
        self.lin_2 = torch.nn.Linear(lin_hidden_size, output_size)

    def forward(self, latent_input):
        decoded_output, (hidden, cell) = self.lstm(latent_input)
        hidden = self.lin_1(decoded_output)
        hidden = self.relu(hidden)
        hidden = self.lin_2(hidden)
        hidden_disp, hidden_cat = hidden[:,:,:2], self.sig(hidden[:,:,2:])
        return hidden_disp, hidden_cat

class PredictLSTM(torch.nn.Module):
    def __init__(self, predict_size, lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(PredictLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.predict_size = predict_size
        self.num_layers = num_layers

        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.lstm = torch.nn.LSTM(hidden_size, lin_output_size, num_layers)
        self.lin_1 = torch.nn.Linear(lin_output_size, lin_hidden_size)
        self.lin_2 = torch.nn.Linear(lin_hidden_size, predict_size)

    def forward(self, encoded_input):
        decoded_output, (hidden, cell) = self.lstm(encoded_input)
        hidden = self.lin_1(decoded_output)
        hidden = self.relu(hidden)
        hidden = self.lin_2(hidden)
        hidden_disp, hidden_cat = hidden[:,:,:2], self.sig(hidden[:,:,2:])
        return hidden_disp, hidden_cat

class LSTMAE(torch.nn.Module):
    def __init__(self, input_size, predict_size, lin_hidden_size, hidden_size, lin_output_size, num_layers, pred_len):
        super(LSTMAE, self).__init__()
        self.pred_len = pred_len
        self.encoder = EncoderLSTM(input_size, lin_hidden_size, hidden_size, lin_output_size, num_layers)
        self.decoder = DecoderLSTM(input_size, lin_hidden_size, hidden_size, lin_output_size, num_layers)
        self.predicter = PredictLSTM(predict_size, lin_hidden_size, hidden_size, lin_output_size, num_layers)

    def forward(self, input):
        latent_input = self.encoder(input)
        decoded_disp, decoded_cat = self.decoder(latent_input)
        predict_disp, predict_cat = self.predicter(latent_input[:,:self.pred_len])

        return decoded_disp, decoded_cat, predict_disp, predict_cat
