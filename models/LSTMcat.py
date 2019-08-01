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

        #Encoder Branch
        self.relu = torch.nn.ReLU()
        self.lin_1 = torch.nn.Linear(input_size, lin_hidden_size)
        self.lin_2 = torch.nn.Linear(lin_hidden_size, lin_output_size)
        self.lstm = torch.nn.LSTM(lin_output_size, hidden_size, num_layers)

    def forward(self, input):
        hidden = self.lin_1(input)
        hidden = self.relu(hidden)
        hidden = self.lin_2(hidden)
        encoded_input, hidden = self.lstm(hidden)

        return encoded_input

class EncodeCat(torch.nn.Module):
    def __init__(self, hidden_size, cat_dim):
        super(EncodeCat, self).__init__()
        self.hidden_size = hidden_size
        self.cat_dim = cat_dim

        self.cat_lin_1 = torch.nn.Linear(hidden_size, 64)
        self.cat_relu = torch.nn.ReLU()
        self.cat_lin_2 = torch.nn.Linear(64, cat_dim)
        self.cat_SM = torch.nn.Softmax()

    def forward(self, encoded_input):

        encoded_cat = self.cat_lin_1(encoded_input)
        encoded_cat = self.cat_relu(encoded_cat)
        encoded_cat = self.cat_lin_2(encoded_cat)
        encoded_cat = self.cat_SM(encoded_cat)

        return encoded_cat

class EncodeNorm(torch.nn.Module):
    def __init__(self):
        super(EncodeNorm, self).__init__()

    def forward(self):

        return

class DecoderLSTM(torch.nn.Module):
    def __init__(self, output_size, lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.relu = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(hidden_size, lin_output_size, num_layers)
        self.lin_1 = torch.nn.Linear(lin_output_size, lin_hidden_size)
        self.lin_2 = torch.nn.Linear(lin_hidden_size, output_size)

    def forward(self, encoded_input):
        decoded_output, hidden = self.lstm(encoded_input)
        hidden = self.lin_1(decoded_output)
        hidden = self.relu(hidden)
        output = self.lin_2(hidden)
        return output

class PredictLSTM(torch.nn.Module):
    def __init__(self, predict_size, lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(PredictLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.predict_size = predict_size
        self.num_layers = num_layers

        self.relu = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(hidden_size, lin_output_size, num_layers)
        self.lin_1 = torch.nn.Linear(lin_output_size, lin_hidden_size)
        self.lin_2 = torch.nn.Linear(lin_hidden_size, predict_size)

    def forward(self, encoded_input):
        decoded_output, hidden = self.lstm(encoded_input)
        hidden = self.lin_1(decoded_output)
        hidden = self.relu(hidden)
        output = self.lin_2(hidden)
        return output

class DiscCat(torch.nn.Module):
    def __init__(self, cat_size):
        super(DiscCat, self).__init__()
        self.cat_size

    def forward(self, cat_input):
        return

class DiscNormal(torch.nn.Module):
    def __init__(self, cat_size):
        super(DiscNormal, self).__init__()
        self.cat_size

    def forward(self, cat_input):
        return

class LSTMAE(torch.nn.Module):
    def __init__(self, input_size, predict_size, lin_hidden_size, hidden_size, lin_output_size, num_layers):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderLSTM(input_size, lin_hidden_size, hidden_size, lin_output_size, num_layers)
        self.decoder = DecoderLSTM(input_size, lin_hidden_size, hidden_size, lin_output_size, num_layers)
        self.predicter = PredictLSTM(predict_size, lin_hidden_size, hidden_size, lin_output_size, num_layers)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        predict_output = self.predicter(encoded_input)

        return decoded_output, predict_output
