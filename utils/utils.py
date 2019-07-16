import torch
import torch.nn as nn

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.xavier_normal_(param.data, gain=20)
        elif 'weight_hh' in name:
            #torch.nn.init.orthogonal_(param.data)
            torch.nn.init.xavier_normal_(param.data, gain=20)
        elif 'bias' in name:
            param.data.fill_(0)
