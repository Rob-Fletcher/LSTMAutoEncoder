import torch
import torch.nn as nn
from matplotlib import pyplot as plt

def init_weights(m):
    for name, param in m.named_parameters():
        param.requires_grad = True
        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
            #torch.nn.init.xavier_normal_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
            #torch.nn.init.xavier_normal_(param.data, gain=20)
        elif 'bias' in name:
            param.data.fill_(0)


def drawValidation(target, recon, file_name):
    fig, ax = plt.subplots(1,2, figsize=(14,5))
    ax[0].plot(target[:,0].data.numpy(), target[:,1].data.numpy(), '-')
    ax[1].plot(recon[:,0].data.numpy(), recon[:,1].data.numpy(), '-', c='r')
    plt.savefig(file_name)
    return file_name
