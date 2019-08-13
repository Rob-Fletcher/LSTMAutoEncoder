import torch
import torch.nn as nn
from matplotlib import pyplot as plt

plt.ioff()

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
        elif name=='weight':
            # torch.nn.init.xavier_normal_(param.data)
            torch.nn.init.xavier_uniform_(param.data)


def drawValidation(targets, recons, file_name):
    fig, ax = plt.subplots(10,3, figsize=(20,60))
    for t, r, a in zip(targets, recons, ax):
        a[0].plot(t[:,0].data.numpy(), '-')
        a[1].plot(r[:,0].data.numpy(), '-', c='r')
        a[2].plot(t[:,0].data.numpy(), '-')
        a[2].plot(r[:,0].data.numpy(), '-', c='r')
        a[0].set_title(f"Input Sequence [{t[0,2]},{t[0,3]},{t[0,4]}]")
        a[1].set_title("Output Sequence")
        a[2].set_title("Input/Output Sequences")
    plt.savefig(file_name)
    return file_name
