import torch

class PartitionLoss(torch.nn.Module):
    """ Loss funtion described by the paper https://arxiv.org/pdf/1509.05982.pdf

    This is used to attempt to partition the latent space of an autoencoder into
    signal and background regions.
    """
    def __init__(self, lmbda=99):
        super(PartitionLoss, self).__init__()
        self.lmbda = lmbda

    def forward(self, weaklabels, latent, C):
        mask = (C * latent)**2
        loss = (self.lmbda/C.mean()) * weaklabels * mask
        return torch.mean(loss)
