from __future__ import division
from model.LSTMAE import *
from utils.datasets import *
from utils.utils import *

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

def train(args):

    #Tensorboard writer
    writer = SummaryWriter()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    input_size = 2
    model = LSTMAE(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, isCuda=torch.cuda.is_available()).to(device)
    model.apply(init_weights)
    print("+ Model Loaded.")

    if args.weights:
        model.load_state_dict(torch.load(args.weights))

    dataset = PathData(args.train_data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu
    )

    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    mse_loss = torch.nn.MSELoss()
    #CE_loss = torch.nn.CrossEntropyLoss()

    print("+ Starting Training Loop")
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0
        for batch_i, paths in enumerate(dataloader):
            batches_done = len(dataloader)*epoch + batch_i

            paths = Variable(paths.to(device))
            outpaths = model(paths)
            loss = mse_loss(paths, outpaths)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar(tag="loss/MSE Loss", scalar_value=loss, global_step=int(batches_done) )
            #print(f"loss {loss}, seq_len {paths.shape}   batches_done  {batches_done}")
            epoch_loss += loss

        print(f"Epoch: {epoch} loss:  {epoch_loss/float(len(dataloader))}   Time: {time.time()-start_time}")
        writer.add_scalar(tag="loss/epoch loss", scalar_value=epoch_loss/float(len(dataloader)), global_step=int(epoch))
        if epoch%5==4:
            torch.save(model.state_dict(), f"checkpoints/lstmAE_ckpt_epoch_{epoch}.pth")

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--weights", type=str, help="Path to weights to continue training.")
    parser.add_argument("--n_cpu", type=int, default=0, help="")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data")
    parser.add_argument("--valid_data", type=str, help="Path to the validation data")
    parser.add_argument("--hidden_size", type=int, default=20, help="dimension of hidden/encoded size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the encoder/decoder LSTM")
    args = parser.parse_args()

    train(args)
