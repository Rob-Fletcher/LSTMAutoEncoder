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

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    mse_loss = torch.nn.MSELoss()

    #writer.add_graph(model)

    "+ Starting Training Loop"
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0
        for batch_i, paths in enumerate(dataloader):
            print("Starting batch...")
            batches_done = len(dataloader)*epoch + batch_i

            paths = Variable(paths.to(device))
            outpaths = model(paths)
            loss = mse_loss(paths, outpaths)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss
            if batch_i%10 == 0:
                writer.add_scalar("loss", epoch_loss/float(batches_done), epoch )

        print(f"Epoch: {epoch}   loss: {epoch_loss/float(args.batch_size)}")
        if epoch%10==0:
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
