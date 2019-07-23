from __future__ import division
from model.LSTMAE import *
from utils.datasets import *
from utils.utils import *

import argparse
import os
import time
import random

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import mlflow

def train(args):

    mlflow.set_experiment("LSTM Autoencoder")
    mlflow.start_run(run_name=args.tag)
    params = vars(args)
    for par in params:
        mlflow.log_param(par, params[par])

    #Tensorboard writer
    writer = SummaryWriter()
    if args.n_cpu > 0:
        print("Pytables is currently not thread safe. Setting n_cpu to 0.")
    args.n_cpu = 0

    vconfig = None
    try:
        with open(args.geo) as vcf:
            vconfig = json.load(vcf)
    except:
        pass


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    input_size = 2
    model = LSTMAE(input_size=input_size,lin_hidden_size=20, hidden_size=args.hidden_size, lin_output_size=30, num_layers=1).to(device)
    model.apply(init_weights)
    print("+ Model Loaded.")

    start_epoch = 1
    if args.weights:
        print(f"Loading model weights from {args.weights}")
        start_epoch = args.weights.split('_')[-1]
        start_epoch = int(start_epoch.split('.')[0])+1 #continue from the epoch we left off at
        print(f"Continuing training from epoch {start_epoch}")
        model.load_state_dict(torch.load(args.weights))

    mlflow.log_param("Start Epoch", start_epoch)
    model.train()

    dataset = PathData(args.train_data, sequence_length=args.seq_len)
    #valid_data = PathData(args.valid_data, sequence_length=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu
    )

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    loss_func = torch.nn.MSELoss()

    writer.add_graph(model=model, input_to_model=next(iter(dataloader)), verbose=True)

    print("+ Starting Training Loop")
    for epoch in range(start_epoch,start_epoch+args.epochs):
        start_time = time.time()
        epoch_loss = 0
        for batch_i, paths in enumerate(dataloader):
            #batches_done = len(dataloader)*epoch + batch_i

            optimizer.zero_grad()

            # Run the model for the current batch and get loss
            paths = paths.to(device)
            outpaths = model(paths)
            loss = loss_func(outpaths, paths)

            # Backprop the loss function and step the optimizer
            loss.backward()
            optimizer.step()

            #Every 10 epochs pick a path and draw it next to its reconstruction
            #then log this as an artifact in mlflow
            #TODO: Make this work with validation data
            if epoch%10==1 and batch_i == 0:
                print("Generating validation image...")
                #bp = random.randint(0, len(dataloader))
                bp = 10
                img_name = drawValidation(paths[[bp, bp+2, bp+4]].cpu(), outpaths[[bp, bp+2, bp+4]].cpu(), f"output/Valid_img_epoch_{epoch}.png")
                mlflow.log_artifact(img_name)


            # writer.add_scalar(tag="loss/MSE Loss", scalar_value=loss, global_step=int(batches_done) )
            #print(f"loss {loss.item()}, batches_done  {batches_done}")
            epoch_loss += loss.item()

        print(f"Epoch: {epoch} loss:  {epoch_loss/float(len(dataloader))}   Time: {time.time()-start_time}")
        mlflow.log_metric(key="Epoch Loss", value=epoch_loss/float(len(dataloader)), step=int(epoch))
        # writer.add_scalar(tag="loss/epoch loss", scalar_value=epoch_loss/float(len(dataloader)), global_step=int(epoch))
        if epoch%10==0 or epoch==args.epochs:
            print("Saving checkpoint")
            torch.save(model.state_dict(), f"checkpoints/lstmAE_ckpt_epoch_{epoch}.pth")

    writer.close()
    torch.save(model.state_dict(), f"checkpoints/lstmAE_ckpt_final.pth")
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--weights", type=str, help="Path to weights to continue training.")
    parser.add_argument("--n_cpu", type=int, default=0, help="")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data")
    parser.add_argument("--valid_data", type=str, help="Path to the validation data")
    parser.add_argument("--hidden_size", type=int, default=100, help="dimension of hidden/encoded size")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length to train on.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--tag", type=str, default='run', help="A tag to help identify the run in MLFlow")
    args = parser.parse_args()

    train(args)
