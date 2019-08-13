from __future__ import division
from models.LSTMPredict import *
from utils.datasetNP import *
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
# import mlflow

def train(args):

    # mlflow.set_experiment("LSTM Autoencoder")
    # mlflow.start_run(run_name=args.tag)
    # params = vars(args)
    # for par in params:
    #     mlflow.log_param(par, params[par])

    #Tensorboard writer
    writer = SummaryWriter(comment='_'+args.tag.replace(' ', '_'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    input_size = 5
    model = LSTMAE(input_size=input_size,predict_size=input_size,lin_hidden_size=20, hidden_size=args.hidden_size, lin_output_size=30, num_layers=1, pred_len=args.pred_len).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters())
    print("+ Model Loaded.")

    start_epoch = 1
    if args.weights:
        print(f"Loading model weights from {args.weights}")
        checkpoint = torch.load(args.weights)
        start_epoch = checkpoint['epoch']+1  # continue from the next epoch
        print(f"Continuing training from epoch {start_epoch}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    # mlflow.log_param("Start Epoch", start_epoch)
    model.train()

    dataset = PathDataNP(args.train_data, seq_len=args.seq_len, pred_len=args.pred_len)
    #valid_data = PathData(args.valid_data, sequence_length=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu
    )
    mse_loss = torch.nn.MSELoss()
    # bce_loss = torch.nn.BCELoss()
    # criterion = torch.nn.L1Loss()

    # writer.add_graph(model=model, input_to_model=next(iter(dataloader)))

    print("+ Starting Training Loop")
    for epoch in range(start_epoch,start_epoch+args.epochs):
        start_time = time.time()
        e_recon_loss = 0
        # e_recon_cat_loss = 0
        e_pred_loss = 0
        e_pred_cat_loss = 0
        e_total_loss = 0
        for batch_i, (paths, pred) in enumerate(dataloader):
            optimizer.zero_grad()

            # Run the model for the current batch and get loss
            model_s_time = time.time()
            paths = paths.to(device)
            recon, recon_cat, predict, predict_cat = model(paths)
            recon_loss = mse_loss(recon, paths[:,:,:2])
            # recon_cat_loss = bce_loss(recon_cat, paths[:,:,2:])
            pred_loss = mse_loss(predict, pred[:,:,:2])
            # pred_cat_loss = bce_loss(predict_cat, pred[:,:,2:])
            total_loss = recon_loss + pred_loss #+ recon_cat_loss + pred_cat_loss

            # Backprop the loss function and step the optimizer
            total_loss.backward()
            optimizer.step()
            model_time = time.time() - model_s_time


            #Every 10 epochs pick a path and draw it next to its reconstruction
            #then log this as an artifact in mlflow
            #TODO: Make this work with validation data
            if epoch%10==0 and batch_i == 0:
                print("Generating validation image...")
                bp = []
                for _ in range(10):
                    bp.append(random.randint(0, paths.shape[0]-1))
                recon_img = drawValidation(paths[bp].cpu(), recon[bp].cpu(), f"{args.output}/recon_img_epoch_{epoch}.png")
                pred_img = drawValidation(pred[bp].cpu(), predict[bp].cpu(), f"{args.output}/pred_img_epoch_{epoch}.png")
                # mlflow.log_artifact(recon_img)
                # mlflow.log_artifact(pred_img)


            # writer.add_scalar(tag="loss/MSE Loss", scalar_value=loss, global_step=int(batches_done) )
            #print(f"loss {loss.item()}, batches_done  {batches_done}")
            e_recon_loss += recon_loss.item()
            # e_recon_cat_loss += recon_cat_loss.item()
            e_pred_loss += pred_loss.item()
            #e_pred_cat_loss += pred_cat_loss.item()
            e_total_loss += total_loss.item()

        # mlflow.log_metric(key="Recon Loss", value=e_recon_loss/float(len(dataloader)), step=int(epoch))
        # mlflow.log_metric(key="pred_loss Loss", value=e_pred_loss/float(len(dataloader)), step=int(epoch))
        # mlflow.log_metric(key="total Loss", value=e_total_loss/float(len(dataloader)), step=int(epoch))
        writer.add_scalar(tag="loss/recon loss", scalar_value=e_recon_loss/float(len(dataloader)), global_step=int(epoch))
        # writer.add_scalar(tag="loss/recon cat loss", scalar_value=e_recon_cat_loss/float(len(dataloader)), global_step=int(epoch))
        writer.add_scalar(tag="loss/pred loss", scalar_value=e_pred_loss/float(len(dataloader)), global_step=int(epoch))
        # writer.add_scalar(tag="loss/pred cat loss", scalar_value=e_pred_cat_loss/float(len(dataloader)), global_step=int(epoch))
        writer.add_scalar(tag="loss/total loss", scalar_value=e_total_loss/float(len(dataloader)), global_step=int(epoch))
        if epoch%10==0 or epoch==args.epochs:
            print(f"Saving checkpoint for epoch {epoch}")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        },"checkpoints/lstmAE_ckpt.pth")
        print(f"Epoch: {epoch} total loss:  {e_total_loss/float(len(dataloader))}   Time: {.2f:time.time()-start_time}  Model Time: {.2f: model_time}")
        print(f"               recon loss:  {e_recon_loss/float(len(dataloader))}")
        print(f"               pred loss:   {e_pred_loss/float(len(dataloader))}")

    writer.close()
    torch.save(model.state_dict(), f"checkpoints/lstmAE_ckpt_epoch_{start_epoch+args.epochs}.pth")
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size.")
    parser.add_argument("--weights", type=str, help="Path to weights to continue training.")
    parser.add_argument("--n_cpu", type=int, default=0, help="")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data")
    parser.add_argument("--valid_data", type=str, help="Path to the validation data")
    parser.add_argument("--hidden_size", type=int, default=50, help="dimension of hidden/encoded size")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length to train on.")
    parser.add_argument("--pred_len", type=int, default=10, help="Sequence length to predict on.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--tag", type=str, default='run', help="A tag to help identify the run in MLFlow")
    parser.add_argument("--output", type=str, default='output', help='Output location for validation plots')
    args = parser.parse_args()

    train(args)
