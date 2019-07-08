from models import *
from utils.logger import *
from utils.datasets import *

import argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import mlflow

def train(args):

    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    model = LSTMAE(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, isCuda=device.is_cuda).to(device)

    if args.weights:
        model.load_state_dict(torch.load(args.weights))

    dataset = PathData(args.train_data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu
    )

    optimizer = torch.optim.Adam(model.parameters())
    mse_loss = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        for batch_i, paths in enumerate(dataloader):
            batches_done = len(dataloader)*epoch + batch_i

            paths = Variable(paths.to(device))
            outpaths = model(paths)
            loss = mse_loss(paths, outpaths)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logger.scalar_summary(tag="train loss", value=loss, step=batches_done)

        if epoch%10==0:
            torch.save(model.state_dict(), f"checkpoints/lstmAE_ckpt_epoch_{epoch}.pth")

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
    parser.add_argument("--weights", type=str, help="Path to weights to continue training.")
    parser.add_argument("--n_cpu", type=int, default=4, help="")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data")
    parser.add_argument("--valid_data", type=str, help="Path to the validation data")
    parser.add_argument("--hidden_size", type=int, default=20, help="dimension of hidden/encoded size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the encoder/decoder LSTM")
    args = parser.parse_args()

    train(args)
