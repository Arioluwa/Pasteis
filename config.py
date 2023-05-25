from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import SITSDataset
from vicreg import VICReg, VICLoss
import augmentation as aug
from optimization import optim , adjust_learning_rate
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from backbones.utae import UTAE


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a u_ltae model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir",  help='Path to dataset')

    # Checkpoints
    parser.add_argument("--save-freq", type=int, default=1000, help='Print logs to the stats.txt file every [save-freq-] seconds')
    parser.add_argument('--log_dir', default= r'logs', help = 'directory to save logs')
    parser.add_argument('--save_chpt', default = 'checkpoints', help = 'path to save checkpoints')

    # Model
    #parser.add_argument("--projector-dim", default=[32, 64, 128, 256], help='Dimension of Projector / Expander')

    # Optim
    parser.add_argument("--epochs", type=int, default=1,help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default= 1, help='Batch Size')
    parser.add_argument("--lr", type=float, default=0.2,help='learning rate')
    parser.add_argument("--weight-decay", type=float, default=1e-6,help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')

    return parser


def main(args):
    print(args)
    print("Training Starts")
    gpu = torch.device(args.device)
    writer = SummaryWriter(log_dir=args.log_dir)

    transforms = aug.TrainTransform()
    dataset = SITSDataset(args.data_dir, norm=True, transform = transforms, year= ['2018','2019'])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle = True 
    )

    model = VICReg().cuda(gpu)
    optimizer = optim(model, args.weight_decay)
    os.makedirs(os.path.join('.', args.save_chpt), exist_ok=True)

    for epoch in range(0, args.epochs):
        loop = tqdm(enumerate(loader, start=epoch * len(loader)), total=len(loader), leave=False)
        for step, (view_a, view_b, dates) in loop:
            adjust_learning_rate(args, optimizer,loader, step)
            optimizer.zero_grad()
            view_a = view_a.to(args.device, dtype=torch.float)
            view_b = view_b.to(args.device, dtype=torch.float)
            d1, d2 = dates
            d1 = d1.to(args.device)
            d2 = d2.to(args.device)
            repr1, repr2 = model(view_a, d1, view_b, d2)
            
            loss, all_losses = VICLoss()(repr1, repr2)
            repr_loss, std_loss, cov_loss = all_losses
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Loss/sim", repr_loss, epoch)
            writer.add_scalar("Loss/std", std_loss, epoch)
            writer.add_scalar("Loss/cov", cov_loss, epoch)

            if step % int(args.save_freq) == 0 and step:
                with open(os.path.join(args.log_dir, 'logs.txt'), 'a') as log_file:
                    log_file.write(f'Epoch: {epochs}, Step: {step}, Train loss: {loss.cpu().detach().numpy()}, Sim loss: {repr_loss.cpu().detach().numpy()}, Std loss: {std_loss.cpu().detach().numpy()}, Cov loss: {cov_loss.cpu().detach().numpy()}\n')


            loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(loss = loss.cpu().detach().numpy())

        print(f'Loss for epoch {epoch} is {loss.cpu().detach().numpy()}')
    print('End of the Training. Saving final checkpoints.')
    state = dict(epoch=args.epochs, model=model.state_dict(), optimizer=optimizer.state_dict())
    torch.save(state, os.path.join('.', args.save_chpt,  'final_checkpoint.pth'))
    writer.flush()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)

