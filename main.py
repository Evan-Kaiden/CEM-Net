import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch

from train import train
from cem_model import CEMModelWrapper
from data import get_dataloader

import argparse
import utils
from mapping import map_arg

import os
import json
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50',
                                                                         'resnet101', 'resnet152', 'vgg11', 
                                                                         'vgg16', 'vgg19', 'mobilenetv2',
                                                                        ])

parser.add_argument('--lamb_ce', type=int, default=100)
parser.add_argument('--lamb_tv', type=int, default=0.05)
parser.add_argument('--lamb_bin', type=int, default=1)


parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'rmsprop', 'sgd'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_scheduler', type=str, default='none', choices=['cosine', 'linear', 'step', 'none'])
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--run_dir', type=str)
args = parser.parse_args()


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"

if args.run_dir is None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir
    print(f"saving to {args.run_dir}")
backbone = map_arg[args.backbone].to(device)
m = CEMModelWrapper(backbone, 10, 32, device=device).to(device)
opt = map_arg[args.optimizer](m.parameters(), lr=args.lr)
scheduler = utils.get_scheduler(map_arg, opt, args.lr_scheduler, args.epochs, args.lr)
dset = get_dataloader(args.dataset, args.batch_size)

config = vars(args)

start_epoch = 0

state_path = os.path.join(args.run_dir, "state.pth")



config.update({
    "device": device,
    "num_classes": 10,
    "epoch": start_epoch,
    "optimizer": args.optimizer,
    "lr": args.lr,
    "epochs": args.epochs,
    "lr_scheduler": args.lr_scheduler,
    "backbone": args.backbone,
})

os.makedirs(args.run_dir, exist_ok=True)

with open(os.path.join(args.run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# ---- create fresh state file ----
torch.save({
    "epoch": 0,
    "test_loss": 0,
    "test_acc": 0,
    "model_state": None,
    "optimizer_state": None,
    "scheduler_state": None,
    "config": config,
}, state_path)

print(f"Training\n  Model {args.backbone}\n  Epochs {args.epochs}\n  Device {device}")

train(
    epochs=args.epochs,
    model=m,
    trainloader=dset.train_loader,
    testloader=dset.test_loader,
    optimizer=opt,
    scheduler=scheduler,
    config=config,
    start_epoch=start_epoch,
    device=device
)

