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

parser.add_argument('--lamb_ce', type=float, default=1.)
parser.add_argument('--lamb_tv', type=float, default=0.01)
parser.add_argument('--lamb_peak', type=float, default=0.125)
parser.add_argument('--lamb_active', type=float, default=0.15)
parser.add_argument('--lamb_wass', type=float, default=0.1)
parser.add_argument('--lamb_spread', type=float, default=0.1)


parser.add_argument('--lamb_alignment', type=float, default=0.1)



parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'rmsprop', 'sgd'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_scheduler', type=str, default='none', choices=['cosine', 'linear', 'step', 'none'])
parser.add_argument('--pretrain_epochs', type=int, default=20)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'stl10'])
parser.add_argument('--run_dir', type=str)
parser.add_argument('--pretrained', action='store_true', default=True)
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


dset = get_dataloader(args.dataset, args.batch_size)

img_size = next(iter(dset.train_loader))[0].size(-1)
num_class = len(dset.classes)
backbone = map_arg[args.backbone](pretrained=args.pretrained).to(device)
m = CEMModelWrapper(backbone, num_class, img_size, device=device, skip_layer_names=['5', '4', '3']).to(device)
opt = map_arg[args.optimizer](m.parameters(), lr=args.lr)
scheduler = utils.get_scheduler(map_arg, opt, args.lr_scheduler, args.epochs, args.lr)


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

# 2. confirm feature map size coming out of backbone
with torch.no_grad():
    dummy = torch.zeros(1, 3, img_size, img_size).to(device)
    features = m.backbone(dummy).squeeze(0).shape
    print(f"  Input Shape to CEM {[f for f in features]}")
train(
    pretrain_epochs=args.pretrain_epochs,
    epochs=args.epochs,
    model=m,
    dset=dset,
    optimizer=opt,
    scheduler=scheduler,
    config=config,
    start_epoch=start_epoch,
    device=device
)


