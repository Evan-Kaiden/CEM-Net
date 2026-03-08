import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm

import time

from utils import print_row
from losses import binary_loss, mask_overlap_loss, mask_area_loss, mask_tv_loss

criterion = nn.CrossEntropyLoss()

def train_one_epoch(epoch : int, model : nn.Module, trainloader : DataLoader, optimizer : Optimizer, scheduler, device=None):
    model.train()
    num_batches = len(trainloader)

    total_loss = 0.0
    metrics = {
        "ce": 0.0,
        "tv": 0.0,
        "binary": 0.0,
        "small_maps": 0.0,
        "large_maps": 0.0,
        "map_total": 0.0,
        "correct": 0,
        "total": 0,
    }

    lamb_bin = 2.0
    lamb_ce = 50.0
    lamb_tv = 0.05

    for images, targets in tqdm(trainloader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        
        logits, maps = model(images, return_maps=True)   
        ce_loss = criterion(logits, targets)
        bin_loss = binary_loss(maps)
        tv_loss = mask_tv_loss(maps)
 

        # scale loss terms
        eps = 1e-9
        step_loss = (lamb_ce * (ce_loss / (ce_loss.detach() + eps)) + 
                    lamb_bin * (bin_loss / (bin_loss.detach() + eps)) + 
                    lamb_tv * (tv_loss / (tv_loss.detach() + eps)))
        total_loss += lamb_ce * ce_loss.detach() + lamb_bin * bin_loss.detach() + lamb_tv * tv_loss.detach()
        step_loss.backward()
        optimizer.step()

        metrics["tv"] += lamb_tv * tv_loss.item()
        metrics["binary"] += lamb_bin * bin_loss.item()
        metrics["ce"] += ce_loss.item()
        metrics["correct"] += (logits.argmax(dim=-1) == targets).sum().item()
        metrics["total"] += targets.numel()
        metrics["small_maps"] += (maps < 0.1).sum().item()
        metrics["large_maps"] += (maps > 0.9).sum().item()
        metrics["map_total"] += maps.numel()
     

    avg_epoch_loss = total_loss / num_batches
    if scheduler is not None:
        try:
            scheduler.step(avg_epoch_loss)
        except TypeError:
            scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    avg_tv = metrics['tv'] / num_batches
    avg_ce = metrics["ce"] / num_batches
    avg_bin = metrics["binary"] / num_batches
    acc = metrics["correct"] / max(1, metrics["total"])
    small_map_avg = metrics["small_maps"] / metrics["map_total"]
    large_map_avg = metrics["large_maps"] / metrics["map_total"]
    return acc, avg_ce, avg_bin, avg_tv, lr, small_map_avg, large_map_avg

def test(args, epoch: int, model : nn.Module, testloader : DataLoader, device=None):
    model.eval()
    
    correct_total = 0
    loss_total = 0
    total = 0

    with torch.no_grad():
        total = 0

        for images, targets in testloader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)#, inference=True)

                loss = criterion(logits, targets)
                pred_labels = logits.argmax(dim=1)

                correct_total += (pred_labels == targets).sum().item()
                loss_total += loss.item()
                total += images.size(0)

    acc = correct_total / total
    loss = loss_total / len(testloader)

    return loss, acc

def train(epochs : int, model : nn.Module, trainloader : DataLoader, testloader: DataLoader, optimizer : Optimizer, scheduler, config, start_epoch=0, device=None):  

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_acc, avg_ce, avg_bin, avg_tv, lr, small_map_avg, large_map_avg = train_one_epoch(epoch, model, trainloader, optimizer, scheduler, device)
        test_loss, test_acc = test(config, epoch, model, testloader, device)
        end_time = time.time()
        epoch_time = int(end_time - start_time)
        metrics = [train_acc, test_acc, avg_ce, avg_bin, avg_tv, small_map_avg, large_map_avg, lr, epoch_time]
        print_row(epoch, metrics, config['run_dir'])
        
        state_dict = scheduler.state_dict() if scheduler is not None else None
        state = {                
                "epoch": epoch + 1,
                "test_loss" : test_loss,
                "test_acc" : test_acc * 100,
                "model_state": model.state_dict(),
                # "optimizer_state": optimizer.state_dict(),
                # "scheduler_state": state_dict,
                "config": config,
                }
        torch.save(state, os.path.join(os.path.curdir, config["run_dir"], f"state.pth"))
