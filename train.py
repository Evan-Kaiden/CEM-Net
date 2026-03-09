import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm

import time

from utils import print_row
from viz import plot_masks_together
from losses import ce_loss_func, fg_bg_contrast_loss_func, attn_sparsity_loss_func, attn_entropy_loss_func, tv_loss_func


def train_one_epoch(args, epoch : int, model : nn.Module, trainloader : DataLoader, optimizer : Optimizer, scheduler, device=None):
    model.train()
    num_batches = len(trainloader)

    total_loss = 0.0
    metrics = {
        "ce": 0.0,
        "tv": 0.0,
        "contrast": 0.0,
        "sparsity": 0.0,
        "entropy": 0.0,
        "abstained": 0.0,
        "correct": 0,
        "total": 0,
    }

    
    lamb_ce = args["lamb_ce"]
    lamb_tv = args["lamb_tv"]
    lamb_entropy = args["lamb_entropy"]
    lamb_contrast = args["lamb_contrast"]
    lamb_sparsity = args["lamb_sparsity"]
    

    for images, targets in tqdm(trainloader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        
        logits, maps, attn = model(images, return_maps=True)   
        logits = logits[:, :-1]
        abstained = maps[:, -1, :, :]
        
        ce_loss =  ce_loss_func(logits, targets)
        contrast_loss =  fg_bg_contrast_loss_func(maps, attn)
        sparsity_loss =  attn_sparsity_loss_func(attn, target_coverage=0.25)
        entropy_loss =  attn_entropy_loss_func(attn)
        tv_loss =  tv_loss_func(maps)


 
        step_loss = lamb_ce * ce_loss + \
                    lamb_tv * tv_loss + \
                    lamb_sparsity * sparsity_loss + \
                    lamb_contrast * contrast_loss + \
                    lamb_entropy * entropy_loss 
                    
                    
        total_loss += step_loss
        step_loss.backward()
        optimizer.step()

        metrics["abstained"] += abstained.mean().item()
        metrics["ce"] += ce_loss.item()
        metrics["tv"] += lamb_tv * tv_loss.item()
        metrics["sparsity"] += lamb_sparsity * sparsity_loss.item()
        metrics["constrast"] += lamb_contrast * contrast_loss.item()
        metrics["entropy"] += lamb_entropy * entropy_loss.item()
        metrics["correct"] += (logits.argmax(dim=-1) == targets).sum().item()
        metrics["total"] += targets.numel()
     

    avg_epoch_loss = total_loss / num_batches
    if scheduler is not None:
        try:
            scheduler.step(avg_epoch_loss)
        except TypeError:
            scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    avg_abstain = metrics["abstained"] / num_batches
    avg_ce = metrics["ce"] / num_batches
    avg_tv = metrics['tv'] / num_batches
    avg_sparsity = metrics["sparsity"] / num_batches
    avg_contrast = metrics["constrast"] / num_batches
    avg_entropy = metrics["entropy"] / num_batches
    
    
    acc = metrics["correct"] / max(1, metrics["total"])
    
    return acc, avg_ce, avg_tv, avg_sparsity, avg_contrast, avg_entropy, lr, avg_abstain

def test(args, epoch: int, model : nn.Module, testloader : DataLoader, dset, device=None):
    model.eval()
    
    correct_total = 0
    loss_total = 0
    total = 0

    with torch.no_grad():
        total = 0

        for images, targets in testloader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                logits = logits[:, :-1]
                loss = ce_loss_func(logits, targets)
                pred_labels = logits.argmax(dim=1)

                correct_total += (pred_labels == targets).sum().item()
                loss_total += loss.item()
                total += images.size(0)

    acc = correct_total / total
    loss = loss_total / len(testloader)
    # visualize 10 of the last batch
    for i in range(10):
        image = images[i:i+1]
        _, maps, attn = model(image, return_maps=True)
        plot_masks_together(maps.squeeze(0), attn.squeeze(0), image.squeeze(0), dset, args["run_dir"], epoch, f"{i}.png")
    return loss, acc

def train(epochs : int, model : nn.Module, optimizer : Optimizer, dset, scheduler, config, start_epoch=0, device=None):  
    trainloader = dset.train_loader
    testloader = dset.test_loader

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_acc, avg_ce, avg_tv, avg_sparsity, avg_contrast, avg_entropy, lr, avg_abstain = train_one_epoch(config, epoch, model, trainloader, optimizer, scheduler, device)
        test_loss, test_acc = test(config, epoch, model, testloader, dset, device)
        end_time = time.time()
        epoch_time = int(end_time - start_time)
        metrics = [train_acc, test_acc, avg_ce, avg_tv, avg_sparsity, avg_contrast, avg_entropy, avg_abstain, lr, epoch_time]
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
