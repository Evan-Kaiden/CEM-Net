import os
import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import print_row
from viz import plot_masks_together, plot_attention_only
from losses import ce_loss_func, tv_loss_func, attention_alignment_loss, laplacian_smoothness_loss, peak_spread_loss, topk_peak_loss, attn_distribution_loss



def train_one_epoch(args, epoch, model, trainloader, optimizer, scheduler, device, pretrain):
    model.train()
    num_batches = len(trainloader)
    total_loss  = 0.0

    if pretrain:
        metrics = {"ce": 0.0, "tv": 0.0, "active": 0.0, "topk": 0.0,
                   "bg_percent": 0.0, "correct": 0, "total": 0}
    else:
        metrics = {"ce": 0.0, "alignment": 0.0,
                   "bg_percent": 0.0, "correct": 0, "total": 0}

    for images, targets in tqdm(trainloader, leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        if pretrain:
            logits, attn = model(images, train_attention=True)

            ce = ce_loss_func(logits, targets)
            peak_loss = topk_peak_loss(attn)
            tv_loss = tv_loss_func(attn)
            # peak_spread = peak_spread_loss(attn)
            # dist_loss = attn_distribution_loss(attn, attn.device)

            loss = (
                  args["lamb_ce"]     * ce
                + args["lamb_wass"]   * attn_distribution_loss(attn, attn.device)
                + args["lamb_active"] * attn.mean()
                + args["lamb_peak"]   * topk_peak_loss(attn)
                + args["lamb_tv"]     * laplacian_smoothness_loss(attn)
                + args["lamb_spread"] * peak_spread_loss(attn)
                )

            metrics["ce"] += args["lamb_ce"] * ce.item()
            metrics["active"] += args["lamb_active"] * attn.mean().item()
            metrics["topk"] += args["lamb_peak"] * peak_loss.item()
            metrics["tv"] += args["lamb_tv"] * tv_loss.item()

        else:
            logits, maps, attn = model(images, return_maps=True)

            ce = ce_loss_func(logits, targets)
            attention_alignment = attention_alignment_loss(maps, attn, targets)

            loss = (args["lamb_ce"] * ce
                  + args["lamb_alignment"] * attention_alignment)

            metrics["ce"] += args["lamb_ce"] * ce.item()
            metrics["alignment"] += args["lamb_alignment"] * attention_alignment.item()
            metrics["bg_percent"] += (maps.argmax(dim=1) == maps.shape[1] - 1).float().mean().item()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        metrics["correct"] += (logits.argmax(dim=-1) == targets).sum().item()
        metrics["total"] += targets.numel()

    if scheduler is not None:
        try: scheduler.step(total_loss / num_batches)
        except: scheduler.step()

    n = num_batches
    acc = metrics["correct"] / max(1, metrics["total"])
    lr = optimizer.param_groups[0]["lr"]

    if pretrain:
        return (acc,
                metrics["ce"] / n,
                metrics["tv"] / n,
                metrics["active"] / n,
                metrics["topk"] / n,
                lr)
    else:
        return (acc,
                metrics["ce"] / n,
                metrics["alignment"] / n,
                metrics["bg_percent"] / n,
                lr)




def test(args, epoch, model, testloader, dset, device, pretrain=False):
    model.eval()
    correct_total = loss_total = total = 0

    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images, train_attention=True)[0] if pretrain else model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss_total += ce_loss_func(logits, targets).item()
            correct_total += (logits.argmax(dim=1) == targets).sum().item()
            total += images.size(0)

    with torch.no_grad():
        for i in range(min(10, images.size(0))):
            if pretrain:
                _, attn = model(images[i:i+1], train_attention=True)
                plot_attention_only(attn.squeeze(0), images[i], dset,
                                    args["run_dir"], f"pretrain_{epoch}", f"attn_{i}.png")
            else:
                _, maps, attn = model(images[i:i+1], return_maps=True)
                plot_masks_together(maps.squeeze(0), attn.squeeze(0), images[i],
                                    dset, args["run_dir"], epoch, f"{i}.png")

    return loss_total / len(testloader), correct_total / total


def _save(model, scheduler, config, epoch, test_loss, test_acc):
    torch.save({
        "epoch": epoch + 1,
        "test_loss": test_loss,
        "test_acc": test_acc * 100,
        "model_state": model.state_dict(),
        "config": config,
    }, os.path.join(config["run_dir"], "state.pth"))


def _run_epoch(tag, pretrain, epoch, model, trainloader, testloader,
               optimizer, scheduler, dset, config, device):
    start = time.time()

    if pretrain:
        acc, avg_ce, avg_tv, avg_active, avg_topk, lr = train_one_epoch(
            config, epoch, model, trainloader, optimizer, scheduler, device, pretrain
        )
    else:
        acc, avg_ce, avg_alignment, avg_bg, lr = train_one_epoch(
            config, epoch, model, trainloader, optimizer, scheduler, device, pretrain
        )

    test_loss, test_acc = test(config, epoch, model, testloader, dset, device, pretrain)
    epoch_time = int(time.time() - start)

    if pretrain:
        names = ('train acc %', 'test acc %', 'ce', 'tv', 'active', 'topk', 'lr', 'epoch time')
        vals = (acc * 100, test_acc * 100, avg_ce, avg_tv, avg_active, avg_topk, lr, epoch_time)
    else:
        names = ('train acc %', 'test acc %', 'ce', 'alignment', 'bg %', 'lr', 'epoch time')
        vals = (acc * 100, test_acc * 100, avg_ce, avg_alignment, avg_bg * 100, lr, epoch_time)

    print_row(f"{tag}{epoch}", vals, names, config["run_dir"], print_header=(epoch == 0))
    _save(model, scheduler, config, epoch, test_loss, test_acc)

def train(pretrain_epochs, epochs, model, optimizer, dset, scheduler, config, start_epoch=0, device=None):
    trainloader = dset.train_loader
    testloader  = dset.test_loader

    for param in model.backbone.parameters():
        param.requires_grad = True

    for epoch in range(pretrain_epochs):
        _run_epoch("pretrain-", True, epoch, model, trainloader, testloader,
                   optimizer, scheduler, dset, config, device)

    # freeze attention head before main training
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.attn_head.parameters():
        param.requires_grad = False

    # rebuild optimizer over only trainable params before main phase
    optimizer = type(optimizer)(
        filter(lambda p: p.requires_grad, model.parameters()),
        **{k: v for k, v in optimizer.defaults.items() if k != 'params'}
    )
    for epoch in range(start_epoch, epochs):
        _run_epoch("", False, epoch, model, trainloader, testloader,
                   optimizer, scheduler, dset, config, device)
        

