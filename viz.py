import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from threading import Thread


def _plot_worker(mask_np, attn_np, image_np, class_id_to_name, mean, std, runs_dir, epoch, save_name):
    attn_map   = attn_np[0]
    argmax_map = np.argmax(mask_np, axis=0)
    fg_mask    = attn_map > np.median(attn_map)

    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = image_np * np.array(std) + np.array(mean)
    image_np = np.clip(image_np, 0, 1)

    colors      = plt.cm.tab10(np.linspace(0, 1, mask_np.shape[0]))
    num_classes = mask_np.shape[0] - 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: class map (what) ─────────────────────────────────────────
    ax = axes[0]
    ax.imshow(image_np)
    legend_handles = []
    for i in range(num_classes):
        region = (argmax_map == i).astype(float)
        if np.any(region):
            ax.contourf(region, levels=[0.5, 1], colors=[colors[i]], alpha=0.35)
            ax.contour( region, levels=[0.5],    colors=[colors[i]], linewidths=1.5)
            legend_handles.append(mpatches.Patch(color=colors[i], alpha=0.6,
                                                  label=class_id_to_name[i]))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.7)
    ax.set_title("Class Maps (what)")
    ax.axis("off")

    # ── Panel 2: attention map (where) ───────────────────────────────────
    ax = axes[1]
    ax.imshow(image_np)
    attn_overlay = ax.imshow(attn_map, cmap='hot', alpha=0.55, vmin=0, vmax=1)
    ax.contour(fg_mask.astype(float), levels=[0.5], colors=['cyan'],
               linewidths=1.5, linestyles='--')
    plt.colorbar(attn_overlay, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Attention Map (where)")
    ax.axis("off")

    # ── Panel 3: attended class map (what x where) ───────────────────────
    ax = axes[2]
    ax.imshow(image_np)
    legend_handles = []
    for i in range(num_classes):
        region = ((argmax_map == i) & fg_mask).astype(float)
        if np.any(region):
            mean_attn = attn_map[region.astype(bool)].mean() if region.sum() > 0 else 0.4
            ax.contourf(region, levels=[0.5, 1], colors=[colors[i]],
                        alpha=float(np.clip(mean_attn * 0.8, 0.2, 0.7)))
            ax.contour( region, levels=[0.5], colors=[colors[i]], linewidths=1.5)
            legend_handles.append(mpatches.Patch(color=colors[i], alpha=0.6,
                                                  label=class_id_to_name[i]))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.7)
    ax.set_title("Attended Evidence (what x where)")
    ax.axis("off")

    fig.suptitle(f"Epoch {epoch}", fontsize=11, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(runs_dir, f"figs/epoch{epoch}")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight', dpi=120)
    plt.close()


def plot_masks_together(mask, attn, image, dset, runs_dir, epoch, save_name):
    SIZE = 224
    # do all GPU/tensor ops on the calling thread before handing off
    mask  = F.interpolate(mask.detach().unsqueeze(0),  size=SIZE, mode='bilinear', align_corners=False).squeeze(0)
    attn  = F.interpolate(attn.detach().unsqueeze(0),  size=SIZE, mode='bilinear', align_corners=False).squeeze(0)
    image = F.interpolate(image.detach().unsqueeze(0), size=SIZE, mode='bilinear', align_corners=False).squeeze(0)

    # move to cpu numpy before spawning thread — no CUDA tensors across threads
    mask_np  = mask.cpu().numpy()
    attn_np  = attn.cpu().numpy()
    image_np = image.cpu().numpy()

    class_id_to_name = {i: name for i, name in enumerate(dset.classes)}

    t = Thread(
        target=_plot_worker,
        args=(mask_np, attn_np, image_np, class_id_to_name,
              dset.mean, dset.std, runs_dir, epoch, save_name),
        daemon=True   # won't block process exit if still running
    )
    t.start()
    # no t.join() — fire and forget, GPU continues immediately