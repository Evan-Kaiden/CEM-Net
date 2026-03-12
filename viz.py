import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_masks_together(mask, attn, image, dset, runs_dir, epoch, save_name):
    """
    mask:  (C+1, H, W) - entmax class evidence maps
    attn:  (1, H, W)   - spatial attention in (0,1)
    image: (3, H, W)   - normalized input image
    """
    class_id_to_name = {i: name for i, name in enumerate(dset.classes)}

    # --- detach and upscale everything to 224 for visibility
    SIZE = 224
    mask  = F.interpolate(mask.detach().unsqueeze(0),  size=SIZE, mode='bilinear', align_corners=False).squeeze(0)
    attn  = F.interpolate(attn.detach().unsqueeze(0),  size=SIZE, mode='bilinear', align_corners=False).squeeze(0)
    image = F.interpolate(image.detach().unsqueeze(0), size=SIZE, mode='bilinear', align_corners=False).squeeze(0)

    mask  = mask.cpu().numpy()   # (C+1, H, W)
    attn  = attn.cpu().numpy()   # (1, H, W)
    image = image.cpu().numpy()  # (3, H, W)

    image = np.transpose(image, (1, 2, 0))
    image = image * np.array(dset.std) + np.array(dset.mean)
    image = np.clip(image, 0, 1)

    attn_map   = attn[0]                          # (H, W), values in (0,1)
    argmax_map = np.argmax(mask, axis=0)          # (H, W), winner class per pixel
    fg_mask    = attn_map > np.median(attn_map)   # boolean fg region

    colors = plt.cm.tab10(np.linspace(0, 1, mask.shape[0]))
    num_classes = mask.shape[0] - 1               # exclude background

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: class map (what) ──────────────────────────────────────────
    ax = axes[0]
    ax.imshow(image)
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

    # ── Panel 2: attention map (where) ────────────────────────────────────
    ax = axes[1]
    ax.imshow(image)
    attn_overlay = ax.imshow(attn_map, cmap='hot', alpha=0.55, vmin=0, vmax=1)
    # draw the fg/bg threshold contour so you can see the split used in loss
    ax.contour(fg_mask.astype(float), levels=[0.5], colors=['cyan'],
               linewidths=1.5, linestyles='--')
    plt.colorbar(attn_overlay, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Attention Map (where)")
    ax.axis("off")

    # ── Panel 3: attended class map (what × where) ────────────────────────
    # This is the view that directly corresponds to your logit computation.
    # Only shows class assignments inside the attended region.
    ax = axes[2]
    ax.imshow(image)
    legend_handles = []
    for i in range(num_classes):
        # only show pixels where: this class wins AND attention is above median
        region = ((argmax_map == i) & fg_mask).astype(float)
        if np.any(region):
            # scale alpha by mean attention strength in this region for visual clarity
            mean_attn = attn_map[region.astype(bool)].mean() if region.sum() > 0 else 0.4
            ax.contourf(region, levels=[0.5, 1], colors=[colors[i]],
                        alpha=float(np.clip(mean_attn * 0.8, 0.2, 0.7)))
            ax.contour( region, levels=[0.5], colors=[colors[i]], linewidths=1.5)
            legend_handles.append(mpatches.Patch(color=colors[i], alpha=0.6,
                                                  label=class_id_to_name[i]))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.7)
    ax.set_title("Attended Evidence (what × where)")
    ax.axis("off")

    fig.suptitle(f"Epoch {epoch}", fontsize=11, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(runs_dir, f"figs/epoch{epoch}")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight', dpi=120)
    plt.close()


def plot_attention_only(attn, image, dset, runs_dir, epoch, save_name):
    SIZE = 224
    attn  = F.interpolate(attn.detach().unsqueeze(0),  size=SIZE, mode='bilinear', align_corners=False).squeeze(0)
    image = F.interpolate(image.detach().unsqueeze(0), size=SIZE, mode='bilinear', align_corners=False).squeeze(0)

    attn_map = attn.cpu().numpy()[0]         # (H, W)
    image_np = image.cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = image_np * np.array(dset.std) + np.array(dset.mean)
    image_np = np.clip(image_np, 0, 1)

    # Stats for title — quick health check without opening every image
    attn_std  = attn_map.std()
    attn_mean = attn_map.mean()
    attn_max  = attn_map.max()
    is_collapsed = attn_std < 0.05  # flag uniform maps

    fg_mask   = attn_map > np.median(attn_map)
    fg_pixels = image_np * attn_map[..., None]          # pixel values weighted by attention
    fg_pixels = np.clip(fg_pixels / (attn_map[..., None].max() + 1e-6), 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ── Panel 1: raw heatmap overlay ─────────────────────────────────────
    ax = axes[0]
    ax.imshow(image_np)
    overlay = ax.imshow(attn_map, cmap='hot', alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Heatmap  (std={attn_std:.3f}{'  ⚠ COLLAPSED' if is_collapsed else ''})",
                 color='red' if is_collapsed else 'black')
    ax.axis("off")

    # ── Panel 2: binary fg/bg split ──────────────────────────────────────
    ax = axes[1]
    ax.imshow(image_np)
    ax.imshow(fg_mask, cmap='cool', alpha=0.4, vmin=0, vmax=1)
    ax.contour(fg_mask.astype(float), levels=[0.5], colors=['cyan'], linewidths=1.5)
    ax.set_title("Foreground (above median)")
    ax.axis("off")

    # ── Panel 3: attention-weighted image (what the classifier actually sees)
    ax = axes[2]
    ax.imshow(fg_pixels)
    ax.set_title("Attended Pixels")
    ax.axis("off")

    # ── Panel 4: attention histogram (best single indicator of collapse)
    ax = axes[3]
    ax.hist(attn_map.ravel(), bins=50, color='steelblue', edgecolor='none')
    ax.axvline(attn_mean, color='red',    linestyle='--', linewidth=1.5, label=f'mean={attn_mean:.2f}')
    ax.axvline(attn_max,  color='orange', linestyle='--', linewidth=1.5, label=f'max={attn_max:.2f}')
    ax.set_xlim(0, 1)
    ax.set_title("Attention Distribution")
    ax.set_xlabel("Attention value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    fig.suptitle(f"Attention — Epoch {epoch}", fontsize=11, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(runs_dir, f"figs/epoch{epoch}")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight', dpi=120)
    plt.close()