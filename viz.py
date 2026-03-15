import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_masks_together(mask, attn, image, dset, runs_dir, epoch, save_name):
    """
    mask:  (C+1, H, W) - entmax class evidence maps (last channel = background)
    attn:  (1, H, W)   - spatial attention in (0,1)
    image: (3, H, W)   - normalized input image

    Layout: 4 rows x (C+1) cols
      Row 0: Global views (cols 0-3 only)
             [0] raw image  [1] FG/BG by last channel  [2] FG/BG by attn median  [3] argmax class map
      Row 1: Per-channel raw heatmap (one col per channel)
      Row 2: Per-channel heatmap overlaid on image
      Row 3: Per-channel FG/BG split (above-median pixels highlighted per channel)
    """
    # build index -> name, appending "Background" for the last channel
    class_id_to_name = {i: name for i, name in enumerate(dset.classes)}
    num_channels = mask.shape[0]          # C+1
    num_classes  = num_channels - 1       # C
    class_id_to_name[num_classes] = "Background"

    # ── upsample everything to 224 ──────────────────────────────────────
    SIZE = 224
    mask  = F.interpolate(mask.detach().unsqueeze(0),  size=SIZE, mode='bilinear', align_corners=False).squeeze(0)
    attn  = F.interpolate(attn.detach().unsqueeze(0),  size=SIZE, mode='bilinear', align_corners=False).squeeze(0)
    image = F.interpolate(image.detach().unsqueeze(0), size=SIZE, mode='bilinear', align_corners=False).squeeze(0)

    mask  = mask.cpu().numpy()    # (C+1, H, W)
    attn  = attn.cpu().numpy()    # (1, H, W)
    image = image.cpu().numpy()   # (3, H, W)

    image = np.transpose(image, (1, 2, 0))
    image = image * np.array(dset.std) + np.array(dset.mean)
    image = np.clip(image, 0, 1)

    attn_map   = attn[0]                   # (H, W)
    argmax_map = np.argmax(mask, axis=0)   # (H, W)

    # global FG/BG masks
    fg_last_ch   = (argmax_map != num_classes)          # True = FG (not background winner)
    fg_attn_med  = (attn_map > np.median(attn_map))     # True = FG by attention median

    # consistent color per channel across all rows
    colors = plt.cm.tab10(np.linspace(0, 1, num_channels))

    n_rows, n_cols = 4, num_channels
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.2, n_rows * 2.4),
                             squeeze=False)

    # hide every axis by default; re-enable individually
    for ax in axes.flatten():
        ax.axis('off')

    # ── helper: render a boolean FG/BG mask as a 2-color RGBA overlay ───
    def fg_bg_rgba(fg_bool, fg_color_rgb, fg_alpha=0.55, bg_alpha=0.35):
        """Returns an RGBA image (H, W, 4) with fg_color for True pixels."""
        out = np.zeros((*fg_bool.shape, 4), dtype=float)
        out[fg_bool]  = [*fg_color_rgb, fg_alpha]
        out[~fg_bool] = [0.15, 0.15, 0.15, bg_alpha]
        return out

    # ── helper: quick axis setup ─────────────────────────────────────────
    def prep_ax(ax, title):
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=7, pad=3)

    # ════════════════════════════════════════════════════════════════════
    # ROW 0 — global views (only cols 0-3 are used)
    # ════════════════════════════════════════════════════════════════════

    # Col 0 — raw input image
    ax = axes[0, 0]
    ax.imshow(image)
    prep_ax(ax, "Input Image")

    # Col 1 — FG/BG by last channel being the argmax
    ax = axes[0, 1]
    ax.imshow(image)
    ax.imshow(fg_bg_rgba(fg_last_ch, [0.2, 0.85, 0.2]))
    prep_ax(ax, "FG/BG (last-ch argmax)")

    # Col 2 — FG/BG by attention median threshold
    ax = axes[0, 2]
    ax.imshow(image)
    ax.imshow(fg_bg_rgba(fg_attn_med, [0.2, 0.6, 1.0]))
    prep_ax(ax, "FG/BG (attn median)")

    # Col 3 — winner class per pixel (argmax coloured by class)
    ax = axes[0, 3]
    ax.imshow(image)
    legend_handles = []
    for i in range(num_classes):
        region = (argmax_map == i).astype(float)
        if np.any(region):
            ax.contourf(region, levels=[0.5, 1], colors=[colors[i]], alpha=0.35)
            ax.contour( region, levels=[0.5],    colors=[colors[i]], linewidths=1.2)
            legend_handles.append(mpatches.Patch(color=colors[i], alpha=0.7,
                                                  label=class_id_to_name[i]))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=5, framealpha=0.7)
    prep_ax(ax, "Winner Class (argmax)")

    # ════════════════════════════════════════════════════════════════════
    # ROWS 1-3 — per-channel views, one column per channel
    # ════════════════════════════════════════════════════════════════════

    global_min = mask.min()
    global_max = mask.max() if mask.max() > mask.min() else mask.min() + 1e-6

    for i in range(num_channels):
        ch_map    = mask[i]                          # (H, W)
        ch_median = np.median(ch_map)
        ch_fg     = ch_map > ch_median               # per-channel FG/BG

        ch_name   = class_id_to_name[i]
        col_color = colors[i]                        # RGBA from tab10

        # ── Row 1: raw heatmap (no image underneath, full channel range) ──
        ax = axes[1, i]
        im = ax.imshow(ch_map, cmap='hot', vmin=global_min, vmax=global_max)
        prep_ax(ax, ch_name)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ── Row 2: heatmap overlaid on image ─────────────────────────────
        ax = axes[2, i]
        ax.imshow(image)
        ax.imshow(ch_map, cmap='hot', alpha=0.55, vmin=global_min, vmax=global_max)
        prep_ax(ax, ch_name)

        # ── Row 3: per-channel FG/BG split (median threshold) ────────────
        ax = axes[3, i]
        ax.imshow(image)
        ax.imshow(fg_bg_rgba(ch_fg, col_color[:3], fg_alpha=0.6, bg_alpha=0.25))
        prep_ax(ax, ch_name)

    # ── row labels on the left-most visible column ──────────────────────
    row_labels = [
        "Global Views",
        "Channel Heatmaps",
        "Heatmap × Image",
        "FG/BG per Channel\n(median thresh)",
    ]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=8, fontweight='bold',
                               rotation=90, labelpad=6, va='center')

    fig.suptitle(f"Epoch {epoch}", fontsize=12)
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