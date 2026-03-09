import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_masks_together(mask, image, dset, runs_dir, epoch, save_name):
    class_id_to_name = {i: name for i, name in enumerate(dset.classes)}
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(mask, "detach"):
        mask = mask.detach()

    image_up = F.interpolate(image.unsqueeze(0), size=64, mode='bilinear', align_corners=False)
    mask_up = F.interpolate(mask.unsqueeze(0), size=64, mode='bilinear', align_corners=False)

    image_up = image_up.squeeze(0).cpu().numpy()
    mask_up = mask_up.squeeze(0).cpu().numpy()

    if image_up.ndim == 3 and image_up.shape[0] in [1, 3]:
        image_up = np.transpose(image_up, (1, 2, 0))

    mean = np.array(dset.mean)
    std = np.array(dset.std)
    image_up = image_up * std + mean
    image_up = np.clip(image_up, 0, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, mask_up.shape[0]))

    argmax_map = np.argmax(mask_up, axis=0)  # HW

    plt.figure()
    plt.imshow(image_up)

    legend_handles = []
    for i in range(mask_up.shape[0] - 1): # skip NONE channel
        binary = (argmax_map == i).astype(float) 

        if np.any(binary):
            plt.contourf(binary, levels=[0.5, 1], colors=[colors[i]], alpha=0.35)
            plt.contour(binary, levels=[0.5], colors=[colors[i]], linewidths=2)
            handle = mpatches.Patch(color=colors[i], alpha=0.6, label=f'{class_id_to_name[i]}')
            legend_handles.append(handle)

    save_path = os.path.join(runs_dir, f"figs/epoch{epoch}")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, save_name)

    plt.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.7)
    plt.title("All Mask Regions")
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


