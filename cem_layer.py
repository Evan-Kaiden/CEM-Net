import torch
import torch.nn as nn
import torch.nn.functional as F


from entmax import sparsemax, entmax15

class EvidenceMapModule(nn.Module):
    def __init__(self, num_classes:int, size_after_backbone: tuple[int, int, int], original_image_dimension:int):
        super().__init__()
        in_channels, in_h, in_w = size_after_backbone

        # add a NONE class with +1
        self.num_classes = num_classes + 1
        self.original_image_dimension = original_image_dimension
        self.upscale = self._build_upsampler(in_channels, in_h, in_w, original_image_dimension, num_classes + 1)
        
        self.entmax15 = entmax15

    def _build_upsampler(self, in_channels, in_h, in_w, target_size, num_classes):
        layers = []

        current_h = in_h
        current_w = in_w
        channels = in_channels

        while current_h * 2 <= target_size and current_w * 2 <= target_size:

            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            
            next_channels = max(channels // 2, num_classes * 2)
            
            layers.append(nn.Conv2d(channels, next_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

            channels = next_channels
            current_h *= 2
            current_w *= 2

        if current_h != target_size or current_w != target_size:
            layers.append(nn.Upsample(size=(target_size, target_size), mode="bilinear",align_corners=False))

        layers.append(nn.Conv2d(channels, num_classes, kernel_size=1))

        return nn.Sequential(*layers)
    
    def forward(self, x, attn, return_maps=False):
        upscaled = self.upscale(x)
        maps = self.entmax15(upscaled, dim=1)

        attended = maps * attn
        attn_mass = attn.sum(dim=(-2,-1), keepdim=True) + 1e-6
        logits_full = attended.sum(dim=(-2,-1)) / attn_mass.squeeze(-1).squeeze(-1)
        logits = logits_full[:, :-1]

        if return_maps:
            return logits, maps, attn
        return logits
