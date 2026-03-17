import torch
import torch.nn as nn
import torch.nn.functional as F


from entmax import sparsemax, entmax15

def topk_mean_logits(maps, k_percent=0.1):
    B, C, H, W = maps[:, :-1].shape
    flat = maps[:, :-1].view(B, C, -1)
    k = max(1, int(k_percent * H * W))
    return flat.topk(k, dim=-1).values.mean(dim=-1)

class UpsampleBlock(nn.Module):
    """Single upsampling stage with optional skip connection fusion."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        fused_channels = in_channels + skip_channels  # concat skip before conv
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Align spatial dims in case of off-by-one from strided ops
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv(x))
        return x




class EvidenceMapModule(nn.Module):
    def __init__(
        self,
        num_classes: int,
        size_after_backbone: tuple[int, int, int],
        original_image_dimension: int,
        skip_channels_per_stage: list[int] | None = None,
    ):
        super().__init__()
        in_channels, in_h, in_w = size_after_backbone

        self.num_classes = num_classes + 1  # +1 for NONE
        self.target_size = original_image_dimension

        self.skip_channels_per_stage: list[int] = skip_channels_per_stage or []

        self.stages = nn.ModuleList()
        self.entmax = entmax15
        channels = in_channels
        current_h, current_w = in_h, in_w
        stage_idx = 0

        while current_h * 2 <= self.target_size and current_w * 2 <= self.target_size:
            skip_ch = (
                self.skip_channels_per_stage[stage_idx]
                if stage_idx < len(self.skip_channels_per_stage)
                else 0
            )
            out_ch = max(channels // 2, self.num_classes * 2)
            self.stages.append(UpsampleBlock(channels, skip_ch, out_ch))
            channels = out_ch
            current_h *= 2
            current_w *= 2
            stage_idx += 1

        self.num_upsample_stages = stage_idx
        self.need_final_resize = (current_h != self.target_size or current_w != self.target_size)
        
        self.final_conv = nn.Conv2d(channels, self.num_classes, kernel_size=1)

    def forward(self, x, skips: list[torch.Tensor] | None = None):
        skips = skips or []

        for i, stage in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            x = stage(x, skip)

        if self.need_final_resize:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)

        upscaled = self.final_conv(x)
        maps = torch.sigmoid(upscaled)

    
        # logits_full = maps.sum(dim=(-2, -1))
        logits = topk_mean_logits(maps)
        return logits, maps