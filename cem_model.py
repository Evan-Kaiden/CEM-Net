import torch
import torch.nn as nn
import torch.nn.functional as F

from cem_layer import EvidenceMapModule

class AttentionHead(nn.Module):
    """
    Upsamples from backbone resolution all the way to target_size,
    fusing skip connections at matching resolutions along the way.
    """
    def __init__(self, in_channels: int, skip_channels_per_stage: list[int],
                 target_size: int):
        super().__init__()
        self.target_size = target_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        current_ch = in_channels // 2

        # Fusion blocks — one per skip (deep → shallow)
        self.fusion_blocks = nn.ModuleList()
        for skip_ch in skip_channels_per_stage:
            fused_ch = current_ch + skip_ch
            out_ch   = max(fused_ch // 2, 16)
            self.fusion_blocks.append(nn.Sequential(
                nn.Conv2d(fused_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ))
            current_ch = out_ch

        # Extra upsampling blocks to reach target_size after skips are exhausted
        # We don't know exact resolution at init, so we use a few more refinement convs
        # and let forward() handle the remaining resize via interpolate
        self.refine = nn.Sequential(
            nn.Conv2d(current_ch, max(current_ch // 2, 16), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(current_ch // 2, 16), max(current_ch // 4, 16), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        final_ch = max(current_ch // 4, 16)

        self.final_conv = nn.Conv2d(final_ch, 1, kernel_size=1)
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=1)
        nn.init.constant_(self.final_conv.bias, 0.0)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = self.stem(x)

        # Upsample + fuse each skip
        for block, skip in zip(self.fusion_blocks, skips):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        # Continue upsampling to target_size after skips are exhausted
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)
        x = self.refine(x)

        return self.final_conv(x) 


class CEMModelWrapper(nn.Module):
    def __init__(self, backbone, num_classes, input_size: int,
                 skip_layer_names: list[str] | None = None, device=None):
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        self.skip_layer_names = skip_layer_names or []
        _device = device or next(backbone.parameters()).device

        # --- Probe backbone + collect skip shapes via hooks ---
        skip_channels = []
        hooks = []

        def make_probe_hook(store):
            def hook(module, input, output):
                store.append(output.shape[1])
            return hook

        for name in self.skip_layer_names:
            layer = dict(backbone.named_modules())[name]
            hooks.append(layer.register_forward_hook(make_probe_hook(skip_channels)))

        with torch.no_grad():
            dummy = torch.zeros((1, 3, input_size, input_size), device=_device)
            out = backbone(dummy).squeeze(0)

        for h in hooks:
            h.remove()

        size_after_backbone = out.shape
        in_channels = out.shape[0]

        skip_channels = skip_channels[::-1]  # deep → shallow

        self.evidence_mapper = EvidenceMapModule(
            num_classes, size_after_backbone, input_size,
            skip_channels_per_stage=skip_channels,
        )
        self.attn_head = AttentionHead(in_channels, skip_channels, input_size)

        self.attn_classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, num_classes),
        )

    def _backbone_forward(self, x):
        skips = []

        def make_hook():
            def hook(module, input, output):
                skips.append(output)
            return hook

        hooks = []
        for name in self.skip_layer_names:
            layer = dict(self.backbone.named_modules())[name]
            hooks.append(layer.register_forward_hook(make_hook()))

        features = self.backbone(x)

        for h in hooks:
            h.remove()

        return features, skips

    def forward(self, x, train_attention=False, return_maps=False):
        features, skips = self._backbone_forward(x)
        skips = skips[::-1]
        attn_logits = self.attn_head(features, skips)
        attn = torch.sigmoid(attn_logits)

        if train_attention:
            features_up = F.interpolate(features, size=attn_logits.shape[-2:],
                                mode="bilinear", align_corners=False)
            gated  = features_up * attn
            pooled = gated.mean(dim=(-2, -1))
            logits = self.attn_classifier(pooled)
            return logits, attn
        else:
            return self.evidence_mapper(
                features, attn, skips=skips,
                return_maps=return_maps,
            )