import torch
import torch.nn as nn
import torch.nn.functional as F

from cem_layer import EvidenceMapModule

class SpatialSoftmaxAttention(nn.Module):
    def __init__(self, in_channels, temperature=5.0):
        super().__init__()
        
        self.init_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.temperature = temperature

        nn.init.normal_(self.final_conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.final_conv.bias, val=0)

    def forward(self, x):
        x = self.init_conv(x)
        x = F.relu(x)
        logits = self.final_conv(x)

        attn = torch.sigmoid(logits / 0.3)
        attended = x * attn

        return attended, attn

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
        self.attn_head = SpatialSoftmaxAttention(in_channels)

        self.attn_classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, num_classes, 1)
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
        attended, attn = self.attn_head(features)

        if train_attention:
            x = self.attn_classifier(attended)
            logits = x.mean(dim=(-2,-1))
            return logits, attn
        else:
            logits, maps = self.evidence_mapper(features, skips=skips)
            if return_maps:
                attn_up = F.interpolate(attn, size=maps.shape[-2:], 
                                    mode='bilinear', align_corners=False)
                return logits, maps, attn_up
            else:
                return logits