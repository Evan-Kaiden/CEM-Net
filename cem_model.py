import torch
import torch.nn as nn
import torch.nn.functional as F

from cem_layer import EvidenceMapModule

def normalize_attn(attn, eps=1e-6, clip_low=0.05, clip_high=0.95):
    B = attn.shape[0]
    flat = attn.view(B, -1)
    mn = flat.min(dim=1).values.view(B, 1, 1, 1)
    mx = flat.max(dim=1).values.view(B, 1, 1, 1)
    normalized = (attn - mn) / (mx - mn + eps)
    return normalized.clamp(clip_low, clip_high)

class SpatialSoftmaxAttention(nn.Module):
    def __init__(self, in_channels, temperature=5.0):
        super().__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU()
        )
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

class ChannelPool(nn.Module):
    def forward(self, x):
        max_pool = x.amax(dim=1, keepdim=True)
        avg_pool = x.mean(dim=1, keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)
    
class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(2, 1, kernel_size=7, padding=0),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale, scale

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel attention
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
        )
        # Spatial attention
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        # Channel gate
        avg = x.mean(dim=(-2,-1))
        mxp = x.amax(dim=(-2,-1))
        ch_attn = self.channel_fc(avg) + self.channel_fc(mxp)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1).sigmoid()
        x = x * ch_attn
        return self.spatial_gate(x)

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
        # self.attn_head = SpatialSoftmaxAttention(in_channels)
        self.attn_head = CBAM(in_channels)
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
                norm_attn = normalize_attn(attn_up)
                return logits, maps, norm_attn
            else:
                return logits