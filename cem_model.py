import torch
import torch.nn as nn
import torch.nn.functional as F

from cem_layer import EvidenceMapModule

from archetectures import resnet

## assume resnet18
class CEMModelWrapper(nn.Module):
    def __init__(self, backbone, num_classes, input_size: int, device=None):
        super().__init__()
        self.backbone = backbone
        _device = device or next(backbone.parameters()).device

        with torch.no_grad():
            out = self.backbone(torch.zeros((1, 3, input_size, input_size), device=_device)).squeeze(0)
        
        size_after_backbone = out.shape
        in_channels = out.shape[0]
        self.input_size = input_size
        self.temperature = 50.
        self.evidence_mapper = EvidenceMapModule(num_classes, size_after_backbone, input_size)
        self.attn_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
        )

        self.attn_classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, num_classes)
        )
        
        final_conv = self.attn_head[-1]
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_conv.bias, 0.0)


    def forward(self, x, train_attention=False, return_maps=False):
        features = self.backbone(x)

        if train_attention:
            attn_logits = self.attn_head(features)
            attn = F.interpolate(attn_logits, size=(self.input_size,) * 2,
                                mode='bilinear', align_corners=False)
            attn = torch.sigmoid(attn)

            # gate at backbone resolution
            attn_small = torch.sigmoid(attn_logits)
            gated  = features * attn_small
            pooled = gated.mean(dim=(-2, -1))
            logits = self.attn_classifier(pooled)
            return logits, attn

        else:
            attn_logits = self.attn_head(features)
            attn = F.interpolate(attn_logits, size=(self.input_size,) * 2,
                                mode='bilinear', align_corners=False)
            attn = torch.sigmoid(attn)
            return self.evidence_mapper(features, attn, return_maps=return_maps)