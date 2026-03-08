import torch
import torch.nn as nn

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
        self.evidence_mapper = EvidenceMapModule(num_classes, size_after_backbone, input_size)

    def forward(self, x, inference=False, return_maps=False, inference_thresh=None):
        x = self.backbone(x)
        x = self.evidence_mapper(x, inference=inference,return_maps=return_maps, inference_thresh=inference_thresh)
        return x