import torch
import torch.nn as nn
import torch.nn.functional as F


from entmax import sparsemax, entmax15

class EvidenceMapModule(nn.Module):
    """
        Initialize the Class Evidence Map (CEM) layer.

        This layer takes the feature map output of a backbone network and transforms it
        into a class evidence map with spatial dimensions matching the original image.
        The layer upsamples the backbone features and projects them into a map with
        `num_classes` channels, where each channel corresponds to the spatial evidence
        for a class. Input image original dimension should be divisible by 2.

        Parameters
        ----------
        num_classes : int
            Number of output classes. The final output will have this many channels.

        size_after_backbone : tuple[int, int, int]
            Shape of the feature map coming from the backbone network in the form
            (C, H, W), where:
            - C = number of feature channels
            - H = feature map height
            - W = feature map width

        original_image_dimension : int
            Spatial dimension of the original input image (assumed square).
            The output evidence maps will be upsampled to (num_classes, original_image_dimension, original_image_dimension).
        """
    def __init__(self, num_classes:int, size_after_backbone: tuple[int, int, int], original_image_dimension:int):
        super().__init__()
        in_channels, in_h, in_w = size_after_backbone

        # add a NONE class with +1
        self.num_classes = num_classes + 1
        
        self.upscale = self._build_upsampler(in_channels, in_h, in_w, original_image_dimension, num_classes + 1)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
            )
        
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
    
    def forward(self, x, inference=False, return_maps=False, inference_thresh=None):
        scale = self.attention(x)
        x = x * scale

       
        upscaled = self.upscale(x)
        maps = self.entmax15(upscaled, dim=1) 

        # if self.training:
        #     with torch.no_grad():
        #         mask = (maps.max(dim=1, keepdim=True)[0] > 0.5).float()
        #         upscaled = upscaled * (1 - mask)
        #         maps = self.entmax15(upscaled, dim=1)
        
        # logits = maps.sum(dim=(-2, -1))
        logits = torch.logsumexp(maps * 10, dim=(-2, -1))
        if return_maps:
            return logits, maps#, scale
        return logits
