import torch

def binary_loss(masks):
    """
    encourages values to be 0 or 1
    """
    # loss = (masks * (1 - masks)).pow(2).mean()
    loss = -(masks * torch.log(masks + 1e-8)).sum(dim=1).mean()
    return loss

def mask_tv_loss(masks):
    """
    encourages spatially contiguous regions
    """
    dx = torch.abs(masks[:, :, 1:] - masks[:, :, :-1]).mean()
    dy = torch.abs(masks[:, 1:, :] - masks[:, :-1, :]).mean()

    return dx + dy

def mask_area_loss(masks):
    return masks.view(masks.shape[0], -1).mean(dim=1).mean()

def mask_overlap_loss(masks):
    """
    encourages masks to be independent of eachother
    """
    K = masks.shape[0]
    masks = masks.view(K, -1)          # (K, HW)

    overlap = torch.matmul(masks, masks.T)   # (K, K)
    overlap = overlap - torch.diag(torch.diag(overlap))  # remove self-overlap

    return overlap.mean()