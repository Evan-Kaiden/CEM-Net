import torch
import torch.nn.functional as F


def ce_loss_func(logits, targets):
    return F.cross_entropy(logits, targets)


def fg_bg_contrast_loss_func(maps, attn, margin=0.3):
    """
    The core anti-flood-fill loss. Forces the class distribution inside
    the attended region to differ from outside. If the model flood-fills,
    fg_dist and bg_dist will be nearly identical and this fires hard.
    """
    B, C, H, W = maps.shape
    attn_map = attn.squeeze(1)

    median = attn_map.flatten(1).median(dim=1).values.view(B, 1, 1)
    fg = (attn_map > median).float()
    bg = 1.0 - fg

    fg_mass = fg.sum(dim=(-2, -1)).view(B, 1) + 1e-6 
    bg_mass = bg.sum(dim=(-2, -1)).view(B, 1) + 1e-6 

    fg_dist = (maps * fg.unsqueeze(1)).sum(dim=(-2, -1)) / fg_mass
    bg_dist = (maps * bg.unsqueeze(1)).sum(dim=(-2, -1)) / bg_mass

    similarity = F.cosine_similarity(fg_dist, bg_dist, dim=1)
    return F.relu(similarity - margin).mean()


def attn_sparsity_loss_func(attn, target_coverage=0.25):
    """
    Forces attention to cover at most target_coverage of the image.
    Without this the attention collapses to covering everything,
    which makes fg_bg_contrast degenerate (fg == bg == whole image).
    """
    mean_coverage = attn.mean(dim=(-2, -1))  # (B, 1)
    return F.relu(mean_coverage - target_coverage).mean()


def attn_entropy_loss_func(attn):
    """
    Pushes attention values toward 0 or 1 by minimizing entropy.
    """
    p = attn.clamp(1e-6, 1 - 1e-6)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log())
    return entropy.mean()


def tv_loss_func(maps):
    """
    spatially smooth class maps are still desirable.
    Removed the attention TV — sparsity + entropy losses handle
    attention structure better than smoothness would.
    """
    dx = (maps[:, :, :, 1:] - maps[:, :, :, :-1]).abs().mean()
    dy = (maps[:, :, 1:, :] - maps[:, :, :-1, :]).abs().mean()
    return dx + dy

def masking_consistency_loss(model, images, logits, attn, targets):
    B = images.shape[0]

    masked_logits = model.forward_masked(images, attn.detach())

    # probabilities instead of raw logits
    original_probs = F.softmax(logits, dim=1)
    masked_probs   = F.softmax(masked_logits, dim=1)

    original_score = original_probs[torch.arange(B), targets]  # (B,)
    masked_score   = masked_probs[torch.arange(B), targets]     # (B,)

    drop = original_score - masked_score
    return -drop.mean()