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

def tv_loss_func(maps):
    """
    spatially smooth class maps are still desirable.
    Removed the attention TV — sparsity + entropy losses handle
    attention structure better than smoothness would.
    """
    dx = (maps[:, :, :, 1:] - maps[:, :, :, :-1]).abs().mean()
    dy = (maps[:, :, 1:, :] - maps[:, :, :-1, :]).abs().mean()
    return dx + dy

def attention_alignment_loss(maps, attn, targets, reduction='mean'):
    B, Cplus1, H, W = maps.shape
    C = Cplus1 - 1

    attn = attn.squeeze(1)

    # Normalize attn to [0,1] range per image — prevents collapsed attn poisoning targets
    flat = attn.view(B, -1)
    mn = flat.min(dim=1).values.view(B, 1, 1)
    mx = flat.max(dim=1).values.view(B, 1, 1)
    attn_norm = (attn - mn) / (mx - mn + 1e-6)

    target_map = maps[torch.arange(B), targets]
    bg_map = maps[:, -1]

    mask = torch.ones(B, C, dtype=torch.bool, device=maps.device)
    mask[torch.arange(B), targets] = False
    all_other_maps = maps[:, :-1][mask].view(B, C - 1, H, W)

    loss_target = F.mse_loss(target_map, attn_norm, reduction=reduction)
    loss_bg     = F.mse_loss(bg_map, 1.0 - attn_norm, reduction=reduction)
    loss_other  = F.mse_loss(all_other_maps, torch.zeros_like(all_other_maps))

    return loss_target + 0.5 * loss_bg + 0.1 * loss_other

def topk_peak_loss(attn, k_percent = 0.05):
    attn_flat = attn.view(attn.shape[0], -1)
    B, C, H, W = attn.shape
    attn_flat = attn.view(B, C, -1)
    k = max(1, int(k_percent * H * W))
    topk_vals = attn_flat.topk(k, dim=-1).values
    return -topk_vals.mean()

def laplacian_smoothness_loss(attn):
    """
    Penalizes sharp transitions by computing the Laplacian.
    A dot has high curvature at its edge — this penalizes that.
    """
    lap_x = attn[:, :, 2:, :] - 2 * attn[:, :, 1:-1, :] + attn[:, :, :-2, :]
    lap_y = attn[:, :, :, 2:] - 2 * attn[:, :, :, 1:-1] + attn[:, :, :, :-2]
    return lap_x.pow(2).mean() + lap_y.pow(2).mean()

def peak_spread_loss(attn):
    B, C, H, W = attn.shape
    flat = attn.view(B, C, -1)
    peak = flat.max(dim=-1).values
    total = flat.sum(dim=-1)
    return (peak / (total + 1e-6)).mean()

def border_suppression_loss(attn, border=4):
    """Penalize attention near image borders."""
    B, _, H, W = attn.shape
    mask = torch.ones_like(attn)
    mask[:, :, :border, :] = 0
    mask[:, :, -border:, :] = 0
    mask[:, :, :, :border] = 0
    mask[:, :, :, -border:] = 0
    return (attn * (1 - mask)).mean()

def activation_loss(attn, target=0.3):
    """Pull mean attention toward a target value."""
    return (attn.mean() - target).pow(2)

def attended_diversity_loss(attn, features):
    import torch.nn.functional as F
    """
    Attended features should vary across batch — 
    if all attentions are the same, pooled features collapse.
    """
    # Spatially pool features weighted by attention
    attended = (features * attn).sum(dim=(-2,-1))  # (B, C)
    attended = F.normalize(attended, dim=1)
    
    # Cosine similarity matrix — penalize high similarity across batch
    sim = attended @ attended.T  # (B, B)
    B = sim.shape[0]
    off_diag = sim - torch.eye(B, device=sim.device)
    return off_diag.clamp(min=0).mean()
    