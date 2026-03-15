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
    B, _, H, W = images.shape

    attn_up   = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
    threshold = attn_up.flatten(2).mean(dim=2, keepdim=True).unsqueeze(-1)
    fg        = (attn_up > threshold).float()
    masked_images = images * (1.0 - fg)

    with torch.no_grad():
        masked_features = model.backbone(masked_images)
    
    # use attn_classifier during pretrain, not evidence_mapper
    masked_attn   = model.attn_head(masked_features)
    masked_attn   = torch.sigmoid(masked_attn)
    gated         = masked_features * masked_attn
    pooled        = gated.mean(dim=(-2, -1))
    masked_logits = model.attn_classifier(pooled)

    original_score = torch.sigmoid(logits)[torch.arange(B), targets]
    masked_score   = torch.sigmoid(masked_logits)[torch.arange(B), targets]

    drop = original_score - masked_score
    return -drop.mean()


def deletion_loss(logits, maps, images, model):
    # predicted class for each image
    pred_classes = logits.argmax(dim=1)

    # get corresponding class map
    B, C, H, W = maps.shape
    mask = maps[torch.arange(B), pred_classes].unsqueeze(1)  # (B,1,H,W)

    # normalize mask to [0,1]
    mask = mask.sigmoid()

    # remove important region
    removed_images = images * (1 - mask)

    # forward pass
    removed_logits = model(removed_images)
    removed_scores = removed_logits.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
    orig_scores = logits.gather(1, pred_classes.unsqueeze(1)).squeeze(1)

    # deletion loss
    delete_loss = torch.relu(removed_scores - orig_scores + 0.2).mean()
    return delete_loss


def spatial_entropy_loss(attn):
    p = attn.clamp(min=1e-8)
    entropy = -(p * p.log())
    return entropy.sum(dim=(2,3)).mean()


def mse_loss(pred, true):
    return F.mse_loss(pred, true)



def attention_alignment_loss(maps, attn, targets, reduction='mean'):
    B, Cplus1, H, W = maps.shape
    C = Cplus1 - 1

    attn = attn.squeeze(1)   # (B, H, W)

    target_map = maps[torch.arange(B), targets]        # (B, H, W)
    bg_map     = maps[:, -1]                           # (B, H, W)

    mask = torch.ones(B, C, dtype=torch.bool, device=maps.device)
    mask[torch.arange(B), targets] = False
    all_other_maps = maps[:, :-1][mask].view(B, C - 1, H, W)   # (B, C-1, H, W)

    loss_target = F.mse_loss(target_map, attn,         reduction=reduction)
    loss_bg     = F.mse_loss(bg_map,     1.0 - attn,   reduction=reduction)

    loss_other = all_other_maps ** 2
    loss_other = loss_other.mean() if reduction == 'mean' else loss_other.sum()

    return loss_target + loss_bg + loss_other


def topk_peak_loss(attn, k_percent = 0.05):
    attn_flat = attn.view(attn.shape[0], -1)
    B, C, H, W = attn.shape
    attn_flat = attn.view(B, C, -1)
    k = max(1, int(k_percent * H * W))
    topk_vals = attn_flat.topk(k, dim=-1).values
    return -topk_vals.mean()


def attn_distribution_loss(attn, device, top_spike_fraction=0.1):
    """
    Encourages attn to match a right-skewed distribution 
    with a spike of high-value pixels near 1.
    
    Target shape:
      - most pixels near 0 (background)
      - smooth right-skewed tail
      - top `top_spike_fraction` of pixels pushed toward 1
    """
    B = attn.shape[0]
    flat = attn.view(B, -1)          # (B, N)
    N = flat.shape[1]

    # build target distribution once per call
    # Beta(0.5, 4) gives right-skewed bulk with most mass near 0
    # then we replace the top fraction with values near 1
    beta_dist = torch.distributions.Beta(
        torch.tensor(0.5), torch.tensor(4.0)
    )
    target = beta_dist.sample((N,)).to(device)
    target, _ = target.sort()

    # override the top fraction with a spike near 1
    k = max(1, int(N * top_spike_fraction))
    target[-k:] = torch.linspace(0.85, 1.0, k, device=device)

    # 1D Wasserstein: sort both and take L2
    sorted_attn, _ = flat.sort(dim=1)                          # (B, N)
    target = target.unsqueeze(0).expand(B, -1)                 # (B, N)

    return F.mse_loss(sorted_attn, target)