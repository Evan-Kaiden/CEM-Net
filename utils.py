import torch

def get_scheduler(map_arg, optimizer, scheduler, epochs, lr):
    if scheduler == "cosine":
        scheduler = map_arg[scheduler](optimizer=optimizer, T_max=epochs, eta_min=(lr / 10))
    elif scheduler == "linear":
        scheduler = map_arg[scheduler](optimizer=optimizer, total_iters=epochs, start_factor=1, end_factor=.75)
    elif scheduler == "step":
        scheduler = map_arg[scheduler](optimizer=optimizer, step_size=max(1, epochs // 10), gamma=0.5)
    else:
        scheduler = None
    
    return scheduler

def print_row(epoch, metrics_vals):
    metrics = ('epoch', 'train acc %', 'test acc %', 'ce', 'bin', 'vary', 'overlap', 'map < 0.1', 'map > 0.9', 'lr', 'epoch time')
    cell_size = max(len(m) for m in metrics)
    cell_size +=  cell_size % 2 + 1

    cell_top = '-' * cell_size + '+'
    row_top = '+' + cell_top * 11
    if epoch == 0:
        header = row_top + "\n|"
        for m in metrics:
            header += m.center(cell_size)
            header += '|' 
        header += f'\n{row_top}'

        print(header)

    built = "|"
    metrics_vals = [epoch] + metrics_vals
    for m in metrics_vals:
        if isinstance(m, float):
            text = f"{m:.6g}" 
        else:
            text = str(m)

        if len(text) > cell_size:
            text = text[:cell_size]

        built += text.center(cell_size) + '|'
    built += f'\n{row_top}'
    print(built)


def mask_overlap_loss(masks):
    """
    masks: (K, H, W) tensor with values in [0,1]
    returns: scalar overlap penalty
    """
    K = masks.shape[0]
    masks = masks.view(K, -1)          # (K, HW)

    overlap = torch.matmul(masks, masks.T)   # (K, K)
    overlap = overlap - torch.diag(torch.diag(overlap))  # remove self-overlap

    return overlap.mean()