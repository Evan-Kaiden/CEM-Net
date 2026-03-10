import os

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


def print_row(epoch, metrics_vals, metrics_names, run_dir, print_header=False):
    cols = ['epoch'] + list(metrics_names)
    num_cols = len(cols)
    cell_size = max(len(m) for m in cols)
    cell_size += cell_size % 2 + 1

    cell_top = '-' * cell_size + '+'
    row_top  = '+' + cell_top * num_cols

    log_path = os.path.join(run_dir, "log.txt")
    output_lines = []

    if print_header:
        header = row_top + "\n|"
        for m in cols:
            header += m.center(cell_size) + '|'
        header += f'\n{row_top}'
        output_lines.append(header)
        print(header)

    built = "|"
    for m in [epoch] + list(metrics_vals):
        text = f"{m:.6g}" if isinstance(m, float) else str(m)
        if len(text) > cell_size:
            text = text[:cell_size]
        built += text.center(cell_size) + '|'
    built += f'\n{row_top}'
    output_lines.append(built)
    print(built)

    with open(log_path, "a") as f:
        for line in output_lines:
            f.write(line + "\n")


