import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, loader, device, max_steps):
    model.eval()
    top1 = top5 = n = steps = 0
    total = max_steps if max_steps is not None else len(loader)
    pbar = tqdm(loader, total=total, desc="Eval", leave=False)
    for batch in pbar:
        x, y = batch["pixel_values"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        logits = model(x)
        _, pred = logits.topk(5, 1, True, True)
        correct = pred.eq(y.view(-1,1))
        top1 += correct[:, :1].sum().item()
        top5 += correct.sum().item()
        n += y.size(0)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            pbar.close()
            break
        if n:
            pbar.set_postfix({
                "top1": f"{100.0*top1/n:.2f}",
                "top5": f"{100.0*top5/n:.2f}",
            })
    return (100.0*top1/n if n else 0.0), (100.0*top5/n if n else 0.0)
