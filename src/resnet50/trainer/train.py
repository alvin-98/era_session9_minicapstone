import time
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

def train_one_epoch(model, loader, device, optimizer, criterion, max_steps=None):
    model.train()
    running_loss = 0.0
    n_seen = 0
    t0 = time.time()
    
    # Check if we're using DDP
    is_distributed = dist.is_initialized()
    is_main = not is_distributed or dist.get_rank() == 0

    if max_steps is None:
        # normal finite dataset (iterate fully)
        pbar = tqdm(loader, desc="Train", leave=False, disable=not is_main)
        for batch in pbar:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            if n_seen and is_main:
                pbar.set_postfix({
                    "loss": f"{running_loss / n_seen:.4f}",
                    "seen": n_seen,
                })
    else:
        # streaming or truncated training
        data_iter = iter(loader)
        steps = 0
        pbar = tqdm(total=max_steps, desc="Train", leave=False, disable=not is_main)
        while steps < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            steps += 1
            pbar.update(1)
            if n_seen and is_main:
                pbar.set_postfix({
                    "loss": f"{running_loss / n_seen:.4f}",
                    "seen": n_seen,
                })
        pbar.close()

    # Synchronize metrics across all ranks
    if is_distributed:
        metrics = torch.tensor([running_loss, n_seen], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        running_loss = metrics[0].item()
        n_seen = int(metrics[1].item())

    dt = time.time() - t0
    avg_loss = running_loss / max(1, n_seen)
    return avg_loss, n_seen, dt
