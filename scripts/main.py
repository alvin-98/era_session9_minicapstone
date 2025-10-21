# main.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from resnet50 import build_dataloaders
from resnet50 import resnet50
from resnet50 import train_one_epoch
from resnet50 import evaluate

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="ILSVRC/imagenet-1k", help="HF dataset id")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-steps", type=int, default=None, help="steps per epoch (streaming)")
    ap.add_argument("--val-steps", type=int, default=None, help="validation steps")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=4, help="use 0 for HF streaming")
    ap.add_argument("--out", type=str, default="runs/streaming_minimal")
    # DDP arguments
    ap.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    return ap.parse_args()

def setup_ddp():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Cleanup the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def main():
    args = parse_args()
    
    # Initialize DDP if LOCAL_RANK is set
    use_ddp = "LOCAL_RANK" in os.environ
    local_rank = 0
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory only on main process
    if is_main_process():
        os.makedirs(args.out, exist_ok=True)
    
    if use_ddp:
        dist.barrier()  # Ensure directory is created before all processes continue

    # Data
    train_loader, val_loader = build_dataloaders(
        args.dataset, args.img_size, args.batch_size, args.workers, use_ddp=use_ddp
    )

    # Model / loss / opt / sched
    model = resnet50(num_classes=1000).to(device)
    
    # Wrap model with DDP
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if is_main_process():
        print(f"Model is on: {next(model.parameters()).device}")
        print(f"Using DDP: {use_ddp}")
        if use_ddp:
            print(f"World size: {dist.get_world_size()}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_top1 = 0.0
    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        avg_loss, n_seen, dt = train_one_epoch(
            model, train_loader, device, optimizer, criterion, max_steps=args.train_steps
        )
        scheduler.step()

        top1, top5 = evaluate(model, val_loader, device, max_steps=args.val_steps)

        if is_main_process():
            print(
                f"Epoch {epoch+1:03d}/{args.epochs} | "
                f"loss {avg_loss:.4f} | seen {n_seen} | top1 {top1:.2f} | top5 {top5:.2f} | {dt:.1f}s"
            )

            if top1 > best_top1:
                best_top1 = top1
                # Save the underlying model (unwrap DDP if needed)
                model_state = model.module.state_dict() if use_ddp else model.state_dict()
                torch.save({"epoch": epoch+1, "model": model_state}, os.path.join(args.out, "best.pt"))
    
    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()

if __name__ == "__main__":
    main()
