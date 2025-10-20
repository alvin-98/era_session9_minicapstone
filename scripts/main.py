# main.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

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
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = build_dataloaders(
        args.dataset, args.img_size, args.batch_size, args.workers
    )

    # Model / loss / opt / sched
    model = resnet50(num_classes=1000).to(device)
    print(f"Model is on: {next(model.parameters()).device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_top1 = 0.0
    for epoch in range(args.epochs):
        avg_loss, n_seen, dt = train_one_epoch(
            model, train_loader, device, optimizer, criterion, max_steps=args.train_steps
        )
        scheduler.step()

        top1, top5 = evaluate(model, val_loader, device, max_steps=args.val_steps)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"loss {avg_loss:.4f} | seen {n_seen} | top1 {top1:.2f} | top5 {top5:.2f} | {dt:.1f}s"
        )

        if top1 > best_top1:
            best_top1 = top1
            torch.save({"epoch": epoch+1, "model": model.state_dict()}, os.path.join(args.out, "best.pt"))

if __name__ == "__main__":
    main()
