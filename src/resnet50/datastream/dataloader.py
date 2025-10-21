# datastream/dataloader.py (non-streaming, on-the-fly transforms)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from datasets import load_dataset, Image as HFImage
from PIL import Image
import torch, os

def make_transforms(img_size=224):
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    return train_tfm, val_tfm

class HFDataset(Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]  # loads from local cache
        img = ex["image"]
        if not isinstance(img, Image.Image):
            # ensure PIL (should already be after cast_column)
            img = Image.fromarray(img)
        x = self.transform(img.convert("RGB"))
        y = ex["label"]
        return {"pixel_values": x, "label": y}

def build_dataloaders(dataset_name, img_size, batch_size, num_workers=8, seed=42, use_ddp=False):
    torch.manual_seed(seed)

    # Load cached splits (no streaming) and ensure PIL decoding
    info = load_dataset(dataset_name)  # metadata to detect split name
    val_split = "validation" if "validation" in info.keys() else "val"

    train_hf = load_dataset(dataset_name, split="train", streaming=False)
    val_hf   = load_dataset(dataset_name, split=val_split, streaming=False)

    train_hf = train_hf.cast_column("image", HFImage(decode=True))
    val_hf   = val_hf.cast_column("image", HFImage(decode=True))

    # Make lazy transforms
    tfm_train, tfm_val = make_transforms(img_size)

    train_ds = HFDataset(train_hf, tfm_train)
    val_ds   = HFDataset(val_hf,   tfm_val)

    # Create samplers for DDP
    train_sampler = None
    val_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    # DataLoaders: workers apply transforms per-sample on demand
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None),  # shuffle only if not using DistributedSampler
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers, 
        pin_memory=True, 
        prefetch_factor=2, 
        persistent_workers=(num_workers>0)
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=val_sampler,
        drop_last=False,
        num_workers=num_workers, 
        pin_memory=True, 
        prefetch_factor=2, 
        persistent_workers=(num_workers>0)
    )
    return train_loader, val_loader
