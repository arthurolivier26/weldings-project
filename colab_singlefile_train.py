

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple


def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def ensure_deps():

    need = []
    try:
        import torch 
        import torchvision  
    except Exception:
        need += ["torch", "torchvision"]

    for pkg in ["timm", "pandas", "numpy", "tqdm", "scikit-learn", "Pillow", "pyarrow", "requests"]:
        try:
            __import__(pkg.split("-")[0])
        except Exception:
            need.append(pkg)

    if need:
        print(f" Installing packages: {need}")
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + need
        subprocess.check_call(cmd)


ensure_deps()


import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm


class ImageMetaDataset(Dataset):


    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cls2id = {"OK": 0, "KO": 1, "UNKNOWN": 2}

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, row) -> Image.Image:
        local_path = None
        if "path" in row and isinstance(row["path"], str) and row["path"]:
            local_path = self.root_dir / row["path"]

       
        if local_path and local_path.is_file():
            return Image.open(local_path).convert("RGB")

        
        url = row.get("external_path")
        if isinstance(url, str) and url.startswith("http"):
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception as e:
                raise FileNotFoundError(f"Cannot fetch image at {url}: {e}")

        raise FileNotFoundError(f"Image not found locally and no valid external_path: {row}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = self._load_image(row)
        if self.transform:
            img = self.transform(img)
        label_str = row.get("class")
        if label_str not in self.cls2id:
            raise ValueError(f"Unexpected class label: {label_str}")
        y = torch.tensor(self.cls2id[label_str], dtype=torch.long)
        return img, y


def stratified_split(df: pd.DataFrame, alpha: float = 0.8, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    # split stratifié
    if "class" not in df.columns:
        raise ValueError("DataFrame must contain a 'class' column for stratification.")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=alpha, random_state=random_state)
    y = df["class"].astype(str)
    for tr_idx, va_idx in splitter.split(df, y):
        return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
    raise RuntimeError("Failed to split dataset.")


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v2(weights=weights)
        else:
            self.backbone = mobilenet_v2(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        logits = self.backbone(x)
        probs = torch.softmax(logits, dim=1)
        return {"logits": logits, "probabilities": probs}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 3e-4,
    out_path: Path = Path("best_model.pth"),
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 50))

    best_val = float("inf")
    history = {"train_losses": [], "val_losses": [], "best_val_loss": None}

    print(" Training started...")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = F.cross_entropy(out["logits"], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            l = float(loss.item())
            train_losses.append(l)
            pbar.set_postfix(loss=f"{l:.4f}")

        val_loss = evaluate(model, val_loader, device)
        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        history["train_losses"].append(avg_train)
        history["val_losses"].append(val_loss)
        print(f" Train {avg_train:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, out_path)
            print(f" Best model updated → {out_path}")

    history["best_val_loss"] = best_val
    print(f" Done in {time.time() - t0:.1f}s | best val {best_val:.4f}")
    return history


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss = F.cross_entropy(out["logits"], labels)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def download_and_extract_zip(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "dataset.zip"
    if not zip_path.exists():
        print(f" Downloading dataset from {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        print(f" Downloaded to {zip_path}")
    else:
        print(f" Reusing existing archive: {zip_path}")

    extract_root = dest_dir / "dataset"
    if not extract_root.exists():
        print(" Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        print(f" Extracted to {extract_root}")
    else:
        print(f" Reusing existing extracted folder: {extract_root}")
    return extract_root


def find_metadata_path(extract_root: Path) -> Optional[Path]:

    for p in extract_root.rglob("ds_meta.parquet"):
        return p
    # Fallback
    for p in extract_root.rglob("*.csv"):
        if p.name.lower().startswith("ds_meta"):
            return p
    return None


def build_dataframe(meta_path: Path, dataset_root: Path, limit: Optional[int] = None) -> pd.DataFrame:
    print(f" Loading metadata: {meta_path}")
    if meta_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(meta_path)
    else:
        df = pd.read_csv(meta_path)

 
    keep = [c for c in ["class", "path", "external_path", "welding-seams", "labelling_type"] if c in df.columns]
    df = df[keep].copy()
  
    if "class" in df.columns:
        df = df[df["class"].isin(["OK", "KO", "UNKNOWN"])].copy()

 
    if limit is not None and limit > 0:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f" Using a subset of {len(df)} samples for quick start")

 
    sample_local = 0
    for i in range(min(50, len(df))):
        p = df.iloc[i].get("path")
        if isinstance(p, str) and (dataset_root / p).is_file():
            sample_local += 1
    print(f" Local path check (first 50): found {sample_local} existing files")
    return df


def create_loaders(df_train: pd.DataFrame, df_val: pd.DataFrame, dataset_root: Path, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    ds_train = ImageMetaDataset(df_train, dataset_root, transform=tf_train)
    ds_val = ImageMetaDataset(df_val, dataset_root, transform=tf_val)

  
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def main():

    dataset_url: str = "https://minio-storage.apps.confianceai-public.irtsysx.fr/challenge-welding/datasets/example_mini_dataset.zip"
    work_dir = Path("/content")
    out_dir = Path("/content/output")
    epochs: int = 10
    batch_size: int = 128
    lr: float = 3e-4
    subset: int = 0  
    num_classes: int = 3  


    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("torch:", torch.__version__)
    try:
        import torchvision
        print("torchvision:", torchvision.__version__)
    except Exception:
        pass
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 1) Download & extract dataset
    extract_root = download_and_extract_zip(dataset_url, work_dir)

    # 2) Locate metadata and build dataframe
    meta_path = find_metadata_path(extract_root)
    if meta_path is None:
        raise FileNotFoundError("Could not find metadata file (ds_meta.parquet or ds_meta*.csv) in the extracted dataset.")
    df_all = build_dataframe(meta_path, extract_root, limit=subset if subset > 0 else None)

    # 3) Stratified split
    df_train, df_val = stratified_split(df_all, alpha=0.8, random_state=42)
    print(f"Train: {len(df_train)} | Val: {len(df_val)}")

    # 4) Dataloaders
    train_loader, val_loader = create_loaders(df_train, df_val, extract_root, batch_size=batch_size)

    # 5) Model
    model = MobileNetV2Classifier(num_classes=num_classes, pretrained=True)

    # 6) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 7) Train
    best_path = out_dir / "best_model.pth"
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        out_path=best_path,
    )

    print(f" Training finished. Weights saved to: {best_path}")


if __name__ == "__main__":
    main()


