import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataFrameDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path = ".",
        path_col: str = "path",
        label_col: str = "label",
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        channels_first: bool = True,
    ):
        """
        df : DataFrame contenant au moins les colonnes `path` et `label`
        root_dir : dossier racine des images (si `path` est relatif)
        transform : transformations sur l'image (e.g. torchvision.transforms)
                    Si None, on applique par défaut ToTensor() -> (C,H,W).
        label_transform : optionnel, pour encoder les labels (e.g. mapping str->int)
        channels_first : 
            - True  -> retourne les images au format (C, H, W) (PyTorch standard)
            - False -> retourne les images au format (H, W, C)
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform
        self.label_transform = label_transform
        self.channels_first = channels_first

        # Transform par défaut si rien n'est fourni
        if self.transform is None:
            self.transform = transforms.ToTensor()  # PIL -> Tensor (C,H,W), [0,1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = self.root_dir / row[self.path_col]
        label = row[self.label_col]

        # Chargement de l'image avec PIL (H,W,C implicite)
        image = Image.open(img_path).convert("RGB")

        # Application des transforms (souvent -> Tensor (C,H,W) avec ToTensor)
        image = self.transform(image)

        # Réorganisation éventuelle des canaux
        if isinstance(image, torch.Tensor) and image.ndim == 3:
            # image.shape = (C,H,W) ou (H,W,C)
            if self.channels_first:
                # On veut (C,H,W)
                if image.shape[0] != 3 and image.shape[-1] == 3:
                    # Cas où l'image serait (H,W,C) par erreur
                    image = image.permute(2, 0, 1)
            else:
                # On veut (H,W,C)
                if image.shape[0] == 3:
                    # Cas standard torchvision : (C,H,W) -> (H,W,C)
                    image = image.permute(1, 2, 0)

        # Transform label si besoin
        if self.label_transform is not None:
            label = self.label_transform(label)

        # Si label est un entier, on le convertit en tensor long
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)

        return image, label