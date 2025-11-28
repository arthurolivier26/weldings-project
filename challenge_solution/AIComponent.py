import os
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA

# torchvision weights API 
try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
except Exception:
    # fallback for older torchvision 
    from torchvision.models import mobilenet_v2
    MobileNet_V2_Weights = None

warnings.filterwarnings("ignore")


# ---------------------------
# ABSTRACT INTERFACE
# ---------------------------
class AbstractAIComponent(ABC):
    def __init__(self):
        self.AI_Component = None
        self.AI_component_meta_informations = {}

    @abstractmethod
    def load_model(self, config_file: Optional[str] = None):
        pass

    @abstractmethod
    def predict(self, input_images: List[np.ndarray], images_meta_informations: List[dict]) -> dict:
        pass


# ---------------------------
# BACKBONE
# ---------------------------
class MobileNetV2Backbone(nn.Module):
    def __init__(self, use_weights: bool = True, num_classes: int = 3):
        super().__init__()
        # use modern weights API if available; else fallback to pretrained bool
        if MobileNet_V2_Weights is not None and use_weights:
            weights = MobileNet_V2_Weights.DEFAULT
            base = mobilenet_v2(weights=weights)
        else:
            base = mobilenet_v2(pretrained=use_weights)

        # capture some params for re-use
        last_channel = getattr(base, "last_channel", 1280)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # new classifier head (kept small)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)  # [B, D]
        logits = self.classifier(embedding)  # [B, C]
        probs = F.softmax(logits, dim=1)
        return {"logits": logits, "probabilities": probs, "embedding": embedding}

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


# ---------------------------
# AUGMENTATION / PREPROCESS
# ---------------------------
class WeldingAugmentation:
    def __init__(self, is_training: bool = True, size: Tuple[int, int] = (224, 224)):
        common = [
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if is_training:
            aug = [
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
            ]
            self.transform = transforms.Compose(aug + common)
        else:
            self.transform = transforms.Compose(common)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.transform(image)


# ---------------------------
# OOD (Mahalanobis) + util
# ---------------------------
class MahalanobisOODDetector:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.pca = None
        self.cov = EmpiricalCovariance(assume_centered=False)
        self._fitted = False

    def _to_np(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def fit(self, embeddings: np.ndarray):
        X = self._to_np(embeddings)
        if self.n_components is not None and self.n_components < X.shape[1]:
            self.pca = PCA(n_components=self.n_components)
            X = self.pca.fit_transform(X)
        else:
            self.pca = None
        self.cov.fit(X)
        self._fitted = True

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        X = self._to_np(embeddings)
        if not self._fitted:
            # fallback: return large values to be conservative OR zeros to be permissive
            # default choose zeros (not OOD) but configurable by caller
            return np.zeros(X.shape[0])
        if self.pca is not None:
            X = self.pca.transform(X)
        return self.cov.mahalanobis(X)  # distances

    def state_dict(self):
        return {"n_components": self.n_components, "pca": self.pca, "cov": self.cov, "_fitted": self._fitted}

    def load_state_dict(self, state: dict):
        self.n_components = state.get("n_components")
        self.pca = state.get("pca")
        self.cov = state.get("cov")
        self._fitted = state.get("_fitted", False)


# ---------------------------
# THRESHOLD CALIBRATOR
# ---------------------------
class SafetyThresholdEstimator:
    """
    Calibre les seuils à partir d'un jeu de validation.
    Expose fit_from_validation(confidences, mahalanobis_distances, labels)
    """
    def __init__(self):
        self.thresholds = {
            "confidence_threshold": 0.6,
            "ood_threshold": 1.0,
            "max_unknown_entropy": 1.0,
        }
        self._fitted = False

    def fit_from_validation(self, confidences: np.ndarray, mahalanobis_dists: np.ndarray, labels: np.ndarray):
        # confidences: top1 prob per sample
        c = np.asarray(confidences).reshape(-1)
        m = np.asarray(mahalanobis_dists).reshape(-1)
        y = np.asarray(labels).reshape(-1)
        if not (len(c) == len(m) == len(y)):
            raise ValueError("Mismatched lengths for calibration data.")

        # Confidence threshold = 10th percentile of correct predictions' confidences
        # compute correctness if label provided as ints and predicted probs included elsewhere
        self.thresholds["confidence_threshold"] = float(np.percentile(c, 10))
        # OOD threshold = 95th percentile of in-distribution mahalanobis distances
        self.thresholds["ood_threshold"] = float(np.percentile(m, 95))
        self._fitted = True
        return self.thresholds

    def is_safe(self, conf: float, mahal: float) -> bool:
        if not self._fitted:
            # conservative fallback: require conf > 0.7 and mahal < 0.7 * ood_threshold
            return (conf >= 0.7) and (mahal <= 0.7 * self.thresholds["ood_threshold"])
        return (conf >= self.thresholds["confidence_threshold"]) and (mahal <= self.thresholds["ood_threshold"])

    def state_dict(self):
        return {"thresholds": self.thresholds, "_fitted": self._fitted}

    def load_state_dict(self, state: dict):
        self.thresholds = state.get("thresholds", self.thresholds)
        self._fitted = state.get("_fitted", False)


# ---------------------------
# AI COMPONENT 
# ---------------------------
class MyAIComponent(AbstractAIComponent):
    def __init__(self, device: Optional[str] = None, batch_size: int = 16):
        super().__init__()
        self.model: Optional[MobileNetV2Backbone] = None
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.preprocess = WeldingAugmentation(is_training=False)
        self.class_names = ["OK", "KO", "UNKNOWN"]

        # Trustworthy modules
        self.ood_detector = MahalanobisOODDetector(n_components=64)
        self.threshold_estimator = SafetyThresholdEstimator()

        # tuning
        self.batch_size = batch_size
        self.blur_threshold = 100.0
        self.darkness_threshold = 40.0

    def init_model(self, use_weights=True):
        self.model = MobileNetV2Backbone(use_weights, num_classes=len(self.class_names))
        self.model.to(self.device)

    def load_model(self, path: str = "best_model.pth", strict: bool = False):
        root = Path(__file__).parent.resolve()
        ckpt = Path(path) if Path(path).is_absolute() else root / path
        if self.model is None:
            self.init_model(use_weights=True)

        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device)
            model_state = state.get("model_state_dict", state)
            self.model.load_state_dict(model_state, strict=strict)
            if "ood_detector" in state:
                self.ood_detector.load_state_dict(state["ood_detector"])
            if "thresholds" in state:
                self.threshold_estimator.load_state_dict(state["thresholds"])
            print("Loaded checkpoint:", ckpt)
        else:
            print("checkpoint not found:", ckpt)

        self.model.eval()
        self._warmup_model()

    def save_model(self, path: str = "best_model.pth", extras: dict = None):
        state = {"model_state_dict": self.model.state_dict()}
        if extras:
            state.update(extras)
        torch.save(state, path)
        print("Saved checkpoint to", path)

    # physical image quality OOD
    def _compute_physical_ood(self, img_array: np.ndarray) -> float:
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif img_array.ndim == 2:
            gray = img_array
        else:
            gray = cv2.cvtColor(img_array.squeeze(), cv2.COLOR_RGB2GRAY)

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = self.blur_threshold / (lap_var + 1e-6)
        mean_brightness = np.mean(gray)
        dark_score = self.darkness_threshold / (mean_brightness + 1e-6)
        return float(max(blur_score, dark_score))

    # batch predict
    def predict(self, input_images: List[np.ndarray], images_meta_informations: List[dict] = None, batch_size: Optional[int] = None) -> dict:
        if self.model is None:
            raise RuntimeError("Model not initialized / loaded.")

        bs = batch_size or self.batch_size
        results = {"predictions": [], "probabilities": [], "OOD_scores": []}
        N = len(input_images)

        # 1) compute physical OOD for all images (fast)
        phys_ood = [self._compute_physical_ood(im) for im in input_images]

        # 2) preprocess into tensors (vectorized)
        tensors = []
        for im in input_images:
            t = self.preprocess(im)  # returns tensor [C,H,W]
            tensors.append(t)
        dataset_tensor = torch.stack(tensors, dim=0).to(self.device)

        # 3) run in batches through the model
        all_probs = []
        all_embeddings = []
        with torch.no_grad():
            self.model.eval()
            for i in range(0, N, bs):
                batch = dataset_tensor[i : i + bs]
                out = self.model(batch)
                probs = out["probabilities"].cpu().numpy()  # [b, C]
                emb = out["embedding"].cpu().numpy()        # [b, D]
                all_probs.append(probs)
                all_embeddings.append(emb)
        all_probs = np.vstack(all_probs)
        all_embeddings = np.vstack(all_embeddings)

        # 4) compute mahalanobis distances in (vectorized)
        mahal_dists = self.ood_detector.predict(all_embeddings)  # np array [N]
        # normalize by learned threshold if threshold estimator fitted, else use raw distances
        # final OOD score: ratio raw / ood_threshold
        ood_thresh = self.threshold_estimator.thresholds.get("ood_threshold", 1.0)
        stat_ood_scores = mahal_dists / (ood_thresh + 1e-12)
        final_ood_scores = np.maximum(stat_ood_scores, np.array(phys_ood))

        # 5) safety decision per sample
        for i in range(N):
            probs = all_probs[i]
            conf = float(np.max(probs))
            mahal = float(mahal_dists[i])
            ood_score = float(final_ood_scores[i])

            safe = self.threshold_estimator.is_safe(conf, mahal)
            if not safe or ood_score >= 1.0:
                # conservative fallback: UNKNOWN with soft probs that keep original shape
                results["predictions"].append("UNKNOWN")
                # soft fallback: shrink original probs toward uniform with factor
                fallback = (0.2 * np.ones_like(probs) + 0.8 * probs).tolist()
                results["probabilities"].append([float(x) for x in fallback])
                results["OOD_scores"].append(ood_score)
            else:
                pred_idx = int(np.argmax(probs))
                results["predictions"].append(self.class_names[pred_idx])
                results["probabilities"].append([float(x) for x in probs.tolist()])
                results["OOD_scores"].append(ood_score)

        return results

    def _warmup_model(self):
        self.model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            for _ in range(2):
                _ = self.model(dummy)

    # helpers for training pipeline
    def freeze_backbone(self):
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        self.model.freeze_backbone()

    def unfreeze_backbone(self):
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        self.model.unfreeze_backbone()
