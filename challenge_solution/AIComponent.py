import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
import os
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

from scipy.optimize import minimize
warnings.filterwarnings('ignore')

# =============================================================================
# ABSTRACT INTERFACE (Challenge Requirement)
# =============================================================================

class AbstractAIComponent(ABC):
    """Abstract base class for AI components interface."""

    def __init__(self):
        self.AI_Component = None
        self.AI_component_meta_informations = {}

    @abstractmethod
    def load_model(self, config_file=None):
        """Abstract method to load the model into the AI Component."""
        pass

    @abstractmethod
    def predict(self, input_images: list[np.ndarray],
                images_meta_informations: list[dict]) -> dict:
        """Abstract method to make predictions using the AI component."""
        pass


# =============================================================================
# CORE MODEL ARCHITECTURE
# =============================================================================

class WeldingQualityModel(nn.Module):
    """
    Streamlined welding quality detection model focusing on:
    - Single efficient backbone (EfficientNet-B0)
    - Uncertainty quantification
    - Fast inference for production use
    """

    def __init__(self, num_classes=3, dropout_rate=0.2):
        super().__init__()

        # Single efficient backbone for speed
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        backbone_features = 1280  # EfficientNet-B0 output features

        # Shared feature representation
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Main classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features from backbone
        backbone_features = self.backbone(x)

        # Process features
        shared_features = self.feature_processor(backbone_features)

        # Main classification
        logits = self.classifier(shared_features)
        probabilities = F.softmax(logits, dim=1)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'embedding': backbone_features}


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class WeldingAugmentation:
    """Simplified, welding-specific augmentation pipeline."""

    def __init__(self, is_training=True):
        if is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=15, fill=0),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    fill=0
                ),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.08))
            ])
        else:
            # Clean validation pipeline
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __call__(self, image):
        return self.transform(image)


# =============================================================================
# AI COMPONENT IMPLEMENTATION
# =============================================================================

class MyAIComponent(AbstractAIComponent):
    """Streamlined AI Component for production use."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        self.preprocess = None
        self.class_names = ['OK', 'KO', 'UNKNOWN']
        
        # --- Composants optionnels Trustworthy AI (non utilisés par défaut) ---
        # 
        # Instancié de possible modile complémentaire 
        # Ex : Module OOD basé Sur embdedding du backbone + PCA + Distance pour produire score OOD 
        # Ex : Temperature scaling / Calibration pour ajuster les sorties de probabilités.
        # 
        # 

        # Safety thresholds for decision making -> Mettre en place un module pour determiner le threshold 
        self.safety_thresholds = {
            'confidence_threshold': 0, # <- Tune values
            'uncertainty_threshold': 1, # <- Tune values
            'ood_threshold': 1} # <- Tune values

    def init_model(self):
        self.model = WeldingQualityModel(num_classes=3, dropout_rate=0.2)

    def load_model(self, config_file=None):
        """Load the trained model."""
        ROOT_PATH = Path(__file__).parent.resolve()
        model_path = ROOT_PATH / 'best_model.pth'

        print(f"🔧 Loading Welding Quality AI Component...")

        # Initialize model
        if self.model is None:
            self.init_model()

        # Load weights if available
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            missing_keys,unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Missing keys (dans le modèle mais pas dans le checkpoint) :", missing_keys)
            print("Unexpected keys (dans le checkpoint mais pas dans le modèle) :", unexpected_keys)

            self.safety_thresholds.update(checkpoint.get('safety_thresholds', {}))
            print(f"✅ Model weights loaded from {model_path}")
        else:
            print(f"⚠️  No pre-trained weights found, using random initialization")

        # --- Composants optionnels Trustworthy AI  ---
        # 
        # Charger les paramètres appris si nécéssaire en utilisant des fonctions module.load 
        # Ex : Module OOD basé Sur embdedding du backbone + PCA + Distance pour produire score OOD 
        # Ex : Temperature scaling / Calibration pour ajuster les sorties de probabilités.
        # 
        # 

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🔧 Using CUDA")
        else:
            self.device = torch.device('cpu')
            print(f"🔧 Using CPU")

        self.model.to(self.device)
        self.model.eval()

        # Setup preprocessing
        self.preprocess = WeldingAugmentation(is_training=False)

        # Warmup
        self._warmup_model()

        print(f"✅ AI Component loaded on {self.device}")

    def predict(self, input_images: list[np.ndarray],
                images_meta_informations: list[dict],**kwargs) -> dict:
        """Make predictions on input images."""

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        predictions = []
        probabilities = []
        ood_scores = []

        self.model.eval()

        with torch.no_grad():
            for i, img_array in enumerate(input_images):
                try:
                    # Preprocess image
                    processed_image = self._preprocess_image(img_array)
                    img_tensor = processed_image.unsqueeze(0).to(self.device)

                    # Model inference
                    outputs = self.model(img_tensor)

                    # Extract results
                    probs = outputs['probabilities'][0] # Could be based on postprocessing of outputs['logits']
                    uncertainty = 0
                    ood_score = 0
                    #embedding = outputs['embedding']

                    # --- Composants optionnels Trustworthy AI  ---
                    # Faire appelle au méthode predict des modules
                    # Ex : ood_score = MahalanobisOODDetector.predict(...)
                    # Ex : uncertainty = TemperatureScaler(...) 

                    # Make safety decision
                    final_prediction, final_probabilities = self._make_safety_decision(probs, uncertainty, ood_score)

                    predictions.append(final_prediction)
                    probabilities.append(final_probabilities)
                    ood_scores.append(max(0.0, ood_score))

                except Exception as e:
                    print(f"⚠️  Error processing image {i}: {e}")
                    predictions.append('UNKNOWN')
                    probabilities.append([0.1, 0.1, 0.8])
                    ood_scores.append(2.0)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "OOD_scores": ood_scores
        }

    def _preprocess_image(self, img_array):
        """Preprocess input image."""
        if img_array is None or img_array.size == 0:
            raise ValueError("Input image is None or empty")

        # Handle different image formats
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # RGB image
            processed = self.preprocess(img_array)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            # Grayscale to RGB
            img_rgb = np.repeat(img_array, 3, axis=2)
            processed = self.preprocess(img_rgb)
        elif len(img_array.shape) == 2:
            # Grayscale to RGB
            img_rgb = np.stack([img_array] * 3, axis=2)
            processed = self.preprocess(img_rgb)
        else:
            raise ValueError(f"Unsupported image format: {img_array.shape}")
        return processed

    def _make_safety_decision(self, probs, uncertainty, ood_score):
        """Make final prediction with safety considerations.

        Note: 'uncertainty' et 'ood_score' peuvent être alimentés par :
          - un estimateur d'incertitude (MC Dropout, ensembles, etc.)
          - un MahalanobisOODDetector (voir class plus bas).
        Les probabilités 'probs' peuvent être calibrées au préalable
        via TemperatureScaler.
        """
        # Get basic prediction
        max_prob = probs.max().item()
        predicted_class_idx = probs.argmax().item()

        # Safety checks
        low_confidence = max_prob < self.safety_thresholds['confidence_threshold']
        high_uncertainty = uncertainty > self.safety_thresholds['uncertainty_threshold']
        is_ood = ood_score > self.safety_thresholds['ood_threshold']

        # Decision logic
        if low_confidence or high_uncertainty or is_ood:
            # Conservative: classify as UNKNOWN if any safety check fails
            final_prediction = 'UNKNOWN'
            final_probabilities = [0.2, 0.2, 0.6]
        elif predicted_class_idx == 1 and max_prob < 0.8:
            # Extra conservative for KO predictions
            final_prediction = 'UNKNOWN'
            final_probabilities = [0.2, 0.3, 0.5]
        else:
            # Accept prediction
            final_prediction = self.class_names[predicted_class_idx]
            probs_cpu = probs.cpu().numpy()
            final_probabilities = [float(probs_cpu[0]), float(probs_cpu[1]), float(probs_cpu[2])]

        return final_prediction, final_probabilities

    def _warmup_model(self):
        """Warmup model for stable performance."""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
                if self.device.type == 'mps':
                    torch.mps.empty_cache()

        print("🔥 Model warmup completed")

    def train_model(
        self,
        train_dataset,
        val_dataset,
        device,
        save_path="best_model.pth",
        augmentation_fn=None,
        preprocess_fn=None,
        epochs=100,
        batch_size=64,
        lr=3e-4,
    ):
        """
        Entraîne un modèle PyTorch avec support optionnel de data augmentation.
    
        Paramètres
        ----------
        model : torch.nn.Module
            Modèle à entraîner.
        train_dataset : torch.utils.data.Dataset
            Dataset d'entraînement (images brutes + labels).
        val_dataset : torch.utils.data.Dataset
            Dataset de validation.
        device : torch.device
            CPU / CUDA / MPS.
        save_path : str
            Chemin où sauvegarder les meilleurs poids.
        augmentation_fn : callable or None
            Fonction de data augmentation appliquée aux images (facultatif).
            Ex : WeldingAugmentation(is_training=True)
        preprocess_fn : callable
            Pipeline de preprocessing (resize+tensor+normalize).
        epochs : int
            Nombre d'époques.
        batch_size : int
            Taille de batch.
        lr : float
            Learning rate.
    
        Retour
        ------
        dict contenant l'historique d'entraînement :
        {
            "train_losses": [...],
            "val_losses": [...],
            "best_val_loss": float,
        }
        """
        model = self.model
        model.to(device)
    
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-2)
    
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200)
    
        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    
        # Suivi
        best_val_loss = float("inf")
        history = {"train_losses": [], "val_losses": [], "best_val_loss": None}
    
        print("🟦 Training started...")
        start_time = time.time()
    
        # embedding = []
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []

            pbar = tqdm(train_loader, desc="Training")
    
            for images, labels in pbar:
    
                # --- Préparation des images ---
                processed_batch = []
                for img in images:
    
                    # 1) Data augmentation éventuelle
                    if augmentation_fn is not None:
                        img_np = img.numpy()
                        img = augmentation_fn(img_np)
    
                    # 2) Preprocessing obligatoire
                    if preprocess_fn is not None:
                        img_np = img.numpy()
                        img = preprocess_fn(img_np)
                    else:
                        pass
                    processed_batch.append(img)
    
                batch_tensor = torch.stack(processed_batch).to(device)
                labels = labels.to(device)
    
                # --- Forward ---
                outputs = model(batch_tensor)
                logits = outputs["logits"]
    
                loss = F.cross_entropy(logits, labels)
    
                # --- Backprop ---
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
    
            # --- Validation ---
            val_loss = self.evaluate_loss(val_loader, preprocess_fn, device)
    
            avg_train_loss = sum(train_losses) / len(train_losses)
            history["train_losses"].append(avg_train_loss)
            history["val_losses"].append(val_loss)
    
            print(f"📘 Epoch {epoch}/{epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}")
    
            # --- Sauvegarde du meilleur modèle ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                    },
                    save_path,
                )
                print(f"💾 Best model updated → {save_path}")
    
        history["best_val_loss"] = best_val_loss
    
        print(f"🏁 Training completed in {time.time() - start_time:.1f}s")
        return history
    

    def evaluate_loss(self, loader, preprocess_fn, device):
        """Calcule la loss de validation."""
        model=self.model
        model.eval()
        losses = []
    
        with torch.no_grad():
            for images, labels in loader:
    
                processed_batch = []
                for img in images:
                    if not(preprocess_fn is None):
                        img = preprocess_fn(img)
                    processed_batch.append(img)
    
                batch_tensor = torch.stack(processed_batch).to(device)
                labels = labels.to(device)
    
                outputs = model(batch_tensor)
                logits = outputs["logits"]
                loss = F.cross_entropy(logits, labels)
    
                losses.append(loss.item())
    
        return sum(losses) / len(losses)      

# Exemple de module à integrer 
class TemperatureScaler:
    """
    Temperature scaling multiclasse (logits [N, C]) en version compacte.
    """
    def __init__(self):
        self.temperature = 1.0
        self._fitted = False

    def fit(self, logits, labels):
        """
        logits : array-like [N, C]
        labels : array-like [N] (entiers 0..C-1)
        """
        logits = np.asarray(logits)
        labels = np.asarray(labels).astype(int)

        if logits.ndim != 2:
            raise ValueError("logits must be [N, C]")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("logits and labels must have same length.")

        def nll(logT):
            T = np.exp(logT)
            z = logits / T
            z = z - z.max(axis=1, keepdims=True)   # stabilité num.
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)   # softmax
            p_y = np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)
            return -np.mean(np.log(p_y))

        res = minimize(
            fun=nll,
            x0=np.array([0.0]),      # log(T)=0 → T=1
            method="L-BFGS-B"
        )

        self.temperature = float(np.exp(res.x[0]))
        self._fitted = True

    def predict(self, logits):
        """
        logits : array-like [N, C]
        return : numpy.ndarray [N, C] (probabilités calibrées)
        """
        if not self._fitted:
            raise RuntimeError("TemperatureScaler must be fitted first.")

        logits = np.asarray(logits)
        if logits.ndim != 2:
            raise ValueError("logits must be [N, C]")

        z = logits / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs =  e / e.sum(axis=1, keepdims=True)
        return probs.argmax(axis=1)

    def save(self, path):
        torch.save({
            "temperature": self.temperature,
            "_fitted": self._fitted
        }, path)

    def load(self, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        self.temperature = float(state.get("temperature", 1.0))
        self._fitted = state.get("_fitted", True)

class MahalanobisOODDetector:
    def __init__(self, n_components=None):
        """
        n_components : int ou None
            Si défini et < D, applique une PCA scikit-learn avant Mahalanobis.
        """
        self.n_components = n_components
        self.pca = None
        self.cov = EmpiricalCovariance(assume_centered=False)
        self._fitted = False

    # Utilitaire minimal : torch.Tensor -> numpy
    def _np(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def fit(self, embeddings):
        """
        embeddings : array-like [N, D] (torch.Tensor ou numpy)
        """
        X = self._np(embeddings)

        if self.n_components is not None and self.n_components < X.shape[1]:
            self.pca = PCA(n_components=self.n_components)
            X = self.pca.fit_transform(X)  # PCA -> [N, d]
        else:
            self.pca = None

        self.cov.fit(X)
        self._fitted = True

    def predict(self, embeddings):
        """
        embeddings : [B, D] torch.Tensor ou numpy
        return : numpy.ndarray [B] de floats
        """
        if not self._fitted:
            raise RuntimeError("MahalanobisOODDetector must be fitted before predict().")

        X = self._np(embeddings)
        if self.pca is not None:
            X = self.pca.transform(X)

        # EmpiricalCovariance.mahalanobis retourne directement un ndarray float
        return self.cov.mahalanobis(X)

    def save(self, path):
        torch.save({
            "n_components": self.n_components,
            "pca": self.pca,
            "cov": self.cov,
            "_fitted": self._fitted
        }, path)

    def load(self, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        self.n_components = state["n_components"]
        self.pca = state["pca"]
        self.cov = state["cov"]
        self._fitted = state["_fitted"]

class SafetyThresholdEstimator:
    """
    Estimation automatique des seuils de sécurité à partir d'une base de validation.
    Compact et indépendant du modèle.
    """

    def __init__(self):
        self.thresholds = {
            "confidence_threshold": 0.7,
            "uncertainty_threshold": 0.6,
            "ood_threshold": 1.0
        }
        self._fitted = False

    def fit(self, confidences, uncertainties, ood_scores, labels):
        """
        confidences   : array-like [N] (probabilité du top-1)
        uncertainties : array-like [N] (entropie, variance, etc.)
        ood_scores    : array-like [N] (distance ou score OOD)
        labels        : array-like [N] (vrais labels)
        """
        c = np.asarray(confidences).reshape(-1)
        u = np.asarray(uncertainties).reshape(-1)
        o = np.asarray(ood_scores).reshape(-1)
        y = np.asarray(labels).reshape(-1)

        if not (len(c) == len(u) == len(o) == len(y)):
            raise ValueError("All inputs must have same number of samples.")

        # --- Confidence : seuil séparant prédictions correctes / incorrectes ---
        # On prend le plus petit niveau de confiance parmi les prédictions correctes.
        # Approche robuste : percentile 10% pour éviter les cas extrêmes.
        correct = c[y == np.argmax(np.vstack([c, 1-c]).T, axis=1)] if c.ndim == 1 else c
        self.thresholds["confidence_threshold"] = float(np.percentile(correct, 10))

        # --- Uncertainty : plus elle est grande, plus c'est risqué ---
        # Seuil = percentile 90% des échantillons corrects.
        self.thresholds["uncertainty_threshold"] = float(np.percentile(u[y == y], 90))

        # --- OOD : score élevé = suspect ---
        # Seuil = percentile 95% sur validation in-distribution.
        self.thresholds["ood_threshold"] = float(np.percentile(o, 95))

        self._fitted = True
        return self.thresholds

    def predict(self, confidences, uncertainties, ood_scores):
        """
        Retourne un masque booléen : True = "échantillon sûr".
        """
        if not self._fitted:
            raise RuntimeError("SafetyThresholdEstimator must be fitted.")

        c = np.asarray(confidences)
        u = np.asarray(uncertainties)
        o = np.asarray(ood_scores)

        return (
            (c >= self.thresholds["confidence_threshold"]) &
            (u <= self.thresholds["uncertainty_threshold"]) &
            (o <= self.thresholds["ood_threshold"])
        )

    def save(self, path):
        torch.save({"thresholds": self.thresholds, "_fitted": self._fitted}, path)

    def load(self, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        self.thresholds = state["thresholds"]
        self._fitted = state["_fitted"]