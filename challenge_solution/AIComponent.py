import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from torchvision.models import mobilenet_v2

warnings.filterwarnings('ignore')

# =============================================================================
# DEFINITIONS IDENTIQUES A L'ENTRAINEMENT (COLAB)
# =============================================================================

class MobileNetV2Backbone(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = mobilenet_v2(weights=None) 
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base.last_channel, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        emb = torch.flatten(feat, 1)
        logits = self.classifier(emb)
        probs = F.softmax(logits, dim=1)
        return {"logits": logits, "probabilities": probs, "embedding": emb}

class MahalanobisOODDetector:
    def __init__(self, n_components=64):
        self.n_components = n_components
        self.pca = None
        self.cov = EmpiricalCovariance(assume_centered=False)
        self._fitted = False

    def predict(self, embeddings):
        if not self._fitted: return np.zeros(len(embeddings))
        X = embeddings
        if self.pca: X = self.pca.transform(X)
        return self.cov.mahalanobis(X)
    
    def load_state_dict(self, state):
        self.n_components = state['n_components']
        self.pca = state['pca']
        self.cov = state['cov']
        self._fitted = state['_fitted']

class SafetyThresholdEstimator:
    def __init__(self):
        self.thresholds = {"confidence": 0.6, "ood": 100.0}
        self._fitted = False
    
    def load_state_dict(self, state):
        self.thresholds = state['thresholds']
        self._fitted = state['_fitted']

# =============================================================================
# INTERFACE DU CHALLENGE
# =============================================================================

class AbstractAIComponent(ABC):
    @abstractmethod
    def load_model(self, config_file=None): pass
    @abstractmethod
    def predict(self, input_images, images_meta_informations): pass

class MyAIComponent(AbstractAIComponent):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialisation des modules Trustworthy
        self.ood_detector = MahalanobisOODDetector(n_components=64)
        self.thresholds = SafetyThresholdEstimator()
        
        # Preprocessing (Identique à la validation)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['OK', 'KO', 'UNKNOWN']

    def load_model(self, config_file=None):
        # Chemin relatif robuste
        model_path = Path(__file__).parent / 'best_model.pth'
        
        print(f"🔧 Loading model from {model_path}...")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Chargement du dictionnaire global
        # CORRECTION PYTORCH 2.6 : weights_only=False nécessaire pour charger Scikit-Learn
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 1. Charger le Backbone
        self.model = MobileNetV2Backbone(num_classes=3)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Charger les modules Trustworthy
        if 'ood_detector' in checkpoint:
            self.ood_detector.load_state_dict(checkpoint['ood_detector'])
        if 'thresholds' in checkpoint:
            self.thresholds.load_state_dict(checkpoint['thresholds'])
            
        print(f"✅ Model Loaded! Thresholds: {self.thresholds.thresholds}")

    def predict(self, input_images, images_meta_informations=None):
        if self.model is None:
            raise RuntimeError("Model not loaded!")
            
        results = {"predictions": [], "probabilities": [], "OOD_scores": []}
        
        # Batch inference pour la vitesse
        tensors = [self.preprocess(img) for img in input_images]
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs = outputs['probabilities'].cpu().numpy()
            embeddings = outputs['embedding'].cpu().numpy()
            
            # Calcul OOD
            mahal_dists = self.ood_detector.predict(embeddings)
            
            # Application des seuils
            #conf_thresh = self.thresholds.thresholds.get('confidence', 0.5)
            #ood_thresh = self.thresholds.thresholds.get('ood', 100.0)
            
            # --- RÉGLAGE "IA DE CONFIANCE INDUSTRIELLE" ---
            
            # 1. Seuil de confiance : 0.65 ou 0.70
            # On demande une majorité qualifiée, pas une certitude absolue.
            # Cela permet de récupérer les défauts subtils sans accepter n'importe quoi.
            conf_thresh = 0.65 
            
            # 2. Seuil OOD : Calibré + Marge de sécurité
            # Votre calibration donnait ~105. On ajoute 20% de marge pour la variabilité.
            # On ne veut rejeter que ce qui est VRAIMENT anormal (OOD), pas juste "difficile".
            ood_thresh = 120.0
            
            
            for i in range(len(input_images)):
                p = probs[i]
                dist = mahal_dists[i]
                pred_idx = np.argmax(p)
                confidence = p[pred_idx]
                
                # Normalisation du score OOD pour le challenge (doit être > 1 si OOD)
                # Score = distance / seuil
                final_ood_score = dist / (ood_thresh + 1e-6)
                
                # Logique de décision (Trustworthy)
                is_ood = final_ood_score >= 1.0
                is_uncertain = confidence < conf_thresh
                
                if is_ood or is_uncertain:
                    pred_label = 'UNKNOWN'
                    # On lisse les probas pour refléter l'incertitude
                    final_probs = [0.33, 0.33, 0.34] 
                else:
                    pred_label = self.class_names[pred_idx]
                    final_probs = p.tolist()
                
                results["predictions"].append(pred_label)
                results["probabilities"].append(final_probs)
                results["OOD_scores"].append(float(final_ood_score))
                
        return results