import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- IMPORTS PROJET ---
# Si les scripts sont à la racine, ceci fonctionne.
# Sinon, adapter selon l'arborescence.
try:
    import df_utils as dm
    from torch_dataloader import ImageDataFrameDataset
    # On essaie d'importer depuis le dossier local ou challenge_solution
    try:
        from AIComponent import MyAIComponent
    except ImportError:
        from challenge_solution.AIComponent import MyAIComponent
except ImportError as e:
    print(f"Erreur d'import : {e}")
    print("Assure-toi que df_utils.py, torch_dataloader.py et AIComponent.py sont accessibles.")
    sys.exit(1)

# --- CONFIGURATION ---
BATCH_SIZE = 32         # Ajuster selon VRAM (32 ou 64 pour 16GB VRAM)
LR = 3e-4               # Learning rate standard pour finetuning
EPOCHS = 10             # Suffisant pour converger sur ce dataset

# train.py est dans 'weldings-project/challenge_solution/'
# On remonte d'un cran (..) pour trouver le dossier 'imgs'
ROOT_IMG_DIR = "../imgs/" 
# On peut mettre le cache à la racine du projet ou dans le dossier courant
CSV_CACHE_DIR = "../"

# --- 1. SETUP ---
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# --- 2. DATASETS ---
# Augmentation forte pour l'entrainement (Robustesse)
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), # Rotation +/- 15 deg comme dans l'ODD
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Luminosité variable comme ODD
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Juste normalisation pour la validation
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("📂 Loading Dataframes...")
df_data = dm.explore_csv_hierarchy(
    CSV_CACHE_DIR,
    depth_name_list=['folder_1','folder_2','folder_3','seam','decision','type_label'],
    allowed_ext='.jpeg'
)

# Filtrage et Mapping
mapping = {'OK': 0, 'KO': 1}
df_data = df_data[df_data['decision'].isin(['OK', 'KO'])].copy()
df_data['label'] = df_data['decision'].map(mapping)

# Split Stratifié
df_train, df_val = dm.stratified_train_val_split(
    df_data, ['seam','decision'], alpha=0.95, random_state=42
)

# Création Datasets & Loaders
train_ds = ImageDataFrameDataset(df=df_train, root_dir=ROOT_IMG_DIR, path_col="path", label_col="label", transform=train_transform, channels_first=True)
val_ds = ImageDataFrameDataset(df=df_val, root_dir=ROOT_IMG_DIR, path_col="path", label_col="label", transform=val_transform, channels_first=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- 3. MODELE ---
print("Initializing AI Component...")
ai_component = MyAIComponent(device=device.type)
ai_component.init_model(use_weights=True)

# --- 4. TRAIN LOOP (Backbone) ---
optimizer = torch.optim.AdamW(ai_component.model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    # --- Train ---
    ai_component.model.train()
    train_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        out = ai_component.model(imgs)
        loss = criterion(out['logits'], labels) # Note: on prend 'logits'
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    scheduler.step()
    
    # --- Validation ---
    ai_component.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = ai_component.model(imgs)
            _, predicted = torch.max(out['logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    avg_loss = train_loss / len(train_loader)
    print(f"   End Epoch {epoch+1}: Loss={avg_loss:.4f} | Val Acc={val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Sauvegarde temporaire des poids seulement
        torch.save(ai_component.model.state_dict(), "temp_backbone.pth")

print(f"Training finished. Best Acc: {best_val_acc:.2f}%")

# --- 5. FITTING TRUSTWORTHY MODULES ---
print("\nFitting Trustworthy Modules...")

# Recharger le meilleur modèle
ai_component.model.load_state_dict(torch.load("temp_backbone.pth"))
ai_component.model.eval()

# A. Fit Mahalanobis (sur Train set propre)
print("   1/2 Extracting Train embeddings for OOD...")
train_embeddings = []
# On utilise un loader sans shuffle pour aller plus vite, mais shuffle=True ne gêne pas
with torch.no_grad():
    for imgs, _ in tqdm(train_loader):
        imgs = imgs.to(device)
        out = ai_component.model(imgs)
        train_embeddings.append(out['embedding'].cpu().numpy())

ai_component.ood_detector.fit(np.vstack(train_embeddings))

# B. Fit Thresholds (sur Validation set)
print("   2/2 Calibrating Thresholds on Validation...")
val_embeddings = []
val_confidences = []
val_labels_list = []

with torch.no_grad():
    for imgs, labels in tqdm(val_loader):
        imgs = imgs.to(device)
        out = ai_component.model(imgs)
        
        probs = out['probabilities'].cpu().numpy()
        emb = out['embedding'].cpu().numpy()
        
        val_confidences.extend(np.max(probs, axis=1))
        val_embeddings.append(emb)
        val_labels_list.extend(labels.numpy())

# Calcul distances sur Validation
mahal_dists = ai_component.ood_detector.predict(np.vstack(val_embeddings))

# Calibration
ai_component.threshold_estimator.fit_from_validation(
    confidences=np.array(val_confidences),
    mahalanobis_dists=mahal_dists,
    labels=np.array(val_labels_list)
)

print(f"   Final Thresholds: {ai_component.threshold_estimator.thresholds}")

# --- 6. SAUVEGARDE FINALE ---
final_path = "best_model.pth"
# Création du dossier si inexistant
os.makedirs(os.path.dirname(final_path), exist_ok=True)

ai_component.save_model(final_path, extras={
    "ood_detector": ai_component.ood_detector.state_dict(),
    "thresholds": ai_component.threshold_estimator.state_dict()
})

# Cleanup
if os.path.exists("temp_backbone.pth"):
    os.remove("temp_backbone.pth")

print(f"\nSUCCESS! Model saved to {final_path}")