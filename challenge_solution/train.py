import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# IMPORTS 
try:
    import df_utils as dm
    from torch_dataloader import ImageDataFrameDataset
    try:
        from AIComponent import MyAIComponent
    except ImportError:
        from challenge_solution.AIComponent import MyAIComponent
except ImportError as e:
    print(f" Erreur d'import : {e}")
    sys.exit(1)

# config
BATCH_SIZE = 32         
LR = 3e-4               
EPOCHS = 10             

# DÉFINITION DES CHEMINS 
# Le dossier dataset est à la racine de weldings-project, donc ".." depuis challenge_solution
DATASET_FOLDER_NAME = "example_mini_dataset"
DATASET_LOCAL_PATH = os.path.join("..", DATASET_FOLDER_NAME)
PARQUET_PATH = os.path.join(DATASET_LOCAL_PATH, "metadata/ds_meta.parquet")

# 1. setup
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🔧 Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("🔧 Using CPU")

# 2. datasets
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f" Loading Metadata from: {PARQUET_PATH}")

if not os.path.exists(PARQUET_PATH):
    print(f" ERREUR: Le fichier {PARQUET_PATH} est introuvable.")
    print(f"   Chemin actuel d'exécution: {os.getcwd()}")
    sys.exit(1)

try:
    df_data = pd.read_parquet(PARQUET_PATH)
except Exception as e:
    print(f" Impossible de lire le parquet: {e}")
    sys.exit(1)

#  NETTOYAGE DES CHEMINS

def fix_path_for_local(p):
    p = str(p)
    # On coupe le chemin au mot clé "example_mini_dataset"
    if DATASET_FOLDER_NAME in p:
        
        relative_part = p.split(DATASET_FOLDER_NAME)[-1]
        
        # On enlève les slashs initiaux éventuels
        relative_part = relative_part.lstrip('/\\')
        
        # On reconstruit
        return os.path.join(DATASET_LOCAL_PATH, relative_part)
    return p

# On applique la correction
path_col = 'path' 
df_data['full_path'] = df_data[path_col].apply(fix_path_for_local)

#  LABELS -
label_col = 'class' 
# On ne garde que les lignes valides (OK ou KO)
df_data = df_data[df_data[label_col].isin(['OK', 'KO'])].copy()

mapping = {'OK': 0, 'KO': 1}
df_data['target'] = df_data[label_col].map(mapping).astype(int)

# Split Train/Val
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df_data, test_size=0.2, stratify=df_data['target'], random_state=42)

print(f"   -> Train size: {len(train_df)} | Val size: {len(val_df)}")
# Affichage de débug pour être sûr
print(f"   -> Exemple de chemin corrigé: {train_df['full_path'].iloc[0]}")

# Création Datasets

train_ds = ImageDataFrameDataset(df=train_df, root_dir="", path_col="full_path", label_col="target", transform=train_transform, channels_first=True)
val_ds = ImageDataFrameDataset(df=val_df, root_dir="", path_col="full_path", label_col="target", transform=val_transform, channels_first=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 3. Modele
print(" Initializing AI Component...")
ai_component = MyAIComponent(device=device.type)
ai_component.init_model(use_weights=True)

#  TRAIN LOOP 
optimizer = torch.optim.AdamW(ai_component.model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
print(f" Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    # Train
    ai_component.model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = ai_component.model(imgs)
        loss = criterion(out['logits'], labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    scheduler.step()
    
    # Validation
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
        torch.save(ai_component.model.state_dict(), "temp_backbone.pth")

print(f" Training finished. Best Acc: {best_val_acc:.2f}%")

#  5. FITTING TRUSTWORTHY MODULES

if os.path.exists("temp_backbone.pth"):
    ai_component.model.load_state_dict(torch.load("temp_backbone.pth"))
else:
    print(" Pas de backup trouvé, on continue avec les poids actuels.")
ai_component.model.eval()

# A. Fit Mahalanobis
print("   1/2 Extracting Train embeddings for OOD...")
train_embeddings = []
with torch.no_grad():
    for imgs, _ in tqdm(train_loader):
        imgs = imgs.to(device)
        out = ai_component.model(imgs)
        train_embeddings.append(out['embedding'].cpu().numpy())

ai_component.ood_detector.fit(np.vstack(train_embeddings))

# B. Fit Thresholds
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

mahal_dists = ai_component.ood_detector.predict(np.vstack(val_embeddings))
ai_component.threshold_estimator.fit_from_validation(
    confidences=np.array(val_confidences),
    mahalanobis_dists=mahal_dists,
    labels=np.array(val_labels_list)
)

print(f"    Final Thresholds: {ai_component.threshold_estimator.thresholds}")

# SAUVEGARDE FINALE 
final_path = "best_model.pth"
ai_component.save_model(final_path, extras={
    "ood_detector": ai_component.ood_detector.state_dict(),
    "thresholds": ai_component.threshold_estimator.state_dict()
})

if os.path.exists("temp_backbone.pth"):
    os.remove("temp_backbone.pth")

print(f"\n SUCCESS! Model saved to {final_path}")
