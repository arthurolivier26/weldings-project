import numpy as np
import torch
from pathlib import Path

import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from AIComponent import MyAIComponent
import df_utils as dm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch_dataloader import ImageDataFrameDataset

# Exemple de transform basique
transform = transforms.Compose([transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


df_data = dm.explore_csv_hierarchy('../../notebooks_cache',depth_name_list=['folder_1','folder_2','folder_3','seam','decision','type_label'],allowed_ext='.jpeg')
mapping = {'OK': 0, 'KO': 1}
df_data['label'] = df_data['decision'].map(mapping)
df_train,df_val = dm.stratified_train_val_split(df_data, ['seam','decision'], alpha=0.95, random_state=42)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🔧 Using CUDA")
else:
    device = torch.device('cpu')
    print(f"🔧 Using CPU")

Train_Dataset = ImageDataFrameDataset(df=df_train,root_dir="../Challenge-Welding-Reference-Solution-1/",path_col="path",label_col="label",transform=transform,channels_first=True)
Val_Dataset = ImageDataFrameDataset(df=df_train,root_dir="../Challenge-Welding-Reference-Solution-1/",path_col="path",label_col="label",transform=transform,channels_first=True)

ai_component = MyAIComponent()
ai_component.init_model()
ai_component.load_model()
ai_component.train_model(Train_Dataset,
                         Val_Dataset,
                         device=device,
                         save_path="best_model.pth",
                         augmentation_fn=None,
                         preprocess_fn=None,
                         epochs=50,
                         batch_size=128,
                         lr=3e-4)