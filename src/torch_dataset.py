import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class ArecanutDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_col='target'):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'filepath' and target columns.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_col (str): Column name for the target label.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.target_col = target_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx]['filepath']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image or handle error appropriately? 
            # Ideally we filtered this in step 1, but to be safe:
            image = Image.new('RGB', (224, 224))
            
        label = self.dataframe.iloc[idx][self.target_col]
        
        # Ensure label is appropriate type (int for classification)
        # If binary/multi-class, usually int or long tensor
        label = int(label)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(img_size=(224, 224), is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
