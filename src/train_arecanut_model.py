import os
# STRICTLY FORCE CACHE TO D DRIVE
os.environ['TORCH_HOME'] = r'd:\Krishisethu\.cache\torch'

import time
import copy
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models
from sklearn.preprocessing import LabelEncoder
import config
from torch_dataset import ArecanutDataset, get_transforms
from tqdm import tqdm

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Progress Bar
            loop = tqdm(dataloaders[phase], desc=f'{phase}', leave=True)
            
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                loop.set_postfix(loss=loss.item())

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Combined Data
    data_path = os.path.join(config.DATA_DIR, 'arecanut_dataset.csv')
    df = pd.read_csv(data_path)
    
    # Encoding Labels
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['label'])
    
    # Save Class Mapping
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"Class Mapping: {mapping}")
    with open(os.path.join(config.DATA_DIR, 'universal_mapping.json'), 'w') as f:
        json.dump(mapping, f)
        
    num_classes = len(le.classes_)
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=config.RANDOM_SEED)
    
    # Datasets
    train_dataset = ArecanutDataset(train_df, transform=get_transforms(is_train=True), target_col='target')
    val_dataset = ArecanutDataset(val_df, transform=get_transforms(is_train=False), target_col='target')
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print(f"Dataset Sizes: {dataset_sizes}")
    
    # Model
    model = models.mobilenet_v3_small(weights='DEFAULT')
    
    # Modify Classifier
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    print("\n--- Phase 1: Training Head (4 Epochs) ---")
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=4, device=device)
    
    print("\n--- Phase 2: Fine-Tuning (2 Epochs) ---")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-5)
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, None, num_epochs=2, device=device)
    
    save_path = os.path.join(config.MODELS_DIR, 'arecanut_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
