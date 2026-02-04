import os
# STRICTLY FORCE CACHE TO D DRIVE
os.environ['TORCH_HOME'] = r'd:\Krishisethu\.cache\torch'
import time
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config
from torch_dataset import ArecanutDataset, get_transforms

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            from tqdm import tqdm
            loop = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch+1}/{num_epochs}', leave=True)
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1) # Float for BCELoss

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # Sigmoid already in model? Or use BCEWithLogitsLoss?
                    # Generally safer to use BCEWithLogitsLoss (no sigmoid in model) or sigmoid+BCELoss
                    # We will assume model output is logits, so we use BCEWithLogitsLoss
                    loss = criterion(outputs, labels)
                    
                    preds = (torch.sigmoid(outputs) > 0.5).long()

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Live Progress Update
                loop.set_postfix(loss=loss.item())

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    df = pd.read_csv(os.path.join(config.DATA_DIR, 'full_dataset.csv'))
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['is_arecanut'], random_state=config.RANDOM_SEED)
    
    # Datasets
    train_dataset = ArecanutDataset(train_df, transform=get_transforms(is_train=True), target_col='is_arecanut')
    val_dataset = ArecanutDataset(val_df, transform=get_transforms(is_train=False), target_col='is_arecanut')
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }
    
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print(f"Dataset Sizes: {dataset_sizes}")
    
    # Model: MobileNetV3 Small (Pretrained)
    model = models.mobilenet_v3_small(weights='DEFAULT')
    
    # Freeze Feature Layers (Phase 1)
    for param in model.parameters():
        param.requires_grad = False
        
    # Modify Classifier
    # MobileNetV3 classifier structure: Sequential(Linear, Hardswish, Dropout, Linear)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    
    # Move to GPU
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Phase 1: Train Head
    print("\n--- Phase 1: Training Head ---")
    model, hist1 = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=5, device=device)
    
    # Phase 2: Fine-Tuning
    print("\n--- Phase 2: Fine-Tuning ---")
    for param in model.parameters():
        param.requires_grad = True # Unfreeze all
        
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-5) # Low LR
    
    model, hist2 = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, None, num_epochs=6, device=device)
    
    # Save Model
    save_path = os.path.join(config.MODELS_DIR, 'arecanut_id_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Plot History
    # (Simplified plotting code here)
    
if __name__ == "__main__":
    main()
