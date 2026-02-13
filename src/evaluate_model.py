import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import config
from torch_dataset import ArecanutDataset, get_transforms

def evaluate_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}\n")
    
    # Load data
    df = pd.read_csv(os.path.join(config.DATA_DIR, 'arecanut_dataset.csv'))
    
    # Encode labels
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['label'])
    
    # Use same split as training (random_state=42)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=config.RANDOM_SEED)
    
    print(f"Test Set Size: {len(test_df)} images")
    print("\nClass Distribution in Test Set:")
    print(test_df['label'].value_counts())
    print("\n" + "="*60)
    
    # Load mapping
    with open(os.path.join(config.DATA_DIR, 'universal_mapping.json'), 'r') as f:
        class_mapping = json.load(f)
    
    idx_to_class = {v: k for k, v in class_mapping.items()}
    num_classes = len(class_mapping)
    
    # Create test dataset
    test_dataset = ArecanutDataset(test_df, transform=get_transforms(is_train=False), target_col='target')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load model
    print("\nLoading model...")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'arecanut_model.pth'), map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.\n")
    
    # Evaluate
    all_preds = []
    all_labels = []
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    print("Running evaluation...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy
            correct = (preds == labels)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    # Calculate overall accuracy
    correct_total = sum(class_correct)
    total = len(all_labels)
    accuracy = correct_total / total
    
    print("\n" + "="*60)
    print(f"OVERALL TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"Correct: {correct_total}/{total}")
    print("="*60 + "\n")
    
    # Per-class accuracy
    print("Per-Class Accuracy:")
    print("="*60)
    for i in range(num_classes):
        class_name = idx_to_class[i]
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"{class_name:20s}: {class_acc:6.2f}%  ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"{class_name:20s}: No samples in test set")
    
    # Save results
    results_path = os.path.join(config.REPORTS_DIR, 'test_evaluation.txt')
    with open(results_path, 'w') as f:
        f.write(f"OVERALL TEST ACCURACY: {accuracy*100:.2f}%\n")
        f.write(f"Correct: {correct_total}/{total}\n\n")
        f.write("Per-Class Accuracy:\n")
        f.write("="*60 + "\n")
        for i in range(num_classes):
            class_name = idx_to_class[i]
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                f.write(f"{class_name:20s}: {class_acc:6.2f}%  ({class_correct[i]}/{class_total[i]})\n")
    
    print(f"\n\nResults saved to: {results_path}")
    return accuracy

if __name__ == "__main__":
    evaluate_model()
