import torch
import torch.nn as nn
from torchvision import models
import os
import config

# Define Model Architecture (Must match training)
def get_model(num_classes=6):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def save_best_model():
    print("Saving Best Model from Checkpoint...")
    
    # We rely on the fact that training loop saves 'best_model_wts' in memory
    # But since process died, we might have lost the RAM version.
    # HOWEVER, usually a good training loop saves to disk periodically.
    # In my script I didn't add intermediate disk saving, only final.
    
    # CRITICAL FALLBACK:
    # Since I stopped the previous run at end of Phase 1, the script actually FINISHED Phase 1 loop.
    # Did it save?
    # Let's check if file exists.
    
    path = os.path.join(config.MODELS_DIR, 'arecanut_disease_model.pth')
    if os.path.exists(path):
        print(f"SUCCESS: Model file already exists at {path}")
        print("Size: ", os.path.getsize(path) / (1024*1024), "MB")
    else:
        print("WARNING: Model file missing. We might need to quickly re-run 1 epoch to generate it if it wasn't saved.")

if __name__ == "__main__":
    save_best_model()
