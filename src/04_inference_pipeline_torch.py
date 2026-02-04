import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import config

class ArcanutSystem:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Inference Device: {self.device}")
        
        self.id_model_path = os.path.join(config.MODELS_DIR, 'arecanut_id_model.pth')
        self.disease_model_path = os.path.join(config.MODELS_DIR, 'arecanut_disease_model.pth')
        self.mapping_path = os.path.join(config.DATA_DIR, 'class_mapping.json')
        
        self.id_model = None
        self.disease_model = None
        self.class_mapping = None
        self.idx_to_class = None
        
        # Define Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_models(self):
        try:
            # 1. Load Identification Model
            print("Loading Identification Model...")
            self.id_model = models.mobilenet_v3_small(weights=None)
            self.id_model.classifier[3] = nn.Linear(self.id_model.classifier[3].in_features, 1)
            self.id_model.load_state_dict(torch.load(self.id_model_path, map_location=self.device))
            self.id_model.to(self.device)
            self.id_model.eval()
            
            # 2. Load Class Mapping
            if os.path.exists(self.mapping_path):
                with open(self.mapping_path, 'r') as f:
                    self.class_mapping = json.load(f)
                # Invert mapping: {0: 'Healthy', ...}
                self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
                num_classes = len(self.class_mapping)
            else:
                print("Warning: Class mapping not found. Disease model might fail.")
                num_classes = 6 # Default fallback
            
            # 3. Load Disease Model
            print("Loading Disease Model...")
            self.disease_model = models.mobilenet_v3_small(weights=None)
            self.disease_model.classifier[3] = nn.Linear(self.disease_model.classifier[3].in_features, num_classes)
            self.disease_model.load_state_dict(torch.load(self.disease_model_path, map_location=self.device))
            self.disease_model.to(self.device)
            self.disease_model.eval()
            
        except Exception as e:
            print(f"Error loading models: {e}")

    def predict(self, image_path):
        if self.id_model is None or self.disease_model is None:
            self.load_models()

        try:
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 1. Identification
            with torch.no_grad():
                output = self.id_model(img_tensor)
                prob = torch.sigmoid(output).item()
                
            results = {
                'is_arecanut_prob': prob,
                'is_arecanut': prob > 0.5,
                'message': '',
                'disease': None,
                'disease_confidence': 0.0
            }
            
            if not results['is_arecanut']:
                results['message'] = f"Not Arecanut (Confidence: {(1-prob)*100:.2f}%)"
                return results
            
            results['message'] = f"Arecanut Verified (Confidence: {prob*100:.2f}%)"
            
            # 2. Disease Classification
            with torch.no_grad():
                output = self.disease_model(img_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
            pred_idx = pred_idx.item()
            conf = conf.item()
            
            disease_name = self.idx_to_class.get(pred_idx, "Unknown")
            
            results['disease'] = disease_name
            results['disease_confidence'] = conf
            results['final_output'] = f"{disease_name} ({conf*100:.2f}%)"
            
            return results
            
        except Exception as e:
            return {'error': str(e)}

if __name__ == "__main__":
    import sys
    sys_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    system = ArcanutSystem()
    if sys_path:
        print(system.predict(sys_path))
    else:
        print("Provide image path.")
