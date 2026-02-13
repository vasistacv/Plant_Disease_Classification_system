import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import config

class ArcanutSystem:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Inference Device: {self.device}")
        
        # Single Universal Model
        self.model_path = os.path.join(config.MODELS_DIR, 'arecanut_model.pth')
        self.mapping_path = os.path.join(config.DATA_DIR, 'universal_mapping.json')
        
        self.model = None
        self.idx_to_class = None
        
        # Define Transforms (Same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_models(self):
        try:
            # 1. Load Universal Mapping
            if os.path.exists(self.mapping_path):
                with open(self.mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                # Invert: {0: 'Bud_Borer', 3: 'Not_Arecanut', ...}
                self.idx_to_class = {v: k for k, v in class_mapping.items()}
                num_classes = len(class_mapping)
                print(f"Loaded {num_classes} classes mapping.")
            else:
                raise FileNotFoundError("universal_mapping.json not found! Train the model first.")
            
            # 2. Load Single Model
            print(f"Loading Universal Model from {self.model_path}...")
            self.model = models.mobilenet_v3_small(weights=None)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
            
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, image_path):
        if self.model is None:
            self.load_models()

        try:
            # Preprocess
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
                
            pred_idx = pred_idx.item()
            confidence = confidence.item()
            class_name = self.idx_to_class[pred_idx]
            
            # Structure the Result
            result = {
                'class_name': class_name,
                'confidence': confidence,
                'is_arecanut': True, # Default Assume True
                'message': '',
                'disease': None
            }
            
            # Logic: Handle "Not_Arecanut"
            if class_name == 'Not_Arecanut':
                result['is_arecanut'] = False
                result['message'] = f"Non-Arecanut Plant Detected ({confidence*100:.1f}%)"
                return result
            
            # Logic: It IS Arecanut
            result['is_arecanut'] = True
            result['disease'] = class_name
            result['message'] = f"Arecanut Detected: {class_name}"
            
            return result
            
        except Exception as e:
            return {'error': str(e)}

if __name__ == "__main__":
    system = ArcanutSystem()
    system.load_models()
    # Test with dummy path if running directly
    print("System initialized. Call predict(image_path) to use.")
