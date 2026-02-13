import os
import sys
import json
import time

# Force torch cache to D drive
os.environ['TORCH_HOME'] = r'd:\Krishisethu\.cache\torch'

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import config

# ============================================================
# DISEASE KNOWLEDGE BASE
# ============================================================
DISEASE_INFO = {
    "Healthy": {
        "severity": "None",
        "severity_level": 0,
        "description": "The plant appears healthy with no visible signs of disease or pest damage.",
        "treatment": "No treatment required. Continue regular maintenance.",
        "prevention": [
            "Maintain proper spacing between plants",
            "Ensure adequate drainage",
            "Apply balanced NPK fertilizers periodically",
            "Regular inspection for early detection"
        ]
    },
    "Mahali_Koleroga": {
        "severity": "Critical",
        "severity_level": 3,
        "description": "Koleroga (Mahali/Fruit Rot) is a devastating fungal disease caused by Phytophthora palmivora. It causes rotting of nuts and affects yield severely.",
        "treatment": "Spray 1% Bordeaux Mixture immediately. Remove and destroy infected bunches. Apply Copper Oxychloride (0.25%) as preventive spray before monsoon.",
        "prevention": [
            "Pre-monsoon spraying of Bordeaux Mixture (1%)",
            "Remove fallen nuts and debris regularly",
            "Improve air circulation by pruning",
            "Avoid overcrowding of palms"
        ]
    },
    "Yellow_Leaf_Disease": {
        "severity": "High",
        "severity_level": 2,
        "description": "Yellow Leaf Disease (YLD) is caused by phytoplasma transmitted by lace bugs. Leaves turn yellow progressively from lower crown upwards.",
        "treatment": "Apply Potash-rich fertilizers (K2O). Use Tetracycline hydrochloride (500 ppm) as root feeding. Control insect vectors with appropriate insecticides.",
        "prevention": [
            "Regular application of organic manure",
            "Balanced fertilization with emphasis on Potash",
            "Control of Proutista moesta (lace bug vector)",
            "Remove severely affected palms to prevent spread"
        ]
    },
    "Stem_Cracking": {
        "severity": "Medium",
        "severity_level": 1,
        "description": "Stem cracking is often caused by nutritional deficiencies (especially Boron) or environmental stress. Longitudinal cracks appear on the trunk.",
        "treatment": "Apply Borax (50g/palm/year) to soil. Ensure adequate irrigation. Apply wound-healing paste to prevent secondary infections.",
        "prevention": [
            "Regular micronutrient supplementation",
            "Maintain consistent irrigation schedule",
            "Avoid mechanical damage to trunk",
            "Apply lime to acidic soils"
        ]
    },
    "Stem_bleeding": {
        "severity": "High",
        "severity_level": 2,
        "description": "Stem bleeding disease is caused by Thielaviopsis paradoxa. Dark brown liquid oozes from cracks in the stem, leading to tissue decay.",
        "treatment": "Chisel out infected tissue and apply Bordeaux paste. Apply Tridemorph (5%) on affected area. Provide adequate nutrition to boost recovery.",
        "prevention": [
            "Avoid injury to palm trunk",
            "Apply Bordeaux paste on wounds immediately",
            "Maintain palm vigor through balanced nutrition",
            "Ensure proper drainage around palm base"
        ]
    },
    "Bud_Borer": {
        "severity": "Critical",
        "severity_level": 3,
        "description": "Bud Borer (Oryctes rhinoceros) is a serious pest that bores into the central shoot/crown, damaging the growing point and potentially killing the palm.",
        "treatment": "Fill crown with a mixture of Sevidol 8G (or Carbaryl 10%) with fine sand. Apply Naphthalene balls in leaf axils. Use pheromone traps.",
        "prevention": [
            "Maintain field hygiene - remove decaying organic matter",
            "Hook out beetles from crown periodically",
            "Install rhinoceros beetle pheromone traps",
            "Apply Metarhizium anisopliae to manure pits"
        ]
    },
    "Not_Arecanut": {
        "severity": "N/A",
        "severity_level": -1,
        "description": "The uploaded image does not appear to be an Arecanut (Areca catechu) plant. This system is specialized for Arecanut disease classification only.",
        "treatment": "Please upload an image of an Arecanut leaf, nut, trunk, or any plant part for accurate disease diagnosis.",
        "prevention": []
    }
}

# ============================================================
# MODEL LOADING
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[ENGINE] Inference Device: {device}")

# Load class mapping
mapping_path = os.path.join(config.DATA_DIR, 'universal_mapping.json')
with open(mapping_path, 'r') as f:
    class_mapping = json.load(f)
idx_to_class = {v: k for k, v in class_mapping.items()}
num_classes = len(class_mapping)
print(f"[ENGINE] Loaded {num_classes} classes")

# Load model
model_path = os.path.join(config.MODELS_DIR, 'arecanut_model.pth')
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"[ENGINE] Model loaded from {model_path}")

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="KrishiSethu AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "device": str(device),
        "model": "MobileNetV3-Small",
        "classes": num_classes,
        "accuracy": "99.84%"
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    # Read image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        all_probs = probs[0].cpu().numpy()

    pred_class = idx_to_class[pred_idx.item()]
    conf_value = float(confidence.item())

    # Build probability map
    prob_map = {}
    for idx, prob in enumerate(all_probs):
        prob_map[idx_to_class[idx]] = round(float(prob) * 100, 2)

    # ================================================================
    # SMART CONFIDENCE THRESHOLDING
    # Prevents misclassifying non-arecanut images as diseases
    # ================================================================
    low_confidence = False
    override_reason = ""

    # Sort probabilities to find top-2
    sorted_probs = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    not_arecanut_prob = prob_map.get("Not_Arecanut", 0)

    # Rule 1: Overall confidence too low (model is unsure)
    CONFIDENCE_THRESHOLD = 85.0
    if conf_value * 100 < CONFIDENCE_THRESHOLD and pred_class != "Not_Arecanut":
        low_confidence = True
        override_reason = f"Low confidence ({conf_value*100:.1f}%). The model is uncertain about this image."

    # Rule 2: Not_Arecanut probability is significant (top-3 and > 10%)
    top3_classes = [c for c, _ in sorted_probs[:3]]
    if pred_class != "Not_Arecanut" and "Not_Arecanut" in top3_classes and not_arecanut_prob > 10:
        low_confidence = True
        override_reason = f"Not_Arecanut probability is significant ({not_arecanut_prob:.1f}%). This may not be an Arecanut plant."

    # Rule 3: If confidence is very low (below 60%), force Not_Arecanut
    FORCE_THRESHOLD = 60.0
    if conf_value * 100 < FORCE_THRESHOLD and pred_class != "Not_Arecanut":
        pred_class = "Not_Arecanut"
        low_confidence = True
        override_reason = f"Very low confidence ({conf_value*100:.1f}%). Overridden to Not_Arecanut for safety."

    # Rule 4: If entropy is high (model is confused among multiple classes)
    top2_diff = sorted_probs[0][1] - sorted_probs[1][1] if len(sorted_probs) >= 2 else 100
    if top2_diff < 20 and pred_class != "Not_Arecanut":
        low_confidence = True
        override_reason = f"Top predictions are close ({sorted_probs[0][0]}: {sorted_probs[0][1]:.1f}% vs {sorted_probs[1][0]}: {sorted_probs[1][1]:.1f}%). Model is uncertain."

    # Get disease info
    info = DISEASE_INFO.get(pred_class, DISEASE_INFO['Not_Arecanut'])

    inference_time = round((time.time() - start_time) * 1000, 1)

    result = {
        "class_name": pred_class,
        "display_name": pred_class.replace('_', ' '),
        "confidence": round(conf_value * 100, 2),
        "is_arecanut": pred_class != "Not_Arecanut",
        "inference_ms": inference_time,
        "severity": info["severity"],
        "severity_level": info["severity_level"],
        "description": info["description"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "all_probabilities": prob_map,
        "image_size": {"width": image.size[0], "height": image.size[1]},
        "filename": file.filename,
        "low_confidence": low_confidence,
        "confidence_warning": override_reason,
    }

    if low_confidence:
        print(f"[WARNING] Low confidence prediction: {pred_class} ({conf_value*100:.1f}%) | {override_reason}")

    return result
