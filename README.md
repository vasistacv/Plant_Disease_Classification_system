# 🌿 KrishiSethu AI — Arecanut Disease Classification System

> Enterprise-grade AI-powered diagnostic system for real-time Arecanut plant health monitoring and disease classification with **99.84% accuracy**.

---

## 📌 Project Overview

KrishiSethu AI is a deep learning-based plant disease classification system specifically designed for **Arecanut (Areca catechu)** plants. The system can:

- **Identify** whether an uploaded image is an Arecanut plant or not
- **Classify diseases** among 6 Arecanut-specific conditions + 1 non-arecanut class
- **Provide treatment recommendations** and prevention guidelines
- **Display confidence scores** and probability distributions

### 🎯 Problem Statement

Arecanut farmers face significant crop losses due to diseases like Koleroga, Yellow Leaf Disease, and Bud Borer. Manual identification requires expert knowledge and is time-consuming. This system provides **instant, automated diagnosis** using a smartphone image.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER (Browser)                          │
│                  http://localhost:3000                       │
└────────────────────────┬────────────────────────────────────┘
                         │ Image Upload
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              REACT FRONTEND (Vite)                          │
│  • Premium Light Enterprise Dashboard                       │
│  • Drag & Drop Image Upload                                 │
│  • Real-time Results with Confidence Bars                   │
│  • Treatment Recommendations Display                        │
│  • Scan History & Session Analytics                         │
└────────────────────────┬────────────────────────────────────┘
                         │ POST /api/predict
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FASTAPI BACKEND (Python)                       │
│  • Image Preprocessing (224x224, Normalization)             │
│  • PyTorch Model Inference                                  │
│  • Smart Confidence Thresholding                            │
│  • Disease Knowledge Base                                   │
│  • CORS-enabled REST API                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ Forward Pass
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              DEEP LEARNING MODEL                            │
│  • Architecture: MobileNetV3-Small (Transfer Learning)      │
│  • Framework: PyTorch                                       │
│  • Accuracy: 99.84% (3085/3090 correct)                    │
│  • Classes: 7 (6 Arecanut conditions + Non-Arecanut)       │
│  • Input: 224×224 RGB images                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Detectable Conditions (7 Classes)

| # | Class | Type | Description |
|---|-------|------|-------------|
| 1 | **Healthy** | ✅ Normal | No disease detected |
| 2 | **Mahali Koleroga** | 🔴 Critical | Fruit rot caused by *Phytophthora palmivora* |
| 3 | **Yellow Leaf Disease** | 🟠 High | Phytoplasma infection transmitted by lace bugs |
| 4 | **Stem Cracking** | 🟡 Medium | Nutritional deficiency (Boron) or environmental stress |
| 5 | **Stem Bleeding** | 🟠 High | Caused by *Thielaviopsis paradoxa* |
| 6 | **Bud Borer** | 🔴 Critical | Pest damage by *Oryctes rhinoceros* beetle |
| 7 | **Not Arecanut** | ⚪ N/A | Image is not an Arecanut plant |

---

## 📊 Model Performance

```
OVERALL TEST ACCURACY: 99.84%
Correct: 3085 / 3090

Per-Class Accuracy:
═══════════════════════════════════════════
Bud Borer            :  97.22%  (35/36)
Healthy              :  99.70%  (991/994)
Mahali Koleroga      : 100.00%  (641/641)
Not Arecanut         : 100.00%  (877/877)
Stem Cracking        : 100.00%  (135/135)
Stem Bleeding        :  97.37%  (37/38)
Yellow Leaf Disease  : 100.00%  (369/369)
```

---

## 📁 Project Structure

```
d:\Krishisethu\
│
├── 📂 src/                          # Core ML Pipeline
│   ├── config.py                    # Path configurations
│   ├── 01_data_preparation.py       # Step 1: Dataset organization
│   ├── prepare_combined_dataset.py  # Step 2: CSV dataset creation
│   ├── torch_dataset.py             # Step 3: PyTorch Dataset class
│   ├── 02_train_id_torch.py         # Step 4a: Identification model training
│   ├── 03_train_disease_torch.py    # Step 4b: Disease model training
│   ├── train_arecanut_model.py      # Step 4c: Final unified model training
│   ├── evaluate_model.py            # Step 5: Model evaluation & metrics
│   └── 04_inference_pipeline_torch.py  # Step 6: Inference pipeline class
│
├── 📂 backend/                      # FastAPI Backend Server
│   └── server.py                    # API endpoint + model serving
│
├── 📂 frontend/                     # React Dashboard (Vite)
│   ├── index.html                   # Entry HTML
│   ├── package.json                 # Node.js dependencies
│   └── src/
│       ├── main.jsx                 # React entry point
│       ├── App.jsx                  # Main application component
│       ├── index.css                # Premium light theme CSS
│       └── components/
│           ├── Header.jsx           # Dashboard header
│           ├── Sidebar.jsx          # Side navigation & stats
│           ├── MetricsBar.jsx       # Key metrics display
│           ├── UploadZone.jsx       # Image upload (drag & drop)
│           ├── ResultPanel.jsx      # Disease diagnosis results
│           └── History.jsx          # Scan history log
│
├── 📂 models/                       # Trained Models
│   ├── arecanut_model.pth           # Final trained model (6MB)
│   └── backups/                     # Model backups
│
├── 📂 data_processed/               # Processed Data
│   ├── universal_mapping.json       # Class name ↔ index mapping
│   ├── arecanut_dataset.csv         # Arecanut-only dataset
│   ├── disease_dataset.csv          # Disease classification dataset
│   └── full_dataset.csv             # Combined full dataset
│
├── 📂 reports/                      # Evaluation Reports
│   ├── test_evaluation.txt          # Final test accuracy report
│   └── training_log_id.csv          # Training metrics log
│
├── 📂 Arecanut_data/                # Raw Training Data
│   ├── Arecanut_dataset/            # Arecanut images (train/test)
│   └── final_testing-*/             # Final test images
│
├── 📂 other_plant_data/             # Non-Arecanut Data
│   ├── Plants_2/                    # General plant images
│   └── Refined_Data/                # Downloaded datasets (Mango, etc.)
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

---

## 🔧 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Deep Learning** | PyTorch | Model training & inference |
| **Model Architecture** | MobileNetV3-Small | Lightweight CNN with transfer learning |
| **Backend API** | FastAPI + Uvicorn | REST API for model serving |
| **Frontend** | React + Vite | Enterprise dashboard UI |
| **Styling** | Vanilla CSS | Premium light theme |
| **Data Processing** | Pandas, Scikit-learn | Dataset preparation & evaluation |
| **Image Processing** | Pillow, TorchVision | Image transforms & augmentation |
| **GPU Acceleration** | CUDA (NVIDIA GPU) | Fast inference |

---

## 🧠 How It Works

### Training Pipeline

```
Raw Images → Data Preparation → CSV Dataset → PyTorch DataLoader
    → MobileNetV3-Small (Transfer Learning)
        → Phase 1: Train classifier head (4 epochs)
        → Phase 2: Fine-tune all layers (2 epochs)
    → Save best model → Evaluate on test set
```

1. **Data Preparation** (`01_data_preparation.py`): Organizes raw Arecanut images into train/test splits with disease labels
2. **Dataset Creation** (`prepare_combined_dataset.py`): Combines Arecanut + Non-Arecanut images into a unified CSV with 7 class labels
3. **Custom Dataset** (`torch_dataset.py`): PyTorch Dataset class that loads images, applies transforms (resize, normalize)
4. **Model Training** (`train_arecanut_model.py`): 
   - Uses pre-trained MobileNetV3-Small from ImageNet
   - **Phase 1**: Freeze backbone, train only the classification head (4 epochs)
   - **Phase 2**: Unfreeze all layers, fine-tune with lower learning rate (2 epochs)
   - Saves best model based on validation accuracy
5. **Evaluation** (`evaluate_model.py`): Tests on held-out test set → **99.84% accuracy**

### Inference Pipeline

```
User uploads image → Frontend sends to Backend API
    → Image preprocessed (resize to 224×224, normalize)
    → Forward pass through MobileNetV3-Small
    → Softmax probabilities for all 7 classes
    → Smart confidence thresholding (4 rules)
    → Return: class, confidence, severity, treatment, prevention
```

### Smart Confidence Thresholding (Backend)

The system implements 4 safety rules to prevent misclassification:

| Rule | Trigger | Action |
|------|---------|--------|
| **Rule 1** | Confidence < 85% | Show "Low Confidence" warning |
| **Rule 2** | Not_Arecanut in top-3 predictions & > 10% | Show "May not be Arecanut" warning |
| **Rule 3** | Confidence < 60% | Force override to "Not Arecanut" |
| **Rule 4** | Top-2 predictions within 20% | Show "Model uncertain" warning |

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+ with PyTorch
- Node.js (portable version in `.cache/node/`)

### Step 1: Start Backend (FastAPI)
```bash
cd d:\Krishisethu
.\venv\Scripts\python -m uvicorn backend.server:app --port 8000
```

### Step 2: Start Frontend (React)
```bash
# Set Node.js path and start Vite
cmd /c "set PATH=d:\Krishisethu\.cache\node;%PATH% && cd frontend && npx vite --port 3000"
```

### Step 3: Open Dashboard
Open **http://localhost:3000** in your browser.

---

## 📸 Dashboard Features

- **Premium Light Enterprise Theme** — Clean whites, soft shadows, professional typography
- **Drag & Drop Upload** — Upload any plant image for instant diagnosis
- **Real-time Confidence Bar** — Visual confidence percentage with color coding
- **Severity Badges** — Color-coded severity (None/Medium/High/Critical)
- **Treatment Recommendations** — Specific treatment for each disease
- **Prevention Guidelines** — Best practices to prevent disease recurrence
- **Confidence Distribution** — Bar chart showing probabilities across all 7 classes
- **Scan History** — Log of recent predictions with timestamps
- **Session Analytics** — Total scans, diseases found, healthy count
- **System Status** — Live "System Online" indicator

---

## 👤 Author

**Vasista CV**  
Plant Disease Classification System — KrishiSethu AI  
© 2026

---

## 📝 Future Scope

1. **Expand Dataset**: Add more diverse non-arecanut images for better rejection
2. **Multi-Crop Support**: Extend to Mango, Coconut, and other crops
3. **Mobile App**: Deploy as Android/iOS application for field use
4. **Cloud Deployment**: Host on AWS/GCP for remote access
5. **Geo-Tagging**: Map disease outbreaks geographically
6. **Time-Series Analysis**: Track disease progression over time
