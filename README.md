# Plant Disease Classification System

## Project Overview

This project implements a unified two-stage deep learning pipeline for the identification of Arecanut plants and the subsequent classification of their health status. The system is designed to be robust, utilizing a binary classification model to filter irrelevant images (non-Arecanut) followed by a multi-class classification model to detect specific diseases in validated Arecanut leaf/nut images.

## System Architecture

The workflow consists of two sequential MobileNetV3 models:

1.  **Identification Model**: A binary classifier that acts as a gatekeeper. It verifies whether the input image contains an Arecanut plant.
2.  **Disease Classification Model**: If the identification model confirms the presence of an Arecanut plant, this second model classifies the detailed health condition (e.g., Healthy, Mahali Koleroga, Yellow Leaf Disease).

This split-architecture ensures high reliability by preventing the disease classifier from making erroneous predictions on irrelevant data.

## Key Features

*   **Two-Stage Pipeline**: Minimizes false positives by strictly verifying plant identity first.
*   **Deep Learning Backbone**: Utilizes MobileNetV3-Small for an optimal balance between accuracy and inference speed, suitable for edge deployment.
*   **Transfer Learning**: leverages ImageNet pre-trained weights for faster convergence.
*   **Robust Training**: Includes data augmentation (random rotations, flips, color jitter) to handle real-world variability.
*   **GPU Acceleration**: Fully optimized for NVIDIA GPUs using PyTorch with CUDA support.

## Dataset Structure

The system expects the following data organization:

*   **Non-Arecanut Data**: diverse images of other plants (Mango, Lemon, etc.) to train the negative class.
*   **Arecanut Data**:
    *   **Healthy**: Leaves, nuts, trunks.
    *   **Diseases**:
        *   Mahali Koleroga
        *   Yellow Leaf Disease
        *   Stem Bleeding
        *   Stem Cracking
        *   Bud Borer

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/vasistacv/Plant_Disease_Classification_system.git
    cd Plant_Disease_Classification_system
    ```

2.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: For GPU support, ensure you have the correct CUDA-enabled PyTorch version installed.*

## Usage

### 1. Data Preparation
Run the data preparation script to index images and generate CSV manifests:
```bash
python src/01_data_preparation.py
```

### 2. Training
Train the Identification and Disease models:
```bash
# Train Identification Model
python src/02_train_id_torch.py

# Train Disease Classification Model
python src/03_train_disease_torch.py
```

### 3. Inference
Run the inference pipeline on a single image:
```bash
python src/04_inference_pipeline_torch.py path/to/image.jpg
```

### 4. Interactive Demo
Launch the Streamlit web interface:
```bash
streamlit run src/app_torch.py
```

## Technical Details

*   **Framework**: PyTorch
*   **Base Model**: MobileNetV3-Small
*   **Optimizer**: Adam
*   **Loss Functions**:
    *   Identification: BCEWithLogitsLoss
    *   Disease Classification: CrossEntropyLoss
*   **Image Resolution**: 224x224 pixels

## License

This project is licensed under the MIT License.
