import os

# Base Paths
BASE_DIR = r"d:\Krishisethu"
# Using raw strings to avoid escape character issues
ARECANUT_TRAIN_DIR = os.path.join(BASE_DIR, r"Arecanut_data\Arecanut_dataset\Arecanut_dataset\train")
ARECANUT_TEST_DIR = os.path.join(BASE_DIR, r"Arecanut_data\Arecanut_dataset\Arecanut_dataset\test")

NON_ARECANUT_TRAIN_DIR = os.path.join(BASE_DIR, r"other_plant_data\Plants_2\train")
NON_ARECANUT_TEST_DIR = os.path.join(BASE_DIR, r"other_plant_data\Plants_2\test")

# Output Paths
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Model Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42

# Explicit Class Mappings for Analysis
ARECANUT_DISEASE_MAPPING = {
    'Healthy_Leaf': 'Healthy',
    'Healthy_Nut': 'Healthy',
    'Healthy_Trunk': 'Healthy',
    'healthy_foot': 'Healthy',
    'Mahali_Koleroga': 'Mahali_Koleroga',
    'Stem_bleeding': 'Stem_bleeding',
    'bud borer': 'Bud_Borer',
    'stem cracking': 'Stem_Cracking',
    'yellow leaf disease': 'Yellow_Leaf_Disease'
}

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
