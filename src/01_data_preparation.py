import os
import pandas as pd
import cv2
import sys
import config

# Force unbuffered output
sys.stdout.reconfigure(encoding='utf-8')

def is_valid_image(filepath):
    """Check if an image file is valid and readable."""
    try:
        if not filepath.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            return False
        
        # Verify it can be opened
        img = cv2.imread(filepath)
        if img is None:
            return False
        return True
    except Exception:
        return False

def process_directory(directory, label_class, is_arecanut=True):
    """Process a directory and return a list of file info dictionaries."""
    data = []
    print(f"Walking directory: {directory}", flush=True)
    
    count = 0
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        folder_name = os.path.basename(root)
        
        if root == directory:
            continue
            
        print(f"Processing folder: {folder_name}", flush=True)
        
        valid_files_in_folder = 0
        for file in files:
            filepath = os.path.join(root, file)
            if is_valid_image(filepath):
                item = {
                    'filepath': filepath,
                    'filename': file,
                    'folder': folder_name,
                    'is_arecanut': 1 if is_arecanut else 0,
                    'original_split': 'train' if 'train' in directory else 'test'
                }
                
                if is_arecanut:
                    disease_label = config.ARECANUT_DISEASE_MAPPING.get(folder_name)
                    if disease_label:
                        item['disease_label'] = disease_label
                        item['specific_disease'] = folder_name
                    else:
                        item['disease_label'] = 'Unknown'
                        item['specific_disease'] = folder_name
                else:
                    item['disease_label'] = 'Non_Arecanut'
                    item['specific_disease'] = 'Non_Arecanut'
                
                data.append(item)
                valid_files_in_folder += 1
                count += 1
                
                if count % 100 == 0:
                    print(f"Processed {count} images so far...", flush=True)
        
        print(f" - Found {valid_files_in_folder} valid images in {folder_name}", flush=True)
                
    return data

def prepare_datasets():
    print("Starting Data Preparation...", flush=True)
    
    all_data = []
    
    print("\n--- Arecanut Train Data ---", flush=True)
    all_data.extend(process_directory(config.ARECANUT_TRAIN_DIR, label_class=1, is_arecanut=True))
    
    print("\n--- Arecanut Test Data ---", flush=True)
    all_data.extend(process_directory(config.ARECANUT_TEST_DIR, label_class=1, is_arecanut=True))
    
    print("\n--- Non-Arecanut Train Data ---", flush=True)
    all_data.extend(process_directory(config.NON_ARECANUT_TRAIN_DIR, label_class=0, is_arecanut=False))
    
    # Process Non-Arecanut Test
    print("\n--- Non-Arecanut Test Data ---", flush=True)
    # Check if test dir exists for non-arecanut (it might not be structured same way or empty)
    if os.path.exists(config.NON_ARECANUT_TEST_DIR):
        all_data.extend(process_directory(config.NON_ARECANUT_TEST_DIR, label_class=0, is_arecanut=False))
    else:
        print(f"Non-Arecanut Test Directory not found: {config.NON_ARECANUT_TEST_DIR}", flush=True)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save Full Dataset
    save_path = os.path.join(config.DATA_DIR, 'full_dataset.csv')
    df.to_csv(save_path, index=False)
    print(f"\nSaved Full Dataset to {save_path} with {len(df)} images.", flush=True)
    
    print("\nIdentification Model Distribution:", flush=True)
    print(df['is_arecanut'].value_counts(), flush=True)
    
    df_disease = df[df['is_arecanut'] == 1].copy()
    save_disease_path = os.path.join(config.DATA_DIR, 'disease_dataset.csv')
    df_disease.to_csv(save_disease_path, index=False)
    print(f"\nSaved Disease Dataset to {save_disease_path}", flush=True)
    print("Disease Dataset Distribution:", flush=True)
    print(df_disease['disease_label'].value_counts(), flush=True)
    
    return df

if __name__ == "__main__":
    prepare_datasets()
