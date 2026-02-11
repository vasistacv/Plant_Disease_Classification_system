import os
import pandas as pd
import config

def is_valid_image(filename):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    return any(filename.endswith(ext) for ext in valid_extensions)

def main():
    print("Preparing Combined Arecanut Dataset...")
    
    all_data = []
    
    # 1. Process Arecanut Images (with disease labels)
    print("\n[1/2] Processing Arecanut images...")
    for split_dir in [config.ARECANUT_TRAIN_DIR, config.ARECANUT_TEST_DIR]:
        if not os.path.exists(split_dir):
            continue
            
        for class_folder in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            # Map folder name to clean label
            clean_label = config.ARECANUT_DISEASE_MAPPING.get(class_folder, class_folder)
            
            for img_file in os.listdir(class_path):
                if is_valid_image(img_file):
                    img_path = os.path.join(class_path, img_file)
                    all_data.append({
                        'filepath': img_path,
                        'label': clean_label
                    })
    
    print(f"   Found {len(all_data)} Arecanut images")
    
    # 2. Process Non-Arecanut Images (all labeled as 'Not_Arecanut')
    print("\n[2/2] Processing Non-Arecanut images...")
    non_arecanut_count = 0
    
    for split_dir in [config.NON_ARECANUT_TRAIN_DIR, config.NON_ARECANUT_TEST_DIR]:
        if not os.path.exists(split_dir):
            continue
            
        for class_folder in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            for img_file in os.listdir(class_path):
                if is_valid_image(img_file):
                    img_path = os.path.join(class_path, img_file)
                    all_data.append({
                        'filepath': img_path,
                        'label': 'Not_Arecanut'
                    })
                    non_arecanut_count += 1
    
    print(f"   Found {non_arecanut_count} Non-Arecanut images")
    
    # 3. Create DataFrame and Save
    df = pd.DataFrame(all_data)
    
    print("\n" + "="*50)
    print("Class Distribution:")
    print("="*50)
    print(df['label'].value_counts())
    
    output_path = os.path.join(config.DATA_DIR, 'arecanut_dataset.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Saved combined dataset to: {output_path}")
    print(f"[OK] Total images: {len(df)}")

if __name__ == "__main__":
    main()
