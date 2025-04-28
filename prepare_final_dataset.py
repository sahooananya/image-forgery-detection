import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = 'dataset'
final_dataset_dir = 'final_dataset'
train_dir = os.path.join(final_dataset_dir, 'train')
test_dir = os.path.join(final_dataset_dir, 'test')

# Create necessary folders
def create_folders():
    for folder in [train_dir, test_dir]:
        for label in ['authentic', 'tampered']:
            os.makedirs(os.path.join(folder, label), exist_ok=True)

# Merge datasets
def merge_datasets():
    sources = [
        (os.path.join(dataset_dir, 'CASIA2', 'Au'), 'authentic'),
        (os.path.join(dataset_dir, 'CASIA2', 'Tp'), 'tampered'),
        (os.path.join(dataset_dir, 'COVERAGE', 'image'), 'authentic'),
        (os.path.join(dataset_dir, 'COVERAGE', 'tampered'), 'tampered'),
        (os.path.join(dataset_dir, 'Columbia', 'authentic'), 'authentic'),
        (os.path.join(dataset_dir, 'Columbia', 'spliced'), 'tampered')
    ]

    # Handling Columbia separately (complicated folder structure)
    columbia_dir = os.path.join(dataset_dir, 'Columbia')
    for folder in os.listdir(columbia_dir):
        folder_path = os.path.join(columbia_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder.startswith('Sp'):  # Spliced = tampered
            sources.append((folder_path, 'tampered'))
        else:  # Authentic
            sources.append((folder_path, 'authentic'))

    all_data = []
    all_labels = []

    for src_folder, label in sources:
        for img_name in os.listdir(src_folder):
            img_path = os.path.join(src_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            all_data.append(img)
            all_labels.append(0 if label == 'authentic' else 1)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels

# Save train/test images
def save_split(X_train, X_test, y_train, y_test):
    def save_images(X, y, base_dir):
        for idx in range(len(X)):
            label_dir = 'authentic' if y[idx] == 0 else 'tampered'
            save_path = os.path.join(base_dir, label_dir, f'{idx}.png')
            cv2.imwrite(save_path, (X[idx] * 255).astype('uint8'))

    save_images(X_train, y_train, train_dir)
    save_images(X_test, y_test, test_dir)

# Full process
def full_pipeline():
    create_folders()
    print("✅ Created folders!")
    X, y = merge_datasets()
    print(f"✅ Merged dataset: {X.shape[0]} images total")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]} images")
    save_split(X_train, X_test, y_train, y_test)
    print("✅ Train/test images saved into folders!")

    # Save npz compressed file
    np.savez_compressed('forgery_dataset.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("✅ Dataset also saved as forgery_dataset.npz!")

if __name__ == '__main__':
    full_pipeline()
