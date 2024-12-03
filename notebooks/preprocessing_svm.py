import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images(data_dir, image_size=(128, 128)): 
    # Load images from the specified directory, preprocess them, and return arrays of images and labels.
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    total_images = 0  # Counter for total images
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_count = 0  # Counter for images in the current class
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize(image_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
                    images.append(img_array.flatten())  # Flatten the image
                    labels.append(label_map[class_name])
                    class_count += 1
                    total_images += 1
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        print(f"Loaded {class_count} images for class '{class_name}'")
    
    print(f"Total images loaded: {total_images}")
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y

def split_data(X, y, train_size=0.7, test_size=0.2, val_size=0.1, random_state=42):
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Adjust validation size relative to the remaining data
    val_relative_size = val_size / (train_size + val_size)
    
    # Split the remaining data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state, stratify=y_temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_svm(data_dir, image_size=(128, 128)): 
    X, y = load_images(data_dir, image_size)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    return X_train, X_val, X_test, y_train, y_val, y_test


