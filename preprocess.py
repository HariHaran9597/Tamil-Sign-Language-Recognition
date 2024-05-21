import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# Function to preprocess images
def preprocess_images(directory, english_tamil_mapping=None, img_size=(64, 64), test_size=0.2, val_split=0.2):
    images = []
    labels = []

    if english_tamil_mapping is None:
        english_tamil_mapping = {
            'love': 'அன்பு',
            'grace': 'அருள்',
            'truth': 'உண்மை',
            'blessings': 'நல்வாழ்த்துக்கள்',
            'happiness': 'மகிழ்ச்சி',
            'heart': 'மனம்',
            'hello': 'வணக்கம்',
            'life': 'வாழ்க்கை',
            'victory': 'வெற்றி'
        }

    for english_folder in os.listdir(directory):
        folder_path = os.path.join(directory, english_folder)

        # Check if the item in the directory is a folder
        if os.path.isdir(folder_path):
            # Get the corresponding Tamil word from the mapping
            tamil_word = english_tamil_mapping.get(english_folder, None)

            # Skip if there's no corresponding Tamil word
            if tamil_word is None:
                print(f"No corresponding Tamil word found for English folder: {english_folder}")
                continue

            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)

                # Attempt to read the image
                img = cv2.imread(img_path)

                # Check if the image is valid
                if img is None:
                    print(f"Skipping invalid image: {img_path}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                img = cv2.resize(img, img_size)  # Resize image
                img = img / 255.0  # Normalize pixel values

                label = tamil_word  # Use Tamil word as label

                images.append(img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Split the dataset into training, validation, and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_split, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Define directory containing the dataset
dataset_directory = "data"

# Preprocess dataset
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_images(dataset_directory)

# Print the sizes of the split datasets
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Testing set size:", len(X_test))
