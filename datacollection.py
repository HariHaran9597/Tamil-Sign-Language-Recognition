import os
import cv2
import numpy as np

# Dictionary mapping English folder names to Tamil words
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


def preprocess_images(directory, img_size=(64, 64)):
    images = []
    labels = []

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

    return np.array(images), np.array(labels)


# Define directory containing the dataset
dataset_directory = "data"

# Preprocess dataset and retrieve images and labels
images, labels = preprocess_images(dataset_directory)
