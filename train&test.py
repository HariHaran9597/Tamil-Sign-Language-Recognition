import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess_images
from model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Define directory containing the dataset
dataset_directory = "data"

# Preprocess dataset
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_images(dataset_directory)

# Convert labels to numerical format
label_mapping = {label: index for index, label in enumerate(np.unique(y_train))}
y_train = np.array([label_mapping[label] for label in y_train])
y_val = np.array([label_mapping[label] for label in y_val])
y_test = np.array([label_mapping[label] for label in y_test])

# Create the model
model = create_model()

# Define callbacks
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping, reduce_lr])

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
