import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from PIL import Image
import pickle

# Constants
DATASET_PATH = r'D:\Wheat-diesease\wheat-disease-detection\cropDiseaseDataset'
PREDICT_DIR = r'D:\Wheat-diesease\wheat-disease-detection\testCDD'
IMG_SIZE = 64
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 30
MODEL_FILENAME = 'wheatDiseaseModel.keras'
LB_FILENAME = 'label_binarizer.pkl'

# Helper functions
def convert_gif_to_png(dataset_dir):
    for subdir, _ , files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".gif"):
                gif_path = os.path.join(subdir, file)
                try:
                    img = Image.open(gif_path).convert("RGB")
                    new_path = gif_path.replace(".gif", ".png")
                    img.save(new_path)
                    os.remove(gif_path)
                    print(f"Converted: {gif_path} -> {new_path}")
                except Exception as e:
                    print(f"Failed to convert {gif_path}: {e}")

def load_dataset(dataset_path):
    images = []
    labels = []
    label_binarizer = LabelBinarizer()
    classes = os.listdir(dataset_path)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                        images.append(image)
                        labels.append(class_name)
                    else:
                        print(f"Skipping invalid image: {image_path}")
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

    labels = label_binarizer.fit_transform(labels)

    # Save label binarizer
    with open(LB_FILENAME, "wb") as f:
        pickle.dump(label_binarizer, f)

    return np.array(images), np.array(labels), label_binarizer

def preprocess_images(images):
    return images.astype('float32') / 255.0

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='orange')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# def predict_on_images(model, image_dir, label_binarizer):
#     if not os.path.exists(image_dir):
#         print(f"Error: The directory {image_dir} does not exist.")
#         return

#     image_names = os.listdir(image_dir)
#     processed_images = []

#     for name in image_names:
#         img_path = os.path.join(image_dir, name)
#         img = cv2.imread(img_path)
#         if img is not None:
#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#             processed_images.append(img)
#         else:
#             print(f"Skipping invalid image: {img_path}")

#     processed_images = preprocess_images(np.array(processed_images))
#     predictions = model.predict(processed_images)

#     for i, prediction in enumerate(predictions):
#         predicted_label = label_binarizer.inverse_transform([prediction])[0]
#         print(f"Image: {image_names[i]} → Predicted Label: {predicted_label}")

# Main logic
if __name__ == "__main__":
    # Convert GIFs if needed
    convert_gif_to_png(DATASET_PATH)

    print("🔍 Loading dataset...")
    images, labels, label_binarizer = load_dataset(DATASET_PATH)
    images = preprocess_images(images)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    if os.path.exists(MODEL_FILENAME):
        print(f"📥 Loading existing model from '{MODEL_FILENAME}'...")
        model = load_model(MODEL_FILENAME)
    else:
        print("📦 Building and training a new model...")
        model = build_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)

        data_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        data_gen.fit(train_images)

        history = model.fit(
            data_gen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
            steps_per_epoch=len(train_images) // BATCH_SIZE,
            epochs=EPOCHS
        )
        plot_history(history)
        print("💾 Saving model...")
        model.save(MODEL_FILENAME)

    print("✅ Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # print("🧪 Predicting on new images...")
    # predict_on_images(model, PREDICT_DIR, label_binarizer)
