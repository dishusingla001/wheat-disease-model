import os
import numpy as np
import cv2
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from fertilizer_helper import load_fertilizer_data, get_fertilizer_info

# Constants
PREDICT_DIR = r'D:\Wheat-diesease\wheat-disease-model\wheat-disease-detection\testCDD'
IMG_SIZE = 64
MODEL_FILENAME = 'D:\Wheat-diesease\wheat-disease-model\wheatDiseaseModel.keras'
LABEL_BINARIZER_FILE = 'D:\Wheat-diesease\wheat-disease-model\label_binarizer.pkl'

# Helper functions
def load_trained_model(model_filename):
    try:
        model = load_model(model_filename)
        print(f"✅ Model loaded successfully from {model_filename}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def preprocess_images(image_dir):
    images = []
    image_names = []
    for name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            image_names.append(name)
        else:
            print(f"⚠️ Skipping invalid image: {img_path}")
    return np.array(images), image_names

def predict_on_images(model, image_dir, label_binarizer, fertilizer_data):
    images, image_names = preprocess_images(image_dir)
    images = images.astype('float32') / 255.0  # Normalize images

    predictions = model.predict(images)
    for i, prediction in enumerate(predictions):
        predicted_label = label_binarizer.inverse_transform(np.array([prediction]))[0]
        print(f"\n🖼️ Image: {image_names[i]}")
        print(f"🔍 Predicted Disease: {predicted_label}")

        # Get fertilizer and dosage info
        fertilizer_info = get_fertilizer_info(predicted_label, fertilizer_data)
        if fertilizer_info:
            print(f"🌱 Fertilizer Recommendation: {fertilizer_info['fertilizer']}")
            print(f"💊 Dosage: {fertilizer_info['dosage']}")
        else:
            print("⚠️ No fertilizer info found for this disease.")

# Main logic
if __name__ == "__main__":
    print("🔎 Starting prediction and fertilizer recommendation...\n")

    # Load model
    model = load_trained_model(MODEL_FILENAME)
    if model is None:
        print("Exiting due to model loading error.")
        exit()

    # Load label binarizer
    try:
        with open(LABEL_BINARIZER_FILE, "rb") as f:
            label_binarizer = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading label binarizer: {e}")
        exit()

    # Load fertilizer data
    fertilizer_data = load_fertilizer_data()

    # Predict
    predict_on_images(model, PREDICT_DIR, label_binarizer, fertilizer_data)
