import os
import numpy as np
import cv2
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pickle

# Constants
PREDICT_DIR = r'D:\Wheat-diesease\wheat-disease-detection\testCDD'

IMG_SIZE = 64
MODEL_FILENAME = 'wheatDiseaseModel.keras'

# Helper functions
def load_trained_model(model_filename):
    try:
        model = load_model(model_filename)
        print(f"Model loaded successfully from {model_filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_images(image_dir):
    images = []
    image_names = os.listdir(image_dir)
    for name in image_names:
        img_path = os.path.join(image_dir, name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
        else:
            print(f"Skipping invalid image: {img_path}")
    return np.array(images)

def predict_on_images(model, image_dir):
    images = preprocess_images(image_dir)
    images = images.astype('float32') / 255.0  # Normalize images
    
    # Load the LabelBinarizer (ensure you use the same one used during training)
    # You can either reload the label binarizer here or save/load it like the model.
    with open("label_binarizer.pkl", "rb") as f:
        label_binarizer = pickle.load(f)


    predictions = model.predict(images)
    for i, prediction in enumerate(predictions):
        predicted_label = label_binarizer.inverse_transform(np.array([prediction]))[0]


        print(f"Image: {os.listdir(image_dir)[i]} → Predicted Label: {predicted_label}")

# Main logic
if __name__ == "__main__":
    # Load the trained model (don't retrain, just load)
    model = load_trained_model(MODEL_FILENAME)
    if model is None:
        print("Error loading model, exiting.")
        exit()

    print("🧪 Predicting on new images...")
    predict_on_images(model, PREDICT_DIR)
