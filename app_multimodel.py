"""
============================================================
FLASK WEBAPP - RICE PLANT DISEASE PREDICTION (Multi-Model Version)
============================================================
Loads whichever model the comparison notebook selected as the best
performer (MobileNetV2 / ResNet50 / VGG16 / EfficientNetB0 /
InceptionV3 / DenseNet121), using model_meta.json to know the
correct input image size and preprocessing function for that
specific architecture. This matters because different architectures
expect different pixel preprocessing, not just a generic /255 scale.

Run before starting:
    pip install -r requirements.txt

Run command:
    python app.py

Then open: http://127.0.0.1:5000
============================================================
"""

import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Preprocessing functions for each architecture this project supports.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

app = Flask(__name__)

# ---------- Paths ----------
MODEL_PATH = os.path.join('model', 'rice_disease_model.h5')
LABELS_PATH = os.path.join('model', 'labels.json')
META_PATH = os.path.join('model', 'model_meta.json')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Load model + labels + metadata (once, at server start) ----------
print("Loading model...")
model = load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    labels_map = json.load(f)   # {"0": "Bacterial Leaf Blight", "1": "Brown Spot", ...}

with open(META_PATH, 'r') as f:
    model_meta = json.load(f)   # {"model_name": "...", "img_height": ..., "img_width": ...}

IMG_SIZE = (model_meta['img_height'], model_meta['img_width'])
MODEL_NAME = model_meta['model_name']
print(f"Loaded model: {MODEL_NAME}  |  input size: {IMG_SIZE}")

# Map each architecture name to its correct preprocess_input function.
PREPROCESS_MAP = {
    "MobileNetV2": mobilenet_v2_preprocess,
    "ResNet50": resnet50_preprocess,
    "VGG16": vgg16_preprocess,
    "EfficientNetB0": efficientnet_preprocess,
    "InceptionV3": inception_v3_preprocess,
    "DenseNet121": densenet_preprocess,
}
preprocess_fn = PREPROCESS_MAP.get(MODEL_NAME)
if preprocess_fn is None:
    raise ValueError(f"Unknown model_name '{MODEL_NAME}' in model_meta.json — "
                      f"add its preprocess_input to PREPROCESS_MAP above.")

# ---------------------------------------------------------------
# Advisory text per disease. Keys must EXACTLY match labels.json's
# values (same spelling/spacing/case as your dataset's folder names).
# ---------------------------------------------------------------
DISEASE_INFO = {
    "Healthy Rice Leaf": "The plant looks healthy — no disease detected in this image.",
    "Bacterial Leaf Blight": "Signs of bacterial infection detected — watch for water-soaked streaks on leaves, improve field drainage, and consider a copper-based bactericide.",
    "Brown Spot": "Brown spot symptoms detected — fungicide spray and checking potassium nutrient levels is recommended.",
    "Leaf Blast": "Leaf blast detected — remove affected leaves, reduce nitrogen fertilizer, and apply fungicide.",
    "Leaf scald": "Leaf scald symptoms detected — improve field drainage and consider a resistant variety.",
    "Narrow Brown Leaf Spot": "Narrow brown leaf spot detected — balanced fertilization and fungicide treatment are recommended.",
    "Rice Hispa": "Signs of Rice Hispa (insect pest damage) detected — remove affected leaves and use a recommended insecticide.",
    "Sheath Blight": "Sheath blight symptoms detected — increase plant spacing for better air circulation and apply fungicide."
}


def prepare_image(img_path):
    """Loads an image and applies the correct preprocessing for the
    currently loaded model architecture."""
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = preprocess_fn(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file received'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img_array = prepare_image(filepath)
    predictions = model.predict(img_array)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = labels_map[str(predicted_index)]
    confidence = float(predictions[predicted_index]) * 100

    print("Full prediction probabilities:",
          {labels_map[str(i)]: round(float(p) * 100, 2) for i, p in enumerate(predictions)})

    result = {
        'disease': predicted_label,
        'confidence': round(confidence, 2),
        'info': DISEASE_INFO.get(predicted_label, ""),
        'model_used': MODEL_NAME,
        'image_url': '/' + filepath
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
