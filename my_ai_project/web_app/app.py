import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, static_folder='static')
CORS(app)

# تحميل النموذج والمُحجِّم
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..') if os.path.basename(BASE_DIR) == 'web_app' else BASE_DIR

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_transfer_learning_vgg16_model.keras')
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'train_labels.csv')

model = tf.keras.models.load_model(MODEL_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)
age_scaler = MinMaxScaler(feature_range=(0, 1))
age_scaler.fit(train_df[['age']])

IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128

# تقديم الواجهة الأمامية
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# دالة التوقع (كما كانت في تصميمك الأصلي)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    img = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = np.expand_dims(img / 255.0, axis=0)
    
    predictions = model.predict(img)
    age = age_scaler.inverse_transform(np.array([[predictions[0][0][0]]]))[0][0]
    gender = "Female" if predictions[1][0][0] > 0.5 else "Male"
    confidence = float(predictions[1][0][0] if gender == "Female" else (1 - predictions[1][0][0]))
    
    return jsonify({'age': int(age), 'gender': gender, 'gender_confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)