from flask import Blueprint, request, jsonify, current_app
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
import io
import os
import logging
import time
import imghdr

model = Blueprint("model", __name__)
PREDICTION_THRESHOLD = 0.5
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_TYPES = {'jpg', 'jpeg', 'png'}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models')

try:
    saved_model = load_model(os.path.join(model_dir, 'saved_model_MobileNetV2_32'))
    label_to_index = np.load(os.path.join(model_dir, 'label_mapping.npy'), allow_pickle=True).item()
    index_to_label = {v: k for k, v in label_to_index.items()}
    logger.info("Model and label mappings loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or label mappings: {str(e)}")
    raise


def validate_image(image_bytes):
    # Check file size
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError(f"Image size exceeds the maximum allowed size of {MAX_IMAGE_SIZE / 1024 / 1024} MB")

    # Check file type
    file_type = imghdr.what(None, image_bytes)
    if file_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(f"Unsupported image type: {file_type}. Allowed types are {ALLOWED_IMAGE_TYPES}")


def preprocess_image(image_bytes, target_size=(224, 224)):
    try:
        # Validate image before processing
        validate_image(image_bytes)

        # Read image with OpenCV (in BGR format)
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise ValueError("Failed to decode image")

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # Resize image
        resized_rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)

        # Normalize to [0,1] which is the format expected by the model
        normalized_rgb_img = resized_rgb_img.astype(np.float32) / 255.0

        return normalized_rgb_img
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise


@model.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        logger.warning("No image file provided in the request")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    try:
        total_start_time = time.time()

        # Read the image file
        img_bytes = file.read()

        # Validate image
        try:
            validate_image(img_bytes)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Measure image processing time
        preprocess_start_time = time.time()
        img_array = preprocess_image(img_bytes)
        preprocess_end_time = time.time()
        image_processing_time = preprocess_end_time - preprocess_start_time

        # Make prediction (add batch dimension for single image)
        prediction_start_time = time.time()
        predictions = saved_model.predict(np.expand_dims(img_array, axis=0))
        prediction_end_time = time.time()
        prediction_time = prediction_end_time - prediction_start_time

        # Convert predictions to human-readable labels
        predicted_labels = [index_to_label[i] for i, prob in enumerate(predictions[0]) if prob > PREDICTION_THRESHOLD]

        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time

        logger.info(f"Prediction made successfully. Labels: {predicted_labels}, "
                    f"Image Processing Time: {image_processing_time:.4f} seconds, "
                    f"Prediction Time: {prediction_time:.4f} seconds, "
                    f"Total Time: {total_processing_time:.4f} seconds")

        return jsonify({
            "predictions": predicted_labels,
            "image_processing_time": image_processing_time,
            "prediction_time": prediction_time,
            "total_processing_time": total_processing_time
        })
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred during prediction"}), 500


@model.route("/modelInfo", methods=["GET"])
def get_model_info():
    return jsonify({
        "message": "Model is ready for predictions",
        "model_type": "MobileNetV2",
        "input_shape": saved_model.input_shape[1:],
        "num_classes": len(label_to_index),
        "label_mapping": label_to_index,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "max_image_size": MAX_IMAGE_SIZE,
        "allowed_image_types": list(ALLOWED_IMAGE_TYPES)
    })