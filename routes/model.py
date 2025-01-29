from flask import Blueprint, request, jsonify, make_response
from tensorflow.keras.models import load_model
import numpy as np
import uuid
import cv2
import os
import logging
from collections import deque
import time
import imghdr
import threading
from firestore import drive_sessions

model = Blueprint("model", __name__)
PREDICTION_THRESHOLD = 0.5
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_TYPES = {"jpg", "jpeg", "png"}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

frame_buffer = deque(maxlen=60)
buffer_lock = threading.Lock()

is_generating = False
generator_thread = None

# Distraction types and their probabilities
DISTRACTIONS = [
    "texting",
    "phone call",
    "safe",
    "radio",
    "reach side",
    "talking to passenger",
    "drowsy",
    "hair and makeup",
]
PROBABILITIES = [0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1]

try:
    saved_model = load_model(os.path.join(model_dir, "saved_model_MobileNetV3Small_16"))
    label_to_index = np.load(
        os.path.join(model_dir, "label_mapping.npy"), allow_pickle=True
    ).item()
    index_to_label = {v: k for k, v in label_to_index.items()}
    logger.info("Model and label mappings loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or label mappings: {str(e)}")
    raise


def validate_image(image_bytes):
    # Check file size
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError(
            f"Image size exceeds the maximum allowed size of {MAX_IMAGE_SIZE / 1024 / 1024} MB"
        )

    # Check file type
    file_type = imghdr.what(None, image_bytes)
    if file_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(
            f"Unsupported image type: {file_type}. Allowed types are {ALLOWED_IMAGE_TYPES}"
        )


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


def save_frames_to_firestore(sessionId):
    """
    Save accumulated frames to Firestore when buffer is full
    """
    global session_start

    with buffer_lock:
        if len(frame_buffer) >= 60:
            frame_data = list(frame_buffer)

            # Create a session document
            session_data = {
                "timestamp_start": session_start,
                "timestamp_end": int(time.time()),
                "frame_count": len(frame_data),
                "frames": frame_data,
                "session_id": str(sessionId),
            }

            # Add to Firestore
            drive_sessions.add(session_data)

            # Clear the buffer
            frame_buffer.clear()
            session_start = None


def normalize_probabilities():
    """Normalize probabilities to sum to 1"""
    return np.array(PROBABILITIES) / sum(PROBABILITIES)


def generate_distractions(sessionId):
    """Generate distraction events every second"""
    global is_generating, session_start, generator_thread

    while is_generating:
        # Generate random distraction based on probabilities
        distraction = np.random.choice(DISTRACTIONS, p=normalize_probabilities())
        print(f"Current distraction: {distraction}")

        with buffer_lock:
            # If this is the first frame in a new session
            if len(frame_buffer) == 0:
                session_start = int(time.time())

            frame_buffer.append(distraction)

        # Check if we should save to Firestore
        if len(frame_buffer) >= 60:
            save_frames_to_firestore(sessionId)

        # Wait for 1 second
        time.sleep(1)


@model.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        logger.warning("No image file provided in the request")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    try:
        total_start_time = time.time()

        # Read the image file
        img_bytes = file.read()
        logger.info(f"Read image with size {len(img_bytes)} bytes")

        # Validate image
        try:
            validate_image(img_bytes)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        logger.info(f"Validated image successfully")

        # Measure image processing time
        preprocess_start_time = time.time()
        img_array = preprocess_image(img_bytes)
        preprocess_end_time = time.time()
        image_processing_time = preprocess_end_time - preprocess_start_time
        logger.info(f"Processed image successfully")

        # Make prediction (add batch dimension for single image)
        prediction_start_time = time.time()
        predictions = saved_model.predict(np.expand_dims(img_array, axis=0))
        prediction_end_time = time.time()
        prediction_time = prediction_end_time - prediction_start_time

        # Convert predictions to human-readable labels
        predicted_labels = [
            index_to_label[i]
            for i, prob in enumerate(predictions[0])
            if prob > PREDICTION_THRESHOLD
        ]

        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time

        logger.info(
            f"Prediction made successfully. Labels: {predicted_labels}, "
            f"Image Processing Time: {image_processing_time:.4f} seconds, "
            f"Prediction Time: {prediction_time:.4f} seconds, "
            f"Total Time: {total_processing_time:.4f} seconds"
        )

        with buffer_lock:
            # If this is the first frame in a new session
            if len(frame_buffer) == 0:
                session_start = int(time.time())

            frame_buffer.append(predicted_labels)

        # Check if we should save to Firestore
        if len(frame_buffer) >= 60:
            save_frames_to_firestore()

        return jsonify(
            {
                "predictions": predicted_labels,
                "image_processing_time": image_processing_time,
                "prediction_time": prediction_time,
                "total_processing_time": total_processing_time,
            }
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred during prediction"}), 500


@model.route("/start-generation", methods=["GET"])
def start_generation():
    global is_generating, generator_thread

    sessionId = uuid.uuid4()
    if not is_generating:
        is_generating = True
        generator_thread = threading.Thread(
            target=generate_distractions, args=(sessionId,)
        )
        generator_thread.start()
        return jsonify({"status": "Generation started"}), 200
    else:
        return jsonify({"status": "Generation already running"}), 400


@model.route("/stop-generation", methods=["GET"])
def stop_generation():
    global is_generating, generator_thread

    if is_generating:
        is_generating = False
        if generator_thread:
            generator_thread.join()
            generator_thread = None
        return jsonify({"status": "Generation stopped"}), 200
    else:
        return jsonify({"status": "Generation not running"}), 400


@model.route("/status", methods=["GET"])
def get_status():
    return (
        jsonify(
            {
                "is_generating": is_generating,
                "distractions": DISTRACTIONS,
                "probabilities": PROBABILITIES,
            }
        ),
        200,
    )


@model.route("/modelInfo", methods=["GET"])
def get_model_info():
    return jsonify(
        {
            "message": "Model is ready for predictions",
            "model_type": "MobileNetV3Small",
            "input_shape": saved_model.input_shape[1:],
            "num_classes": len(label_to_index),
            "label_mapping": label_to_index,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "max_image_size": MAX_IMAGE_SIZE,
            "allowed_image_types": list(ALLOWED_IMAGE_TYPES),
        }
    )
