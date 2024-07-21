from flask import Blueprint, current_app
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import logging
import time
import threading
import queue

stream = Blueprint("stream", __name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 16
QUEUE_SIZE = 1000
PREDICTION_THRESHOLD = 0.5

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

# Load models and label mappings
try:
    face_model = load_model(
        os.path.join(model_dir, "saved_model_MobileNetV3Small_16_face")
    )
    body_model = load_model(
        os.path.join(model_dir, "saved_model_MobileNetV3Small_16_body")
    )

    face_label_to_index = np.load(
        os.path.join(model_dir, "face_label_mapping.npy"), allow_pickle=True
    ).item()
    body_label_to_index = np.load(
        os.path.join(model_dir, "body_label_mapping.npy"), allow_pickle=True
    ).item()

    face_index_to_label = {v: k for k, v in face_label_to_index.items()}
    body_index_to_label = {v: k for k, v in body_label_to_index.items()}

    logger.info("Models and label mappings loaded successfully")
except Exception as e:
    logger.error(f"Error loading models or label mappings: {str(e)}")
    raise

# Queues for face and body frames
face_queue = queue.Queue(maxsize=QUEUE_SIZE)
body_queue = queue.Queue(maxsize=QUEUE_SIZE)


def preprocess_image(frame, target_size=(224, 224)):
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)
    normalized_rgb_img = resized_rgb_img.astype(np.float32) / 255.0
    return normalized_rgb_img


def predict_batch(batch, model, index_to_label):
    predictions = model.predict(np.array(batch))
    predicted_labels = [
        [
            index_to_label[i]
            for i, prob in enumerate(pred)
            if prob > PREDICTION_THRESHOLD
        ]
        for pred in predictions
    ]
    return predicted_labels


def process_stream(stream_url, queue):
    cap = cv2.VideoCapture(stream_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Failed to read frame from {stream_url}")
            break

        processed_frame = preprocess_image(frame)

        if not queue.full():
            queue.put(processed_frame)

    cap.release()


def process_model_queue(queue, model, index_to_label, model_name):
    batch = []
    while True:
        try:
            frame = queue.get(timeout=1)
            batch.append(frame)

            if len(batch) == BATCH_SIZE:
                start_time = time.time()
                predictions = predict_batch(batch, model, index_to_label)
                end_time = time.time()

                for pred in predictions:
                    logger.info(
                        f"{model_name} prediction: {pred}, Time: {end_time - start_time:.4f} seconds"
                    )

                batch = []
        except queue.Empty:
            if batch:
                start_time = time.time()
                predictions = predict_batch(batch, model, index_to_label)
                end_time = time.time()

                for pred in predictions:
                    logger.info(
                        f"{model_name} prediction: {pred}, Time: {end_time - start_time:.4f} seconds"
                    )

                batch = []


def start_streams():
    face_stream_url = (
        "http://192.168.2.175:81/stream"  # Replace with actual IP for face stream
    )
    body_stream_url = (
        "http://192.168.2.176:81/stream"  # Replace with actual IP for body stream
    )

    # Start stream processing threads
    threading.Thread(
        target=process_stream, args=(face_stream_url, face_queue), daemon=True
    ).start()
    threading.Thread(
        target=process_stream, args=(body_stream_url, body_queue), daemon=True
    ).start()

    # Start model processing threads
    threading.Thread(
        target=process_model_queue,
        args=(face_queue, face_model, face_index_to_label, "Face"),
        daemon=True,
    ).start()
    threading.Thread(
        target=process_model_queue,
        args=(body_queue, body_model, body_index_to_label, "Body"),
        daemon=True,
    ).start()

    logger.info("Streams started")


@stream.route("/stream_info", methods=["GET"])
def get_stream_info():
    return {
        "active_streams": 2,
        "face_queue_size": face_queue.qsize(),
        "body_queue_size": body_queue.qsize(),
        "batch_size": BATCH_SIZE,
        "queue_size": QUEUE_SIZE,
        "prediction_threshold": PREDICTION_THRESHOLD,
    }
