from flask import Blueprint, render_template, jsonify, Response
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import json
import logging
import time
import threading
from queue import Queue, Empty

stream_viewer = Blueprint("stream_viewer", __name__)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 16
QUEUE_SIZE = 100
FPS = 1

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

# Load models and label mappings
try:
    eyes_model = load_model(
        os.path.join(model_dir, "face_eyes_state_MobileNetV3Small_16_30epochs")
    )
    actions_model = load_model(
        os.path.join(model_dir, "face_eyes_state_MobileNetV3Small_16_30epochs")
    )

    with open(os.path.join(model_dir, "label_mapping_face_eyes_state.json"), "r") as f:
        eyes_label_to_index = json.load(f)
    with open(os.path.join(model_dir, "label_mapping_face_eyes_state.json"), "r") as f:
        actions_label_to_index = json.load(f)

    eyes_index_to_label = {int(v): k for k, v in eyes_label_to_index.items()}
    actions_index_to_label = {int(v): k for k, v in actions_label_to_index.items()}

    logger.info("Models and label mappings loaded successfully")
except Exception as e:
    logger.error(f"Error loading models or label mappings: {str(e)}")
    raise

# Queues and buffers
frame_queue = Queue(maxsize=QUEUE_SIZE)
prediction_queue = Queue(maxsize=1)
display_frame_queue = Queue(maxsize=1)


def preprocess_image(frame, target_size=(224, 224)):
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)
    normalized_rgb_img = resized_rgb_img.astype(np.float32) / 255.0
    return normalized_rgb_img


def predict_batch(batch, model, index_to_label):
    predictions = model.predict(np.array(batch), batch_size=BATCH_SIZE)
    predicted_labels = [index_to_label[np.argmax(pred)] for pred in predictions]
    return predicted_labels


def process_stream(stream_url):
    logger.info(f"Starting stream processing from {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    while True:
        batch_frames = []
        for _ in range(BATCH_SIZE):
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame from {stream_url}")
                time.sleep(0.1)
                continue
            batch_frames.append(frame)

        if len(batch_frames) == BATCH_SIZE:
            middle_frame = batch_frames[BATCH_SIZE // 2]
            if not display_frame_queue.full():
                display_frame_queue.put(middle_frame)
            else:
                try:
                    display_frame_queue.get_nowait()
                    display_frame_queue.put(middle_frame)
                except Empty:
                    pass

            processed_batch = [preprocess_image(frame) for frame in batch_frames]
            if not frame_queue.full():
                frame_queue.put(processed_batch)
            else:
                logger.warning("Frame queue is full, dropping batch")

        time.sleep(1 / FPS)


def process_predictions():
    while True:
        try:
            batch = frame_queue.get(timeout=1)
            start_time = time.time()
            eyes_predictions = predict_batch(batch, eyes_model, eyes_index_to_label)
            actions_predictions = predict_batch(
                batch, actions_model, actions_index_to_label
            )
            end_time = time.time()

            logger.info(
                f"Batch processed. Eyes: {eyes_predictions[:5]}..., Actions: {actions_predictions[:5]}..."
            )
            logger.info(f"Prediction time: {end_time - start_time:.4f} seconds")

            if not prediction_queue.full():
                prediction_queue.put((eyes_predictions, actions_predictions))
            else:
                try:
                    prediction_queue.get_nowait()
                    prediction_queue.put((eyes_predictions, actions_predictions))
                except Empty:
                    pass
        except Empty:
            pass


@stream_viewer.route("/")
def index():
    return render_template("index.html")


def gen_frames():
    while True:
        frame = display_frame_queue.get()
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@stream_viewer.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@stream_viewer.route("/predictions")
def get_predictions():
    try:
        if not prediction_queue.empty():
            eyes_pred, actions_pred = prediction_queue.get()
            return jsonify(
                {"eyes_predictions": eyes_pred, "actions_predictions": actions_pred}
            )
        else:
            logger.warning("Prediction queue is empty")
            return jsonify({"eyes_predictions": [], "actions_predictions": []})
    except Exception as e:
        logger.error(f"Error in get_predictions: {str(e)}")
        return jsonify({"error": str(e)}), 500


def start_streams():
    face_stream_url = "http://192.168.2.175/stream"  # Adjust if needed
    logger.info(f"Starting streams with URL: {face_stream_url}")

    # Start stream processing thread
    threading.Thread(
        target=process_stream, args=(face_stream_url,), daemon=True
    ).start()

    # Start prediction processing thread
    threading.Thread(target=process_predictions, daemon=True).start()

    logger.info("Streams started")


@stream_viewer.route("/stream_info", methods=["GET"])
def get_stream_info():
    return {
        "frame_queue_size": frame_queue.qsize(),
        "prediction_queue_size": prediction_queue.qsize(),
        "display_queue_size": display_frame_queue.qsize(),
        "batch_size": BATCH_SIZE,
        "fps": FPS,
    }
