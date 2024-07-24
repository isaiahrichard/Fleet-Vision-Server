from flask import Blueprint, render_template, Response
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import json
import logging
import time
import threading
from queue import Queue, Empty
from helpers.model import (
    classify_eye_batch,
    classify_main_batch,
)
import base64
from flask_cors import cross_origin

stream_viewer = Blueprint("stream_viewer", __name__)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 5
PREDICTION_THRESHOLD = 0.7
EVENT_BATCH_SIZE = 40

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

frame_count = 0

# Load models and label mappings
try:
    # eyes_model = load_model(
    #     os.path.join(model_dir, "face_eyes_state_MobileNetV3Small_16_30epochs")
    # )
    actions_model = load_model(
        os.path.join(model_dir, "body_driver_actions_MobileNetV3Small_32_3epochs")
    )

    with open(os.path.join(model_dir, "label_mapping_face_eyes_state.json"), "r") as f:
        eyes_label_to_index = json.load(f)
    with open(
        os.path.join(model_dir, "label_mapping_body_driver_actions.json"), "r"
    ) as f:
        actions_label_to_index = json.load(f)

    eyes_index_to_label = {int(v): k for k, v in eyes_label_to_index.items()}
    actions_index_to_label = {int(v): k for k, v in actions_label_to_index.items()}

    logger.info("Models and label mappings loaded successfully")
except Exception as e:
    logger.error(f"Error loading models or label mappings: {str(e)}")
    raise


def preprocess_image(frame, target_size=(224, 224)):
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)
    normalized_rgb_img = resized_rgb_img.astype(np.float32) / 255.0
    return normalized_rgb_img


def predict_batch(batch, model, index_to_label):
    predictions = model.predict(np.array(batch), batch_size=BATCH_SIZE)
    predicted_labels = []
    for pred in predictions:
        max_prob = np.max(pred)
        if max_prob >= PREDICTION_THRESHOLD:
            predicted_labels.append(index_to_label[np.argmax(pred)])
        else:
            predicted_labels.append("Unknown")
    return predicted_labels


def process_stream(stream_url):
    global frame_count
    cap = cv2.VideoCapture(stream_url)
    currBufferSize = 0
    eyes_predictions_buffer = []
    actions_predictions_buffer = []
    prevEvent = {}
    while True:
        batch_frames = []
        batch_start_frame_count = frame_count
        for _ in range(BATCH_SIZE):
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame from {stream_url}")
                time.sleep(0.1)
                continue
            batch_frames.append(frame)
            frame_count += 1
            currBufferSize += 1

        if len(batch_frames) == BATCH_SIZE:
            processed_batch = [preprocess_image(frame) for frame in batch_frames]
            # eyes_predictions = predict_batch(
            #     processed_batch, eyes_model, eyes_index_to_label
            # )
            actions_predictions = predict_batch(
                processed_batch, actions_model, actions_index_to_label
            )
            # eyes_predictions_buffer += eyes_predictions
            actions_predictions_buffer += actions_predictions

            action_event = 0

            if currBufferSize >= EVENT_BATCH_SIZE:
                # eye_event_label = classify_eye_batch(eyes_predictions_buffer)
                action_event_label = classify_main_batch(actions_predictions_buffer)

                cont = (
                    1
                    if "label" in prevEvent and prevEvent["label"] == action_event_label
                    else 0
                )
                # add_eye_event(
                #     {
                #         "frameStart": batch_start_frame_count,
                #         "frameEnd": frame_count,
                #         "label": eye_event_label,
                #     }
                # )
                action_event = {
                    "frameStart": frame_count - EVENT_BATCH_SIZE,
                    "frameEnd": frame_count,
                    "label": action_event_label,
                    "cont": cont,
                }
                prevEvent = action_event

                # eyes_predictions_buffer = []
                actions_predictions_buffer = []
                currBufferSize = 0

            middle_frame = batch_frames[BATCH_SIZE // 2]
            _, buffer = cv2.imencode(".jpg", middle_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            yield (
                f"data: {{\n"
                f'data: "image": "{frame_base64}",\n'
                f'data: "action_event": "{action_event}",\n'
                f'data: "first_frame_num": "{batch_start_frame_count + 1}",\n'
                f'data: "actions_predictions": {json.dumps(actions_predictions)}\n'
                f"data: }}\n\n"
            )


@stream_viewer.route("/")
def index():
    return render_template("index.html")


@stream_viewer.route("/video_feed")
def video_feed():
    face_stream_url = "http://172.20.10.7/stream"  # Adjust if needed
    res = process_stream(face_stream_url)
    return Response(res, mimetype="text/event-stream")


@stream_viewer.route("/stream_info", methods=["GET"])
def get_stream_info():
    return {
        "batch_size": BATCH_SIZE,
        "prediction_threshold": PREDICTION_THRESHOLD,
    }
