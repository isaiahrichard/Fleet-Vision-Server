from flask import Blueprint, render_template, Response
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import json
import logging
import time
from helpers.model import (
    classify_main_batch,
)
import base64

stream_viewer = Blueprint("stream_viewer", __name__)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE_DISTRACTION = 5
BATCH_SIZE_EYES_STATE = 5
BINARY_EYES_STATE_PREDICTION_THRESHOLD = 0.3
BINARY_DISTRACTION_THRESHOLD = 0.3
EVENT_BATCH_SIZE_DISTRACTION = 40
EVENT_BATCH_SIZE_EYES_STATE = 40

# Setup cv2 classifiers to detect eyes and faces
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

frame_count = 0

# Load models and label mappings
try:
    binary_eyes_state_model = load_model(
        os.path.join(
            model_dir, "binary_MobileNetV3Small_32batchsize_1e-06learningrate_4epochs"
        )
    )
    binary_distraction_model = load_model(
        os.path.join(
            model_dir, "binary_MobileNetV3Small_32batchsize_1e-06learningrate_4epochs"
        )
    )

    with open(
        os.path.join(model_dir, "label_mapping_binary_eyes_state.json"), "r"
    ) as f:
        eyes_label_to_index = json.load(f)
    with open(
        os.path.join(model_dir, "label_mapping_binary_distraction.json"), "r"
    ) as f:
        distraction_label_to_index = json.load(f)

    eyes_index_to_label = {int(v): k for k, v in eyes_label_to_index.items()}
    distraction_index_to_label = {
        int(v): k for k, v in distraction_label_to_index.items()
    }

    logger.info("Models and label mappings loaded successfully")
except Exception as e:
    logger.error(f"Error loading models or label mappings: {str(e)}")
    raise


def preprocess_image(frame, target_size=(224, 224)):
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)
    normalized_rgb_img = resized_rgb_img.astype(np.float32) / 255.0
    return normalized_rgb_img


# the image will already come in in grayscale as the model expects
def preprocess_image_face(frame, target_size=(224, 224)):
    # rgb_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_grayscale_img = cv2.resize(
        rgb_image, target_size, interpolation=cv2.INTER_AREA
    )
    normalized_grayscale_img = resized_grayscale_img.astype(np.float32) / 255.0
    return normalized_grayscale_img


def predict_batch(
    batch,
    model,
    index_to_label,
    is_distraction_model=False,
    curBatchSize=BATCH_SIZE_EYES_STATE,
):
    if is_distraction_model:
        predictions = model.predict(np.array(batch), batch_size=BATCH_SIZE_DISTRACTION)
    else:
        predictions = model.predict(np.array(batch), batch_size=curBatchSize)
    predicted_labels = []
    for pred in predictions:
        if is_distraction_model:
            if pred[0] >= BINARY_DISTRACTION_THRESHOLD:
                predicted_labels.append(index_to_label[1])  # Safe driving
            else:
                predicted_labels.append(index_to_label[0])  # Distracted
        else:
            if pred[0] >= BINARY_EYES_STATE_PREDICTION_THRESHOLD:
                predicted_labels.append(index_to_label[1])  # Eyes open
            else:
                predicted_labels.append(index_to_label[0])  # Eyes closed
    return predicted_labels


def process_stream(stream_url, model, index_to_label, is_distraction_model=False):
    global frame_count
    cap = cv2.VideoCapture(stream_url)
    currBufferSize = 0
    predictions_buffer = []
    prevEvent = {}
    while True:
        batch_frames = []
        batch_start_frame_count = frame_count
        for _ in range(BATCH_SIZE_DISTRACTION):
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame from {stream_url}")
                time.sleep(0.1)
                continue
            batch_frames.append(frame)
            frame_count += 1
            currBufferSize += 1

        if len(batch_frames) == BATCH_SIZE_DISTRACTION:
            processed_batch = [preprocess_image(frame) for frame in batch_frames]
            predictions = predict_batch(
                processed_batch, model, index_to_label, is_distraction_model
            )
            predictions_buffer += predictions

            event = 0

            if currBufferSize >= EVENT_BATCH_SIZE_DISTRACTION:
                event_label = classify_main_batch(predictions_buffer)

                cont = (
                    1
                    if "label" in prevEvent and prevEvent["label"] == event_label
                    else 0
                )
                event = {
                    "frameStart": frame_count - EVENT_BATCH_SIZE_DISTRACTION,
                    "frameEnd": frame_count,
                    "label": event_label,
                    "cont": cont,
                }
                prevEvent = event

                predictions_buffer = []
                currBufferSize = 0

            middle_frame = batch_frames[BATCH_SIZE_DISTRACTION // 2]
            _, buffer = cv2.imencode(".jpg", middle_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            yield (
                f"data: {{\n"
                f'data: "image": "{frame_base64}",\n'
                f'data: "event": "{event}",\n'
                f'data: "first_frame_num": "{batch_start_frame_count + 1}",\n'
                f'data: "predictions": {json.dumps(predictions)}\n'
                f"data: }}\n\n"
            )


# process the face stream
# use cv2 haarcascade classifier to detect eyes
# then use the binary_eyes_state_model to predict the state of each eye that was detected
def process_stream_face(stream_url, model, index_to_label, is_distraction_model=False):
    global frame_count
    cap = cv2.VideoCapture(stream_url)
    currBufferSize = 0
    predictions_buffer = []
    prevEvent = {}
    while True:
        batch_frames = []
        batch_start_frame_count = frame_count
        while len(batch_frames) < BATCH_SIZE_EYES_STATE:
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame from {stream_url}")
                time.sleep(0.1)
                continue

            # Convert to grayscale for face and eye detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)

            frame_with_boxes = frame

            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]  # Get the first face
                # Draw rectangle around the face
                cv2.rectangle(
                    frame_with_boxes, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2
                )

                # Region of interest for eyes detection
                roi_gray = gray[fy : fy + fh, fx : fx + fw]

                # Detect eyes within the face region
                eyes = eyeCascade.detectMultiScale(roi_gray, 1.05, 5)

                eye_images = []
                for ex, ey, ew, eh in eyes:
                    # Draw rectangle around the eye on the main frame
                    cv2.rectangle(
                        frame_with_boxes,
                        (fx + ex, fy + ey),
                        (fx + ex + ew, fy + ey + eh),
                        (0, 255, 0),
                        1,
                    )
                    eye_roi = roi_gray[ey : ey + eh, ex : ex + ew]
                    eye_images.append(eye_roi)

                # If only one eye is detected, add None for the second eye
                if len(eye_images) == 1:
                    eye_images.append(None)
                elif len(eye_images) == 0:
                    eye_images = [None, None]
                elif len(eye_images) > 2:
                    eye_images = eye_images[
                        :2
                    ]  # Take only the first two eyes if more are detected

                batch_frames.append((frame_with_boxes, eye_images[0], eye_images[1]))
                frame_count += 1
                currBufferSize += 1
            else:
                # If no face detected, add frame with no detections
                batch_frames.append((frame_with_boxes, None, None))
                frame_count += 1
                currBufferSize += 1

        if len(batch_frames) == BATCH_SIZE_EYES_STATE:
            # Process the batch
            processed_batch = []
            # we need to keep track of the frames that didn't have any eyes detected
            # so that we can insert unknown predictions at the appropriate indices for them
            unknown_frame_indices = []
            curr_index = 0
            num_images_in_model_batch = 0
            for frame_data in batch_frames:
                # in this case, we detected two eyes
                if frame_data[1] is not None and frame_data[2] is not None:
                    # for now we will just process and feed the left eye into the model and ignore the other one
                    # thus we are only making predictions on the overall eyes state based on a single eye
                    # this should work fine for now and it simplifies this code greatly
                    processed_batch.append(
                        preprocess_image_face(frame_data[0], (224, 224))
                    )
                    num_images_in_model_batch += 1

                # in this case, we detected one eye
                elif frame_data[1] is not None and frame_data[2] is None:
                    processed_batch.append(
                        preprocess_image_face(frame_data[0], (224, 224))
                    )
                    num_images_in_model_batch += 1

                # in this case, we detected no eyes
                # no need to preprocess as we have nothing to feed to the model
                else:
                    unknown_frame_indices.append(curr_index)
                curr_index += 1

            predictions = []
            if num_images_in_model_batch > 0:
                predictions = predict_batch(
                    processed_batch,
                    model,
                    index_to_label,
                    is_distraction_model,
                    num_images_in_model_batch,
                )
                # Insert "Unknown" predictions for frames where no eyes were detected
                for idx in unknown_frame_indices:
                    predictions.insert(idx, "Unknown")
            else:  # If no eyes detected in any frame, insert "Unknown" for all
                predictions = ["Unknown"] * BATCH_SIZE_EYES_STATE

            predictions_buffer += predictions

            event = 0

            if currBufferSize >= EVENT_BATCH_SIZE_EYES_STATE:
                event_label = classify_main_batch(predictions_buffer)

                cont = (
                    1
                    if "label" in prevEvent and prevEvent["label"] == event_label
                    else 0
                )
                event = {
                    "frameStart": frame_count - EVENT_BATCH_SIZE_EYES_STATE,
                    "frameEnd": frame_count,
                    "label": event_label,
                    "cont": cont,
                }
                prevEvent = event

                predictions_buffer = []
                currBufferSize = 0

            middle_frame_data = batch_frames[BATCH_SIZE_EYES_STATE // 2]
            middle_frame, left_eye, right_eye = middle_frame_data

            middle_frame = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)

            # Encode the main frame (now with bounding boxes for face and eyes)
            _, buffer = cv2.imencode(".jpg", middle_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            yield (
                f"data: {{\n"
                f'data: "image": "{frame_base64}",\n'
                f'data: "event": "{event}",\n'
                f'data: "first_frame_num": "{batch_start_frame_count + 1}",\n'
                f'data: "predictions": {json.dumps(predictions)}\n'
                f"data: }}\n\n"
            )


# def process_stream_face(stream_url, model, index_to_label, is_distraction_model=False):
#     global frame_count
#     cap = cv2.VideoCapture(stream_url)
#     currBufferSize = 0
#     predictions_buffer = []
#     prevEvent = {}
#     while True:
#         batch_frames = []
#         batch_start_frame_count = frame_count
#         while len(batch_frames) < BATCH_SIZE_EYES_STATE:
#             ret, frame = cap.read()
#             if not ret:
#                 logger.error(f"Failed to read frame from {stream_url}")
#                 time.sleep(0.1)
#                 continue

#             # Convert to grayscale for face and eye detection
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # Detect faces
#             faces = faceCascade.detectMultiScale(gray, 1.1, 4)

#             frame_with_boxes = frame

#             if len(faces) > 0:
#                 fx, fy, fw, fh = faces[0]  # Get the first face
#                 # Draw rectangle around the face
#                 cv2.rectangle(
#                     frame_with_boxes, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2
#                 )

#                 # Region of interest for eyes detection
#                 roi_gray = gray[fy : fy + fh, fx : fx + fw]

#                 # Detect eyes within the face region
#                 eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 4)

#                 eye_images = []
#                 for ex, ey, ew, eh in eyes:
#                     # Draw rectangle around the eye on the main frame
#                     cv2.rectangle(
#                         frame_with_boxes,
#                         (fx + ex, fy + ey),
#                         (fx + ex + ew, fy + ey + eh),
#                         (0, 255, 0),
#                         1,
#                     )
#                     eye_roi = roi_gray[ey : ey + eh, ex : ex + ew]
#                     eye_images.append(eye_roi)

#                 # If only one eye is detected, add None for the second eye
#                 if len(eye_images) == 1:
#                     eye_images.append(None)
#                 elif len(eye_images) == 0:
#                     eye_images = [None, None]
#                 elif len(eye_images) > 2:
#                     eye_images = eye_images[
#                         :2
#                     ]  # Take only the first two eyes if more are detected

#                 batch_frames.append((frame_with_boxes, eye_images[0], eye_images[1]))
#                 frame_count += 1
#                 currBufferSize += 1
#             else:
#                 # If no face detected, add frame with no detections
#                 batch_frames.append((frame_with_boxes, None, None))
#                 frame_count += 1
#                 currBufferSize += 1

#         if len(batch_frames) == BATCH_SIZE_EYES_STATE:
#             # Process the batch
#             processed_batch = [
#                 preprocess_image_face(frame_data[0], (224, 224))
#                 for frame_data in batch_frames
#             ]
#             predictions = predict_batch(
#                 processed_batch, model, index_to_label, is_distraction_model
#             )
#             predictions_buffer += predictions

#             event = 0

#             if currBufferSize >= EVENT_BATCH_SIZE_EYES_STATE:
#                 event_label = classify_main_batch(predictions_buffer)

#                 cont = (
#                     1
#                     if "label" in prevEvent and prevEvent["label"] == event_label
#                     else 0
#                 )
#                 event = {
#                     "frameStart": frame_count - EVENT_BATCH_SIZE_EYES_STATE,
#                     "frameEnd": frame_count,
#                     "label": event_label,
#                     "cont": cont,
#                 }
#                 prevEvent = event

#                 predictions_buffer = []
#                 currBufferSize = 0

#             middle_frame_data = batch_frames[BATCH_SIZE_EYES_STATE // 2]
#             middle_frame, left_eye, right_eye = middle_frame_data

#             # Encode the main frame (now with bounding boxes for face and eyes)
#             _, buffer = cv2.imencode(".jpg", middle_frame)
#             frame_base64 = base64.b64encode(buffer).decode("utf-8")

#             yield (
#                 f"data: {{\n"
#                 f'data: "image": "{frame_base64}",\n'
#                 f'data: "event": "{event}",\n'
#                 f'data: "first_frame_num": "{batch_start_frame_count + 1}",\n'
#                 f'data: "predictions": {json.dumps(predictions)}\n'
#                 f"data: }}\n\n"
#             )


@stream_viewer.route("/")
def index():
    return render_template("index.html")


@stream_viewer.route("/face_stream")
def face_stream():
    face_stream_url = "http://172.20.10.6/stream"  # Adjust if needed
    return Response(
        process_stream_face(
            face_stream_url, binary_eyes_state_model, eyes_index_to_label
        ),
        mimetype="text/event-stream",
    )


@stream_viewer.route("/body_stream")
def body_stream():
    body_stream_url = "http://172.20.10.3/stream"  # Adjust if needed
    return Response(
        process_stream(
            body_stream_url, binary_distraction_model, distraction_index_to_label, True
        ),
        mimetype="text/event-stream",
    )


@stream_viewer.route("/stream_info", methods=["GET"])
def get_stream_info():
    return {
        "BATCH_SIZE_EYES_STATE": BATCH_SIZE_DISTRACTION,
        "BATCH_SIZE_EYES_STATE": BATCH_SIZE_EYES_STATE,
        "BINARY_EYES_STATE_PREDICTION_THRESHOLD": BINARY_EYES_STATE_PREDICTION_THRESHOLD,
        "BINARY_DISTRACTION_THRESHOLD": BINARY_DISTRACTION_THRESHOLD,
    }
