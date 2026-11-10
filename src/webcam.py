
from pathlib import Path
import json
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import winsound
import wave
import math
import struct

# ==========================================
# PATHS AND MODEL
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_eye_cnn.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"
ALARM_PATH = ROOT / "emergency_alarm.wav"

if not MODEL_PATH.exists():
    raise SystemExit("Trained model not found.")

if not CLASS_PATH.exists():
    raise SystemExit("class_names.json not found.")

model = tf.keras.models.load_model(MODEL_PATH)

class_names = json.loads(CLASS_PATH.read_text())
closed_index = class_names.index("closed")

# ==========================================
# SETTINGS
# ==========================================

CLOSED_THRESHOLD = 0.70
ALERT_AFTER_SECONDS = 3.0

closed_since = None
alarm_active = False

# ==========================================
# CREATE EMERGENCY SIREN SOUND
# ==========================================

def create_alarm():

    sample_rate = 44100

    # Alternating high and low siren tones
    duration_per_tone = 0.35

    frequencies = [
        750,
        1100,
        750,
        1100,
        750,
        1100
    ]

    with wave.open(str(ALARM_PATH), "w") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        frames = bytearray()

        for freq in frequencies:

            total = int(sample_rate * duration_per_tone)

            for i in range(total):

                envelope = min(
                    1.0,
                    i / 500,
                    (total - i) / 500
                )

                value = int(
                    14000 *
                    envelope *
                    math.sin(
                        2 * math.pi * freq * i / sample_rate
                    )
                )

                frames.extend(
                    struct.pack("<h", value)
                )

        wav.writeframes(frames)


create_alarm()

# ==========================================
# MEDIAPIPE FACE MESH
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    face_mesh.close()
    raise SystemExit("Webcam not available.")

# ==========================================
# EYE CROP FUNCTION
# ==========================================

def get_eye_crop(frame, landmarks, indices):

    height, width = frame.shape[:2]

    points = np.array([
        [
            int(landmarks[i].x * width),
            int(landmarks[i].y * height)
        ]
        for i in indices
    ])

    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)

    pad_x = max(5, int((x2 - x1) * 0.30))
    pad_y = max(5, int((y2 - y1) * 0.60))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


# ==========================================
# MAIN LOOP
# ==========================================

print("Press Q to quit.")

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Could not read webcam frame.")
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        closed_probs = []
        detected_eyes = 0

        # ==================================
        # EYE DETECTION
        # ==================================

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark

            eye_list = [
                ("LEFT", LEFT_EYE),
                ("RIGHT", RIGHT_EYE)
            ]

            for eye_name, eye_indices in eye_list:

                crop, box = get_eye_crop(
                    frame,
                    landmarks,
                    eye_indices
                )

                if crop is None:
                    continue

                gray = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2GRAY
                )

                gray = cv2.resize(
                    gray,
                    (64, 64),
                    interpolation=cv2.INTER_AREA
                )

                batch = (
                    gray.astype("float32")[None, ..., None]
                    / 255.0
                )

                prediction = model.predict(
                    batch,
                    verbose=0
                )[0]

                closed_prob = float(
                    prediction[closed_index]
                )

                closed_probs.append(closed_prob)
                detected_eyes += 1

                label = (
                    "CLOSED"
                    if closed_prob >= CLOSED_THRESHOLD
                    else "OPEN"
                )

                color = (
                    (0, 0, 255)
                    if label == "CLOSED"
                    else (0, 200, 0)
                )

                x1, y1, x2, y2 = box

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{eye_name}: {label} {closed_prob:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2
                )

        # ==================================
        # CLOSURE TIMER
        # ==================================

        now = time.time()

        # Both eyes must be detected.
        # At least one eye must be classified CLOSED.

        closure_detected = (
            len(closed_probs) == 2
            and any(
                p >= CLOSED_THRESHOLD
                for p in closed_probs
            )
        )

        if closure_detected:

            if closed_since is None:
                closed_since = now

            duration = now - closed_since

        else:

            closed_since = None
            duration = 0.0

        # ==================================
        # EMERGENCY ALARM
        # ==================================

        if duration >= ALERT_AFTER_SECONDS:

            if not alarm_active:

                winsound.PlaySound(
                    str(ALARM_PATH),
                    winsound.SND_FILENAME
                    | winsound.SND_ASYNC
                    | winsound.SND_LOOP
                )

                alarm_active = True

        else:

            if alarm_active:

                winsound.PlaySound(
                    None,
                    winsound.SND_PURGE
                )

                alarm_active = False

        # ==================================
        # STATUS DISPLAY
        # ==================================

        if detected_eyes < 2:

            status = "EYES NOT DETECTED"
            color = (0, 200, 255)

        elif duration >= ALERT_AFTER_SECONDS:

            status = "EMERGENCY WARNING - ALARM ON"
            color = (0, 0, 255)

        else:

            status = "MONITORING"
            color = (0, 200, 0)

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        # ==================================
        # TIMER DISPLAY
        # ==================================

        cv2.putText(
            frame,
            f"Closure timer: {duration:.1f}s / 3.0s",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2
        )

        # ==================================
        # SHOW WEBCAM
        # ==================================

        cv2.imshow(
            "CNN Eye State - MediaPipe",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    winsound.PlaySound(
        None,
        winsound.SND_PURGE
    )

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
