import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time

# -------------------------------------
# 1. Initialize MediaPipe
# -------------------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# -------------------------------------
# 2. Warm up DeepFace model
# -------------------------------------
print("Loading Emotion model... (this might take a moment)")
try:
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, silent=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warm-up warning: {e}")

cap = cv2.VideoCapture(0)

# Throttle emotion detection (DeepFace is heavy)
last_emotion_time = 0
EMOTION_DELAY = 0.35  # seconds

last_emotion = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    ih, iw, _ = frame.shape

    # MediaPipe requires RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    biggest_face = None
    biggest_box = None

    # -------------------------------------
    # 3. Pick the biggest face only (Faster)
    # -------------------------------------
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box

            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            x, y = max(0, x), max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)

            if w > 0 and h > 0:
                if biggest_face is None or w * h > biggest_face.size:
                    face_roi = frame[y:y+h, x:x+w]
                    biggest_face = face_roi
                    biggest_box = (x, y, w, h)

    # -------------------------------------
    # 4. Only analyze the biggest face + throttle
    # -------------------------------------
    if biggest_face is not None:
        x, y, w, h = biggest_box

        # Resize DOWN → much faster DeepFace call
        resized_face = cv2.resize(biggest_face, (180, 180))

        # Convert to RGB once
        resized_face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

        # Throttle DeepFace calls
        if time.time() - last_emotion_time > EMOTION_DELAY:
            try:
                result = DeepFace.analyze(
                    resized_face_rgb,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )

                # DeepFace sometimes returns a dict, sometimes list
                if isinstance(result, list):
                    result = result[0]

                last_emotion = result['dominant_emotion']
                last_emotion_time = time.time()

            except Exception:
                pass

        # Draw results
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if last_emotion:
            cv2.putText(frame, last_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

    # -------------------------------------
    # Display frame
    # -------------------------------------
    cv2.imshow('Face & Emotion', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
