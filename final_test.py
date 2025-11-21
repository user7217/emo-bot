import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time
import threading
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

try:
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, silent=True)
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

model_id = "microsoft/Florence-2-base"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def wrap_text(text, max_chars=60):
    words = text.split()
    line1 = []
    line2 = []
    current_len = 0
    
    for word in words:
        if current_len + len(word) < max_chars:
            line1.append(word)
            current_len += len(word) + 1
        else:
            line2.append(word)
            
    str_line1 = " ".join(line1)
    str_line2 = " ".join(line2)
    
    if len(str_line2) > max_chars:
        str_line2 = str_line2[:max_chars-3] + "..."
        
    return str_line1, str_line2

current_emotion = 'Detecting...'
display_line1 = 'Initializing AI...'
display_line2 = ''
is_florence_busy = False

def run_florence_worker(frame_rgb):
    global display_line1, display_line2, is_florence_busy

    try:
        pil_image = Image.fromarray(frame_rgb)
        prompt = "<MORE_DETAILED_CAPTION>" 
        
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(pil_image.width, pil_image.height)
        )
        
        raw_desc = parsed['<MORE_DETAILED_CAPTION>']
        
        l1, l2 = wrap_text(raw_desc)
        display_line1 = l1
        display_line2 = l2
        
    except Exception:
        pass
    finally:
        is_florence_busy = False

cap = cv2.VideoCapture(0)

last_emotion_time = 0
EMOTION_DELAY = 0.35

last_desc_time = 0
DESC_DELAY = 4.0 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)

    ih, iw, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(frame_rgb)
    
    biggest_face = None
    biggest_box = None

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            w, h = int(bboxC.width * iw), int(bboxC.height * ih)
            x, y = max(0, x), max(0, y)
            w, h = min(w, iw - x), min(h, ih - y)

            if w > 0 and h > 0:
                if biggest_face is None or w * h > biggest_face.size:
                    biggest_face = frame[y:y+h, x:x+w]
                    biggest_box = (x, y, w, h)

    if biggest_face is not None:
        bx, by, bw, bh = biggest_box
        
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)

        if time.time() - last_emotion_time > EMOTION_DELAY:
            try:
                face_roi = cv2.resize(biggest_face, (180, 180))
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                res = DeepFace.analyze(face_roi_rgb, actions=['emotion'], 
                                     enforce_detection=False, detector_backend='skip', silent=True)
                if isinstance(res, list): res = res[0]
                current_emotion = res['dominant_emotion']
                last_emotion_time = time.time()
            except: pass

        cv2.putText(frame, f"Mood: {current_emotion}", (bx, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if (time.time() - last_desc_time > DESC_DELAY) and not is_florence_busy:
            is_florence_busy = True
            last_desc_time = time.time()
            threading.Thread(target=run_florence_worker, args=(frame_rgb.copy(),)).start()

    cv2.rectangle(frame, (0, ih - 80), (iw, ih), (0, 0, 0), -1)
    
    cv2.putText(frame, display_line1, (10, ih - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, display_line2, (10, ih - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Visual Description AI', frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()