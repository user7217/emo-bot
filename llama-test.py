import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time
import threading
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import ollama
import random
import warnings
import os
import textwrap

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class VisionSystem:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.running = False
        
        self.current_emotion = "neutral"
        self.current_desc = "A person is present"
        
        self.latest_frame = None
        self.face_box = None 
        self.lock = threading.Lock()

        print("[Vision] Loading AI Models...")
        
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

        try:
            DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), 
                           actions=['emotion'], enforce_detection=False, silent=True)
        except: pass

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model_id = "microsoft/Florence-2-base"
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        
        print("[Vision] Models Loaded.")

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index)
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

    def get_data(self):
        with self.lock:
            return self.latest_frame, self.face_box, self.current_emotion, self.current_desc

    def _capture_loop(self):
        last_emotion_time = 0
        last_desc_time = 0
        EMOTION_DELAY = 0.5
        DESC_DELAY = 4.0

        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ih, iw, _ = frame.shape

            results = self.mp_face_detection.process(rgb_frame)
            
            current_box = None
            
            if results.detections:
                detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
                bboxC = detection.location_data.relative_bounding_box
                bx, by = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                bw, bh = int(bboxC.width * iw), int(bboxC.height * ih)
                bx, by = max(0, bx), max(0, by)
                bw, bh = min(bw, iw - bx), min(bh, ih - by)
                
                current_box = (bx, by, bw, bh)

                if time.time() - last_emotion_time > EMOTION_DELAY and bw > 0 and bh > 0:
                    try:
                        face_roi = frame[by:by+bh, bx:bx+bw]
                        res = DeepFace.analyze(face_roi, actions=['emotion'], 
                                             enforce_detection=False, detector_backend='skip', silent=True)
                        with self.lock:
                            self.current_emotion = res[0]['dominant_emotion']
                        last_emotion_time = time.time()
                    except: pass

            if time.time() - last_desc_time > DESC_DELAY:
                threading.Thread(target=self._run_florence, args=(rgb_frame.copy(),)).start()
                last_desc_time = time.time()
            
            with self.lock:
                self.latest_frame = frame.copy()
                self.face_box = current_box

    def _run_florence(self, image_np):
        try:
            pil_image = Image.fromarray(image_np)
            prompt = "<DETAILED_CAPTION>"
            inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device, self.torch_dtype)
            
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,
                do_sample=False,
                num_beams=3
            )
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            desc = self.processor.post_process_generation(text, task=prompt, image_size=(pil_image.width, pil_image.height))[prompt]
            
            with self.lock:
                self.current_desc = desc
        except: pass

class LlamaRoaster:
    def __init__(self, model="llama3.1:8b", max_tokens=60):
        self.model = model
        self.max_tokens = max_tokens
        self.SYSTEM_PROMPT = """
You are a witty roast generator for a robot.

You create short, playful insults based on appearance, expression, speech, and tone.

Never target protected traits. Keep roasts 1–2 lines max.

If greeting is simple (“hi”, “hello”), reply lightly and DO NOT roast.



Your output MUST use:



roast: "<text>"

tone: "<mocking | sarcastic | deadpan | playful | annoyed>"

expression: "<smirk | left-smirk | deadpan | wide-eyes | confused-eyes | side-eye | angry-eyes | happy-blink | tired-blink>"
"""
        print('[Roaster] Warming up Llama...')
        ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}])
        print('[Roaster] Ready.')

    def generate(self, appearance, expression, speech, tone):
        user_prompt = f"""
Target Appearance: "{appearance}"
Target Expression: "{expression}"
Target said: "{speech}"
Target Tone: "{tone}"
"""
        response = ollama.chat(
            model=self.model, 
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={"num_predict": self.max_tokens, "temperature": 0.8}
        )        
        return response["message"]["content"]

if __name__ == "__main__":
    vision = VisionSystem()
    roaster = LlamaRoaster()
    vision.start()
    
    sim_speeches = ["Hello robot", "Do I look good?", "Roast me!", "What are you looking at?"]
    sim_tones = ["confident", "shy", "aggressive", "bored", "happy"]
    
    last_roast_time = time.time()
    ROAST_INTERVAL = 5 
    
    display_text = "Waiting..."
    
    def run_roast_thread(app, emo, speech, tone):
        global display_text
        
        print("\n--- NEW INTERACTION ---")
        print(f"👀 SEES: {app}")
        print(f"🎭 MOOD: {emo}")
        print(f"🗣️ HEARS: '{speech}' ({tone})")
        
        display_text = "Thinking..."
        res = roaster.generate(app, emo, speech, tone)
        clean_result = res.replace('roast:', '').replace('"', '').strip()
        
        display_text = f"ROBOT: {clean_result}"
        print(f"🤖 ROAST: {clean_result}")
        print("-" * 30)

    try:
        while True:
            frame, box, emotion, desc = vision.get_data()
            
            if frame is None:
                time.sleep(0.1)
                continue

            if box:
                bx, by, bw, bh = box
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            ih, iw, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, ih-150), (iw, ih), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            wrapped_lines = textwrap.wrap(display_text, width=60)
            for i, line in enumerate(wrapped_lines):
                cv2.putText(frame, line, (20, ih-120 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Robot Vision", frame)
            
            if time.time() - last_roast_time > ROAST_INTERVAL:
                s = random.choice(sim_speeches)
                t = random.choice(sim_tones)
                threading.Thread(target=run_roast_thread, args=(desc, emotion, s, t)).start()
                last_roast_time = time.time()

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        cv2.destroyAllWindows()