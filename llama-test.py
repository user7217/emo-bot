import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time
import threading
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import ollama
import random
import warnings
import os
import textwrap
import sounddevice as sd
from kokoro_onnx import Kokoro

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class KokoroTTS:
    def __init__(self):
        try:
            cwd = os.getcwd()
            model_path = os.path.join(cwd, "kokoro-v1.0.onnx")
            voices_path = os.path.join(cwd, "voices-v1.0.bin")
            
            if not os.path.exists(model_path):
                print(f"Could not find '{model_path}'.")
            
            self.kokoro = Kokoro(model_path, voices_path)
            
            self.is_speaking = False
            self.tones = {
                "mocking": {"voice": "af_bella", "speed": 1.1},
                "sarcastic": {"voice": "mix_sarcasm", "speed": 0.8},
                "deadpan": {"voice": "am_adam", "speed": 0.9},
                "playful": {"voice": "af_sarah", "speed": 1.2},
                "annoyed": {"voice": "am_michael", "speed": 0.8}
            }
            v1 = self.kokoro.voices["af_bella"]
            v2 = self.kokoro.voices["am_adam"]
            self.sarcastic_mix = (v1 * 0.5) + (v2 * 0.5)

        except Exception as e:
            print(f"TTS Error: {e}")
            exit()

    def speak(self, text, tone_category="deadpan"):
        if self.is_speaking: return
        self.is_speaking = True
        
        profile = self.tones.get(tone_category.lower(), self.tones["deadpan"])
        voice = profile["voice"]
        speed = profile["speed"]
        
        if voice == "mix_sarcasm":
            voice_data = self.sarcastic_mix
        else:
            voice_data = voice
            
        try:
            audio, sample_rate = self.kokoro.create(text, voice=voice_data, speed=speed, lang="en-us")
            sd.play(audio, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.is_speaking = False

class VisionSystem:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.running = False
        self.current_emotion = "neutral"
        self.current_desc = "A person is present"
        self.latest_frame = None
        self.face_box = None 
        self.lock = threading.Lock()
        
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

        try:
            DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=['emotion'], enforce_detection=False, silent=True)
        except: pass

        self.device = "cpu" 
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        
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

                if time.time() - last_emotion_time > 0.5 and bw > 0 and bh > 0:
                    try:
                        face_roi = frame[by:by+bh, bx:bx+bw]
                        res = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='skip', silent=True)
                        with self.lock:
                            self.current_emotion = res[0]['dominant_emotion']
                        last_emotion_time = time.time()
                    except: pass

            if time.time() - last_desc_time > 10:
                threading.Thread(target=self._run_blip, args=(rgb_frame.copy(),)).start()
                last_desc_time = time.time()
            
            with self.lock:
                self.latest_frame = frame.copy()
                self.face_box = current_box

    def _run_blip(self, image_np):
        try:
            pil_image = Image.fromarray(image_np)
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=50)
            desc = self.processor.decode(out[0], skip_special_tokens=True)
            
            with self.lock:
                self.current_desc = desc
        except Exception: 
            pass

class LlamaRoaster:
    def __init__(self, model="llama3.1:8b"):
        self.model = model
        self.SYSTEM_PROMPT = "You are a witty roast generator. Create short insults based on appearance, expression, and tone. Output MUST use format: roast: \"<text>\" tone: \"<mocking|sarcastic|deadpan|playful|annoyed>\""
        try:
            ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}])
        except:
            print("Is Ollama running?")

    def generate(self, appearance, expression, speech, tone):
        user_prompt = f"Appearance: {appearance}\nExpression: {expression}\nSaid: {speech}\nTone: {tone}"
        try:
            response = ollama.chat(model=self.model, messages=[{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], options={"num_predict": 60, "temperature": 0.8})
            return response["message"]["content"]
        except: return "roast: \"I am broken.\" tone: \"deadpan\""

if __name__ == "__main__":
    vision = VisionSystem()
    roaster = LlamaRoaster()
    tts = KokoroTTS()
    vision.start()
    
    sim_speeches = ["Hello robot", "Do I look good?", "Roast me!", "What are you looking at?"]
    sim_tones = ["confident", "shy", "aggressive", "bored", "happy"]
    
    last_roast_time = time.time()
    display_text = "Waiting..."
    
    def run_roast_thread(app, emo, speech, tone_in):
        global display_text
        if tts.is_speaking: return
        
        display_text = "Thinking..."
        res = roaster.generate(app, emo, speech, tone_in)
        
        roast_text = ""
        tone_cat = "deadpan"
        
        lines = res.split('\n')
        for line in lines:
            if "roast:" in line.lower():
                try: roast_text = line.split(':', 1)[1].replace('"', '').strip()
                except: pass
            if "tone:" in line.lower():
                try: tone_cat = line.split(':', 1)[1].replace('"', '').strip()
                except: pass
        
        if not roast_text: roast_text = res

        display_text = f"ROBOT: {roast_text}"
        print(f"Generated Roast: {roast_text} | Tone: {tone_cat}")
        tts.speak(roast_text, tone_cat)

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
            
            if time.time() - last_roast_time > 10 and not tts.is_speaking:
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