import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time
import threading
import torch
import queue
import tempfile
import os
import textwrap
import sounddevice as sd
import warnings
import mlx_whisper
import pyaudio
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import ollama
from kokoro_onnx import Kokoro

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 1. BRAIN UPGRADE: Using 8B model for better conversation logic
STT_MODEL = "mlx-community/distil-whisper-large-v3"
OLLAMA_MODEL = "llama3.1:8b" 

# Audio Settings
VAD_THRESHOLD = 0.5
SILENCE_TIMEOUT = 0.8      
MAX_RECORD_TIME = 10.0

# Identity Settings
IDENTITY_CHECK_INTERVAL = 1.5
IDENTITY_THRESHOLD = 0.7
REQUIRED_CONFIRMATIONS = 3 

GLOBAL_STATE = {
    "is_speaking": False,
    "is_processing": False
}

def find_cosine_distance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# --- 1. SPEECH OUTPUT ---
class KokoroTTS:
    def __init__(self):
        try:
            cwd = os.getcwd()
            model_path = os.path.join(cwd, "kokoro-v1.0.onnx")
            voices_path = os.path.join(cwd, "voices-v1.0.bin")
            self.kokoro = Kokoro(model_path, voices_path)
            
            v1 = self.kokoro.voices["af_bella"]
            v2 = self.kokoro.voices["am_adam"]
            self.sarcastic_mix = (v1 * 0.5) + (v2 * 0.5)
            
            self.tones = {
                "mocking": {"voice": "af_bella", "speed": 1.1},
                "sarcastic": {"voice": "mix_sarcasm", "speed": 0.9},
                "deadpan": {"voice": "am_adam", "speed": 0.9},
                "playful": {"voice": "af_sarah", "speed": 1.1},
                "annoyed": {"voice": "am_michael", "speed": 0.8}
            }
        except Exception as e:
            print(f"[!] TTS Error: {e}")
            exit()

    def speak(self, text, tone_category="deadpan"):
        GLOBAL_STATE["is_speaking"] = True
        profile = self.tones.get(tone_category.lower(), self.tones["deadpan"])
        voice_data = self.sarcastic_mix if profile["voice"] == "mix_sarcasm" else self.kokoro.voices[profile["voice"]]
        try:
            audio, sample_rate = self.kokoro.create(text, voice=voice_data, speed=profile["speed"], lang="en-us")
            sd.play(audio, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"[!] Audio Error: {e}")
        finally:
            time.sleep(0.2)
            GLOBAL_STATE["is_speaking"] = False

# --- 2. AUDIO INPUT ---
class Ear:
    def __init__(self, callback_function):
        self.callback = callback_function
        self.running = False
        self.rate = 16000
        print("[+] Loading VAD & Whisper...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        self.get_speech_timestamps = utils[0]
        print("[+] Ear Ready.")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _listen_loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=512)
        frames_buffer = []
        is_recording = False
        last_speech_time = 0
        buffer_start_time = 0

        while self.running:
            if GLOBAL_STATE["is_speaking"] or GLOBAL_STATE["is_processing"]:
                time.sleep(0.1)
                frames_buffer = [] 
                is_recording = False
                continue

            try:
                data = stream.read(512, exception_on_overflow=False)
            except: continue

            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.flatten().astype(np.float32) / 32768.0
            vad_tensor = torch.from_numpy(audio_float32)
            speech_prob = self.vad_model(vad_tensor, self.rate).item()
            is_speech = speech_prob > VAD_THRESHOLD
            current_time = time.time()

            if is_speech:
                if not is_recording:
                    is_recording = True
                    buffer_start_time = current_time
                    frames_buffer = [data]
                else:
                    frames_buffer.append(data)
                last_speech_time = current_time
            elif is_recording:
                frames_buffer.append(data)
                if (current_time - last_speech_time > SILENCE_TIMEOUT) or (current_time - buffer_start_time > MAX_RECORD_TIME):
                    is_recording = False
                    self._transcribe(b''.join(frames_buffer), p)
                    frames_buffer = []

    def _transcribe(self, audio_data, pa_instance):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import wave
            wf = wave.open(tmp.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pa_instance.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(audio_data)
            wf.close()
            tmp_path = tmp.name
        try:
            res = mlx_whisper.transcribe(tmp_path, path_or_hf_repo=STT_MODEL, language="en", verbose=False)
            text = res["text"].strip()
            
            # Strict Garbage Filter
            garbage = ["thank you", "subtitles", "you", "copyright", "audio", "bye"]
            if len(text) > 2 and text.lower() not in garbage: 
                print(f">> USER: {text}")
                self.callback(text)
        except Exception as e:
            print(f"[!] STT Error: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- 3. VISION SYSTEM ---
class VisionSystem:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.running = False
        self.current_emotion = "neutral"
        self.current_desc = "A person"
        self.latest_frame = None
        self.face_box = None 
        
        self.current_face_embedding = None
        self.new_user_detected = False
        self.new_face_confirm_count = 0 
        
        self.lock = threading.Lock()
        self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.device = "cpu"
        
        print("[+] Loading Vision Models...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        try: 
            DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=['emotion'], enforce_detection=False, silent=True)
            DeepFace.represent(np.zeros((224, 224, 3), dtype=np.uint8), model_name="ArcFace", enforce_detection=False)
        except: pass
        print("[+] Vision Ready.")
        
    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index)
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive(): self.thread.join()
        self.cap.release()

    def get_data(self):
        with self.lock:
            is_new = self.new_user_detected
            if is_new: self.new_user_detected = False
            return self.latest_frame, self.face_box, self.current_emotion, self.current_desc, is_new

    def _capture_loop(self):
        last_emo_t = 0
        last_desc_t = 0
        last_id_t = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ih, iw, _ = frame.shape
            results = self.mp_face.process(rgb)
            curr_box = None
            
            if results.detections:
                d = max(results.detections, key=lambda x: x.location_data.relative_bounding_box.width)
                bb = d.location_data.relative_bounding_box
                bx, by = int(bb.xmin * iw), int(bb.ymin * ih)
                bw, bh = int(bb.width * iw), int(bb.height * ih)
                
                if bw < (iw * 0.15): 
                    continue 

                curr_box = (max(0,bx), max(0,by), bw, bh)
                face_roi = frame[by:by+bh, bx:bx+bw]

                if time.time() - last_emo_t > 0.5:
                    try:
                        res = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='skip', silent=True)
                        with self.lock: self.current_emotion = res[0]['dominant_emotion']
                        last_emo_t = time.time()
                    except: pass

                if time.time() - last_id_t > IDENTITY_CHECK_INTERVAL:
                    try:
                        embedding = DeepFace.represent(face_roi, model_name="ArcFace", enforce_detection=False, detector_backend="skip")[0]["embedding"]
                        
                        with self.lock:
                            if self.current_face_embedding is None:
                                self.current_face_embedding = embedding
                            else:
                                distance = find_cosine_distance(self.current_face_embedding, embedding)
                                
                                if distance > IDENTITY_THRESHOLD:
                                    self.new_face_confirm_count += 1
                                    print(f"[?] Stranger... ({self.new_face_confirm_count}/{REQUIRED_CONFIRMATIONS})")
                                    
                                    if self.new_face_confirm_count >= REQUIRED_CONFIRMATIONS:
                                        self.new_user_detected = True
                                        self.current_face_embedding = embedding
                                        self.new_face_confirm_count = 0
                                        print(f"[!] NEW USER CONFIRMED.")
                                else:
                                    if self.new_face_confirm_count > 0:
                                        self.new_face_confirm_count = 0
                        
                        last_id_t = time.time()
                    except: pass

            if time.time() - last_desc_t > 10:
                threading.Thread(target=self._run_blip, args=(rgb.copy(),)).start()
                last_desc_t = time.time()
            
            with self.lock:
                self.latest_frame = frame.copy()
                self.face_box = curr_box

    def _run_blip(self, img):
        try:
            inputs = self.processor(Image.fromarray(img), return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=30)
            d = self.processor.decode(out[0], skip_special_tokens=True)
            with self.lock: self.current_desc = d
        except: pass

# --- 4. INTELLIGENCE (BRAIN) ---
class LlamaRoaster:
    def __init__(self):
        self.model = OLLAMA_MODEL
        # --- IMPROVED PROMPT FOR CONVERSATIONAL CONTEXT ---
        self.system_prompt = """
        You are a witty, cynical robot assistant engaged in a continuous conversation.
        
        INSTRUCTIONS:
        1. STAY IN CHARACTER: You are superior, sarcastic, and observant.
        2. REPLY TO THE CONTEXT: Use the conversation history to understand what is happening.
        3. ANSWER THE USER: If they ask a question, answer it, but add a roast.
        4. USE VISUALS: Reference the user's appearance naturally in your reply.
        5. BREVITY: Keep responses under 2 sentences.

        OUTPUT FORMAT: 
        roast: "<your reply>" 
        tone: "<mocking|sarcastic|deadpan|playful>"
        expression: "<neutral|happy|angry|suspicious|mocking>"
        """
        
        self.history = [{"role": "system", "content": self.system_prompt}]
        try: ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}])
        except Exception as e: print(f"[!] OLLAMA ERROR: {e}")

    def reset_memory(self):
        print("--- MEMORY WIPED ---")
        self.history = [{"role": "system", "content": self.system_prompt}]
        return "roast: \"Oh, a new human. Try not to bore me like the last one.\"\ntone: \"suspicious\"\nexpression: \"suspicious\""

    def generate(self, appearance, expression, user_speech):
        if "reset" in user_speech.lower() and "memory" in user_speech.lower():
            return self.reset_memory()

        # --- CONTEXT INJECTION ---
        # We frame the input clearly so Llama knows what is vision vs text
        current_input = f"""
        CONTEXT:
        - User Appearance: {appearance}
        - User Emotion: {expression}
        
        USER SAID:
        "{user_speech}"
        """
        
        self.history.append({"role": "user", "content": current_input})

        try:
            response = ollama.chat(
                model=self.model, 
                messages=self.history, 
                options={"num_predict": 50, "temperature": 0.8} # Slightly increased tokens for natural flow
            )
            
            bot_reply = response["message"]["content"]
            self.history.append({"role": "assistant", "content": bot_reply})
            
            # Keep last 6 turns (System + 6 User + 6 Bot)
            if len(self.history) > 13: 
                self.history = [self.history[0]] + self.history[-10:]
            
            return bot_reply
        except: return "roast: \"I lost my train of thought. Probably because of your face.\"\ntone: \"deadpan\"\nexpression: \"dizzy\""

# --- MAIN ---
if __name__ == "__main__":
    vision = VisionSystem()
    roaster = LlamaRoaster()
    tts = KokoroTTS()
    ui_state = {"text": "Listening...", "color": (0, 255, 0)}

    def process_response(raw_res):
        roast_text, tone, robot_face = "", "deadpan", "neutral"
        for line in raw_res.split('\n'):
            line_lower = line.lower()
            if "roast:" in line_lower: roast_text = line.split(':', 1)[1].replace('"','').strip()
            if "tone:" in line_lower: tone = line.split(':', 1)[1].replace('"','').strip()
            if "expression:" in line_lower: robot_face = line.split(':', 1)[1].replace('"','').strip()
        
        if not roast_text: 
            roast_text = raw_res.replace("roast:", "").replace('"', "")
            
        return roast_text, tone, robot_face

    def handle_user_speech(text):
        GLOBAL_STATE["is_processing"] = True
        ui_state["text"] = "Thinking..."
        ui_state["color"] = (0, 255, 255) 
        _, _, emo, desc, _ = vision.get_data()
        
        raw_res = roaster.generate(desc, emo, text)
        roast_text, tone, robot_face = process_response(raw_res)

        face_map = {"happy": ":)", "angry": ">:(", "surprise": ":O", "suspicious": "-_-", "disgust": "X(", "neutral": ":|", "deadpan": ":|", "mocking": ";)"}
        ui_state["text"] = f"{face_map.get(robot_face, '[AI]')} {roast_text}"
        ui_state["color"] = (0, 0, 255) 
        print(f"<< ROBOT: {roast_text} [{tone}]")
        tts.speak(roast_text, tone)
        
        GLOBAL_STATE["is_processing"] = False
        ui_state["text"] = "Listening..."
        ui_state["color"] = (0, 255, 0)

    ear = Ear(callback_function=handle_user_speech)
    vision.start()
    ear.start()
    print("--- SYSTEM LIVE ---")
    
    try:
        while True:
            frame, box, emotion, desc, new_user_alert = vision.get_data()
            if frame is None: 
                time.sleep(0.1)
                continue
            
            if new_user_alert:
                print("--- NEW USER DETECTED ---")
                raw_res = roaster.reset_memory()
                roast_text, tone, robot_face = process_response(raw_res)
                
                GLOBAL_STATE["is_processing"] = True
                tts.speak(roast_text, tone)
                GLOBAL_STATE["is_processing"] = False

            if box:
                bx, by, bw, bh = box
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                cv2.putText(frame, emotion.upper(), (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            ih, iw, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, ih-100), (iw, ih), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            lines = textwrap.wrap(ui_state["text"], width=55)
            for i, line in enumerate(lines):
                y_pos = ih - 70 + (i * 30)
                if y_pos < ih - 10:
                    cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_state["color"], 2)

            color_ind = (0, 255, 0)
            if GLOBAL_STATE["is_speaking"]: color_ind = (0, 0, 255)
            elif GLOBAL_STATE["is_processing"]: color_ind = (0, 255, 255)
            cv2.circle(frame, (iw-30, 30), 15, color_ind, -1)

            cv2.imshow("RoastBot 9000", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
    except KeyboardInterrupt: pass
    finally:
        vision.stop()
        ear.stop()
        cv2.destroyAllWindows()