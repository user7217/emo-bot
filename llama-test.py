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
import pyaudio
import warnings
import mlx_whisper
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import ollama
from kokoro_onnx import Kokoro

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --- CONFIGURATION ---
STT_MODEL = "mlx-community/whisper-large-v3-turbo-q4"
OLLAMA_MODEL = "llama3.1:8b"
VAD_THRESHOLD = 0.5
SILENCE_TIMEOUT = 0.8
MAX_RECORD_TIME = 8.0

# --- SHARED STATE ---
GLOBAL_STATE = {
    "is_speaking": False,
    "is_processing": False
}

class KokoroTTS:
    def __init__(self):
        try:
            cwd = os.getcwd()
            # Ensure these files exist in your folder!
            model_path = os.path.join(cwd, "kokoro-v1.0.onnx")
            voices_path = os.path.join(cwd, "voices-v1.0.bin")
            
            self.kokoro = Kokoro(model_path, voices_path)
            
            self.tones = {
                "mocking": {"voice": "af_bella", "speed": 1.1},
                "sarcastic": {"voice": "mix_sarcasm", "speed": 0.8},
                "deadpan": {"voice": "am_adam", "speed": 0.9},
                "playful": {"voice": "af_sarah", "speed": 1.1},
                "annoyed": {"voice": "am_michael", "speed": 0.8}
            }
            # Create a mixed voice for sarcasm
            v1 = self.kokoro.voices["af_bella"]
            v2 = self.kokoro.voices["am_adam"]
            self.sarcastic_mix = (v1 * 0.5) + (v2 * 0.5)

        except Exception as e:
            print(f"TTS Error: {e}")
            print("Ensure kokoro-v1.0.onnx and voices-v1.0.bin are in the folder.")
            exit()

    def speak(self, text, tone_category="deadpan"):
        GLOBAL_STATE["is_speaking"] = True
        
        profile = self.tones.get(tone_category.lower(), self.tones["deadpan"])
        voice = profile["voice"]
        speed = profile["speed"]
        
        voice_data = self.sarcastic_mix if voice == "mix_sarcasm" else voice
            
        try:
            # Generate audio
            audio, sample_rate = self.kokoro.create(text, voice=voice_data, speed=speed, lang="en-us")
            # Play audio (Blocking)
            sd.play(audio, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            # Short pause after speaking so the mic doesn't catch the echo
            time.sleep(0.5)
            GLOBAL_STATE["is_speaking"] = False

class Ear:
    def __init__(self, callback_function):
        self.callback = callback_function
        self.running = False
        self.rate = 16000
        
        print(" Loading VAD & Whisper...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        self.get_speech_timestamps = utils[0]
        print(" Ear Ready.")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _listen_loop(self):
        # Open Mic
        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=512)
        
        frames_buffer = []
        is_recording = False
        last_speech_time = 0
        buffer_start_time = 0

        while self.running:
            # 1. BLINDNESS CHECK: If Robot is speaking, deafen the ears
            if GLOBAL_STATE["is_speaking"] or GLOBAL_STATE["is_processing"]:
                time.sleep(0.1)
                # Clear buffer so we don't process old audio
                frames_buffer = [] 
                is_recording = False
                continue

            # 2. Read Audio
            try:
                data = stream.read(512, exception_on_overflow=False)
            except: continue

            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.flatten().astype(np.float32) / 32768.0

            # 3. VAD Check
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
                silence_dur = current_time - last_speech_time
                total_dur = current_time - buffer_start_time

                if silence_dur > SILENCE_TIMEOUT or total_dur > MAX_RECORD_TIME:
                    is_recording = False
                    # Send to processor
                    self._transcribe(b''.join(frames_buffer), p)
                    frames_buffer = []

    def _transcribe(self, audio_data, pa_instance):
        # Save to temp file for MLX
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
            # MLX Whisper Inference
            res = mlx_whisper.transcribe(tmp_path, path_or_hf_repo=STT_MODEL, language="en", verbose=False)
            text = res["text"].strip()
            if len(text) > 2: # Ignore tiny hallucinations
                print(f"USER SAID: {text}")
                self.callback(text)
        except Exception as e:
            print(f"STT Error: {e}")
        finally:
            os.remove(tmp_path)

class VisionSystem:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.running = False
        self.current_emotion = "neutral"
        self.current_desc = "A person"
        self.latest_frame = None
        self.face_box = None 
        self.lock = threading.Lock()
        
        self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.device = "cpu" # Keep vision on CPU to save GPU/NPU for Whisper/Llama
        
        print("⬇️ Loading Vision Models...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        # Warmup DeepFace
        try: DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=['emotion'], enforce_detection=False, silent=True)
        except: pass
        print("✅ Vision Ready.")
        
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
            return self.latest_frame, self.face_box, self.current_emotion, self.current_desc

    def _capture_loop(self):
        last_emo_t = 0
        last_desc_t = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ih, iw, _ = frame.shape

            results = self.mp_face.process(rgb)
            curr_box = None
            
            if results.detections:
                # Get largest face
                d = max(results.detections, key=lambda x: x.location_data.relative_bounding_box.width)
                bb = d.location_data.relative_bounding_box
                bx, by = int(bb.xmin * iw), int(bb.ymin * ih)
                bw, bh = int(bb.width * iw), int(bb.height * ih)
                curr_box = (max(0,bx), max(0,by), bw, bh)

                # Emotion Check (Every 0.5s)
                if time.time() - last_emo_t > 0.5 and bw>10:
                    try:
                        roi = frame[by:by+bh, bx:bx+bw]
                        res = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False, detector_backend='skip', silent=True)
                        with self.lock: self.current_emotion = res[0]['dominant_emotion']
                        last_emo_t = time.time()
                    except: pass

            # Scene Description (Every 10s)
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

class LlamaRoaster:
    def __init__(self):
        self.model = OLLAMA_MODEL
        # Updated System Prompt for Conversation
        self.SYSTEM_PROMPT = """
        You are a savage, roasting robot assistant. 
        You have eyes and ears.
        1. Listen to what the user said.
        2. Look at their appearance and facial expression.
        3. Combine these to create a short, witty, biting insult.
        
        If they ask a question, answer it sarcastically.
        If they say nothing interesting, roast their outfit or face.
        
        OUTPUT FORMAT: 
        roast: "your text here" 
        tone: "mocking|sarcastic|deadpan|annoyed"
        """
        try: ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}])
        except: print(" Warning: Ensure Ollama is running.")

    def generate(self, appearance, expression, user_speech):
        prompt = f"""
        User Status:
        - Visuals: {appearance}
        - Emotion: {expression}
        - User Said: "{user_speech}"
        
        Generate a roast now.
        """
        try:
            response = ollama.chat(model=self.model, messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT}, 
                {"role": "user", "content": prompt}
            ], options={"num_predict": 100, "temperature": 0.8})
            return response["message"]["content"]
        except: return "roast: \"I'm malfunctioning, unlike your bad fashion sense.\" tone: \"deadpan\""

# --- MAIN CONTROLLER ---
if __name__ == "__main__":
    vision = VisionSystem()
    roaster = LlamaRoaster()
    tts = KokoroTTS()
    
    # State for UI
    ui_state = {"text": "Listening...", "color": (0, 255, 0)}

    def handle_user_speech(text):
        """Callback when Ear hears something"""
        GLOBAL_STATE["is_processing"] = True
        ui_state["text"] = "Processing Roast..."
        ui_state["color"] = (0, 255, 255) # Yellow
        
        # 1. Get Visual Context
        _, _, emo, desc = vision.get_data()
        
        # 2. Generate Roast
        raw_res = roaster.generate(desc, emo, text)
        
        # 3. Parse Response
        roast_text = ""
        tone = "deadpan"
        for line in raw_res.split('\n'):
            if "roast:" in line.lower():
                roast_text = line.split(':', 1)[1].replace('"','').strip()
            if "tone:" in line.lower():
                tone = line.split(':', 1)[1].replace('"','').strip()
        
        if not roast_text: roast_text = raw_res # Fallback

        # 4. Speak
        ui_state["text"] = f"ROBOT: {roast_text}"
        ui_state["color"] = (0, 0, 255) # Red
        print(f"🤖 {roast_text} [{tone}]")
        
        tts.speak(roast_text, tone)
        
        # Reset State
        GLOBAL_STATE["is_processing"] = False
        ui_state["text"] = "Listening..."
        ui_state["color"] = (0, 255, 0)

    # Start Systems
    ear = Ear(callback_function=handle_user_speech)
    vision.start()
    ear.start()

    print("--- SYSTEM LIVE ---")
    
    try:
        while True:
            frame, box, emotion, desc = vision.get_data()
            if frame is None: 
                time.sleep(0.1)
                continue

            # Draw Face Box
            if box:
                bx, by, bw, bh = box
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                cv2.putText(frame, emotion.upper(), (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # Draw UI Overlay
            ih, iw, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, ih-100), (iw, ih), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            # Display Text (Wrapped)
            lines = textwrap.wrap(ui_state["text"], width=55)
            for i, line in enumerate(lines):
                y_pos = ih - 70 + (i * 30)
                if y_pos < ih - 10:
                    cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_state["color"], 2)

            # Status Indicators
            if GLOBAL_STATE["is_speaking"]:
                cv2.circle(frame, (iw-30, 30), 15, (0, 0, 255), -1) # Red Dot = Talking
            elif GLOBAL_STATE["is_processing"]:
                cv2.circle(frame, (iw-30, 30), 15, (0, 255, 255), -1) # Yellow Dot = Thinking
            else:
                cv2.circle(frame, (iw-30, 30), 15, (0, 255, 0), -1) # Green Dot = Listening

            cv2.imshow("RoastBot 9000", frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        ear.stop()
        cv2.destroyAllWindows()