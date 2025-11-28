import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time
import threading
import torch
import queue
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
import re
import tempfile

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 1. ACCURACY UPGRADE: Using Turbo model (Best balance for Mac)
STT_MODEL = "mlx-community/whisper-large-v3-turbo"
OLLAMA_MODEL = "llama3.1:8b" 

# Audio Tuning
VAD_THRESHOLD = 0.5        # Back to standard sensitivity
SILENCE_TIMEOUT = 1.2      # Wait longer before cutting off (Catch pauses)
MAX_RECORD_TIME = 15.0
SPEECH_PAD_MS = 500        # Record 0.5s AFTER silence to catch last words

# --- QUEUES ---
text_queue = queue.Queue()
audio_queue = queue.Queue()

# --- 1. ASYNC AUDIO PLAYER ---
class AsyncAudioPlayer:
    def __init__(self):
        self.running = True
        self.playing = False 
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def _play_loop(self):
        while self.running:
            try:
                audio_data, sample_rate = audio_queue.get(timeout=0.1)
                self.playing = True
                sd.play(audio_data, sample_rate)
                sd.wait() 
                self.playing = False
                audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[!] Player Error: {e}")
                self.playing = False

# --- 2. TTS WORKER ---
class TTSWorker:
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
            }
            
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"[!] TTS Init Error: {e}")
            exit()

    def _worker_loop(self):
        while self.running:
            try:
                text, tone_cat = text_queue.get(timeout=0.1)
                profile = self.tones.get(tone_cat.lower(), self.tones["deadpan"])
                voice_data = self.sarcastic_mix if profile["voice"] == "mix_sarcasm" else self.kokoro.voices[profile["voice"]]
                audio, sample_rate = self.kokoro.create(text, voice=voice_data, speed=profile["speed"], lang="en-us")
                audio_queue.put((audio, sample_rate))
                text_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[!] TTS Worker Error: {e}")

# --- 3. THE EAR (HIGH FIDELITY) ---
class Ear:
    def __init__(self, callback_function, audio_player_ref):
        self.callback = callback_function
        self.player = audio_player_ref
        self.running = False
        self.rate = 16000
        print("[+] Loading VAD & Whisper Turbo...")
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
        
        # Buffer to keep pre-speech audio (helps catch the first word)
        pre_buffer = queue.deque(maxlen=10) # ~300ms history

        while self.running:
            # Hard mute when bot is active
            if self.player.playing or not audio_queue.empty() or not text_queue.empty():
                if is_recording: is_recording = False
                frames_buffer = []
                time.sleep(0.05)
                try: stream.read(512, exception_on_overflow=False)
                except: pass
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
                    # Dump pre-buffer into recording so we don't cut off the start
                    frames_buffer = list(pre_buffer) 
                    frames_buffer.append(data)
                else:
                    frames_buffer.append(data)
                last_speech_time = current_time
            
            elif is_recording:
                frames_buffer.append(data)
                # If silence exceeds timeout, process audio
                if (current_time - last_speech_time > SILENCE_TIMEOUT):
                    is_recording = False
                    
                    if len(frames_buffer) > 20: # Minimum ~0.6s duration
                        self._transcribe(b''.join(frames_buffer), p)
                    else:
                        print("[x] Ignored short noise")
                    frames_buffer = []
            
            else:
                # Keep a rolling buffer of "silence" to catch the start of words
                pre_buffer.append(data)

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
            # Using Turbo model for better accuracy
            res = mlx_whisper.transcribe(tmp_path, path_or_hf_repo=STT_MODEL, language="en", verbose=False)
            text = res["text"].strip()
            
            garbage = ["thank you", "subtitles", "you", "copyright", "audio", "bye", "watching", "listening"]
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            
            if len(text) > 3 and clean_text not in garbage: 
                print(f">> USER: {text}")
                self.callback(text)
            else:
                print(f"[x] Garbage Filtered: {text}")
        except Exception as e:
            print(f"[!] STT Error: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- 4. BRAIN (CONTEXT AWARE) ---
class LlamaRoaster:
    def __init__(self):
        self.model = OLLAMA_MODEL
        # 2. PROMPT FIX: Forces the bot to actually REPLY to the topic
        self.system_prompt = """
        You are a witty, sarcastic AI assistant.
        
        Step 1: Understand what the user actually said. 
        Step 2: If they asked a question, answer it, but insult them for not knowing the answer.
        Step 3: If they made a statement, mock their opinion or their appearance.
        
        RULES:
        - REFERENCE THE USER'S INPUT DIRECTLY. Do not ignore it.
        - Keep responses under 2 sentences.
        - Be savage but coherent.
        """
        self.history = [{"role": "system", "content": self.system_prompt}]

    def generate_stream(self, appearance, expression, user_speech):
        # We explicitly label the input so Llama knows what to focus on
        current_input = f"""
        VISUAL CONTEXT: User looks {appearance} and seems {expression}.
        USER SAID: "{user_speech}"
        
        (Reply to what the user said, then roast them based on the visuals)
        """
        self.history.append({"role": "user", "content": current_input})

        stream = ollama.chat(model=self.model, messages=self.history, stream=True, options={"num_predict": 75})
        full_response = ""
        buffer = ""
        
        for chunk in stream:
            content = chunk['message']['content']
            buffer += content
            full_response += content

            if re.search(r'[.!?](\s|$)', buffer):
                sentences = re.split(r'(?<=[.!?])\s+', buffer)
                for s in sentences[:-1]:
                    if len(s.strip()) > 2:
                        yield s.strip()
                buffer = sentences[-1]

        if len(buffer.strip()) > 2:
            yield buffer.strip()
        
        self.history.append({"role": "assistant", "content": full_response})
        # Keep history short to prevent context rot
        if len(self.history) > 8: self.history = [self.history[0]] + self.history[-6:]

# --- 5. VISION SYSTEM ---
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
        
        print("[+] Loading Vision Models...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")
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
            return self.latest_frame, self.face_box, self.current_emotion, self.current_desc

    def _capture_loop(self):
        last_emo_t = 0
        last_desc_t = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face.process(rgb)
            curr_box = None
            if results.detections:
                d = max(results.detections, key=lambda x: x.location_data.relative_bounding_box.width)
                bb = d.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bx, by = int(bb.xmin * iw), int(bb.ymin * ih)
                bw, bh = int(bb.width * iw), int(bb.height * ih)
                curr_box = (max(0,bx), max(0,by), bw, bh)
                face_roi = frame[by:by+bh, bx:bx+bw]
                if time.time() - last_emo_t > 0.5 and face_roi.size > 0:
                    try:
                        res = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='skip', silent=True)
                        with self.lock: self.current_emotion = res[0]['dominant_emotion']
                        last_emo_t = time.time()
                    except: pass
            if time.time() - last_desc_t > 10:
                threading.Thread(target=self._run_blip, args=(rgb.copy(),)).start()
                last_desc_t = time.time()
            with self.lock:
                self.latest_frame = frame.copy()
                self.face_box = curr_box
    
    def _run_blip(self, img):
        try:
            inputs = self.processor(Image.fromarray(img), return_tensors="pt").to("cpu")
            out = self.model.generate(**inputs, max_new_tokens=20)
            d = self.processor.decode(out[0], skip_special_tokens=True)
            with self.lock: self.current_desc = d
        except: pass

# --- MAIN ---
if __name__ == "__main__":
    vision = VisionSystem()
    roaster = LlamaRoaster()
    tts_worker = TTSWorker()
    player = AsyncAudioPlayer()
    
    ui_state = {"text": "Listening...", "color": (0, 255, 0)}

    def processing_thread(user_text):
        ui_state["text"] = "Thinking..."
        ui_state["color"] = (0, 255, 255) 
        _, _, emo, desc = vision.get_data()
        
        tone = "deadpan"
        if emo in ["happy", "surprise"]: tone = "mocking"
        if emo in ["angry", "sad"]: tone = "sarcastic"
        
        print(f"[DEBUG] Brain: Input='{user_text}' | Vis='{desc}'")
        
        try:
            for sentence in roaster.generate_stream(desc, emo, user_text):
                print(f"  [DEBUG] LLM: {sentence}")
                ui_state["text"] = sentence
                ui_state["color"] = (0, 0, 255)
                text_queue.put((sentence, tone))
        except Exception as e:
            print(f"Pipeline Error: {e}")

        ui_state["text"] = "Listening..."
        ui_state["color"] = (0, 255, 0)

    def on_user_speech(text):
        if text_queue.empty() and audio_queue.empty():
            t = threading.Thread(target=processing_thread, args=(text,))
            t.start()
        else:
            print("[x] Ignored input (System Busy)")

    ear = Ear(callback_function=on_user_speech, audio_player_ref=player)
    
    vision.start()
    ear.start()
    
    print("--- SYSTEM LIVE (ACCURATE + CONTEXT AWARE) ---")

    try:
        while True:
            frame, box, emotion, desc = vision.get_data()
            if frame is None: 
                time.sleep(0.1)
                continue
            
            if box:
                bx, by, bw, bh = box
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                cv2.putText(frame, emotion.upper(), (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            ih, iw, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, ih-80), (iw, ih), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            wrapper = textwrap.TextWrapper(width=60) 
            text_lines = wrapper.wrap(ui_state["text"])
            if text_lines:
                cv2.putText(frame, text_lines[0], (20, ih-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui_state["color"], 2)

            color_ind = (0, 255, 0) 
            if not audio_queue.empty() or not text_queue.empty() or player.playing:
                color_ind = (0, 0, 255) 
            elif ui_state["color"] == (0, 255, 255):
                color_ind = (0, 255, 255) 
                
            cv2.circle(frame, (iw-30, 30), 15, color_ind, -1)
            cv2.imshow("RoastBot Live", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
    except KeyboardInterrupt: pass
    finally:
        vision.stop()
        ear.stop()
        player.running = False
        tts_worker.running = False
        cv2.destroyAllWindows()