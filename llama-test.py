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

STT_MODEL = "mlx-community/distil-whisper-large-v3"
OLLAMA_MODEL = "llama3.1:8b" 

# Audio Settings
VAD_THRESHOLD = 0.5
SILENCE_TIMEOUT = 0.8       
MAX_RECORD_TIME = 10.0

# Thread Synchronization
interrupt_event = threading.Event()
audio_queue = queue.Queue()

# --- 1. ASYNC AUDIO PLAYER (FIXED) ---
class AsyncAudioPlayer:
    def __init__(self):
        self.running = True
        self.playing = False # Track playing state manually to avoid sd.get_stream error
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def _play_loop(self):
        while self.running:
            try:
                # Wait for audio chunk
                audio_data, sample_rate = audio_queue.get(timeout=0.1)
                
                # Check for interrupt before playing
                if interrupt_event.is_set():
                    self._clear_queue()
                    continue

                self.playing = True
                sd.play(audio_data, sample_rate)
                sd.wait() # Wait for this specific chunk to finish
                self.playing = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[!] Player Error: {e}")
                self.playing = False

    def stop_current_audio(self):
        try:
            sd.stop()
        except: pass 
        self._clear_queue()
        self.playing = False

    def _clear_queue(self):
        while not audio_queue.empty():
            try: audio_queue.get_nowait()
            except queue.Empty: break

# --- 2. STREAMING TTS ---
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

    # Inside KokoroTTS class
    def synthesize_and_queue(self, text, tone_category="deadpan"):
        if interrupt_event.is_set(): return

        profile = self.tones.get(tone_category.lower(), self.tones["deadpan"])
        voice_data = self.sarcastic_mix if profile["voice"] == "mix_sarcasm" else self.kokoro.voices[profile["voice"]]
        
        try:
            audio, sample_rate = self.kokoro.create(text, voice=voice_data, speed=profile["speed"], lang="en-us")
            
        
            audio = audio * 0.6 
            
            if not interrupt_event.is_set():
                audio_queue.put((audio, sample_rate))
        except Exception as e:
            print(f"[!] Synth Error: {e}")

# --- 3. THE EAR (UPDATED FOR BARGE-IN) ---
# --- 3. THE EAR (UPDATED: NO HEADPHONES + FIXED TRANSCRIBE) ---
class Ear:
    def __init__(self, callback_function, audio_player_ref):
        self.callback = callback_function
        self.player = audio_player_ref
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

        # DYNAMIC THRESHOLD SETTINGS
        # 0.5 = Sensitive (Normal listening)
        # 0.85 = Deafened (Ignore robot voice, listen for yelling)
        NORMAL_THRESHOLD = 0.5
        BARGE_IN_THRESHOLD = 0.96

        while self.running:
            try:
                data = stream.read(512, exception_on_overflow=False)
            except: continue

            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.flatten().astype(np.float32) / 32768.0
            vad_tensor = torch.from_numpy(audio_float32)
            
            # SWITCH SENSITIVITY IF ROBOT IS TALKING
            current_threshold = BARGE_IN_THRESHOLD if self.player.playing else NORMAL_THRESHOLD
            
            speech_prob = self.vad_model(vad_tensor, self.rate).item()
            is_speech = speech_prob > current_threshold
            
            current_time = time.time()

            # --- BARGE-IN LOGIC ---
            if is_speech:
                if not audio_queue.empty() or self.player.playing: 
                    print("\n[!] INTERRUPT DETECTED (Loud Input)")
                    interrupt_event.set()
                    self.player.stop_current_audio()
                    frames_buffer = [] 
                    is_recording = True 
                    buffer_start_time = current_time

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
                    if len(frames_buffer) > 10: 
                        self._transcribe(b''.join(frames_buffer), p)
                    frames_buffer = []

    def _transcribe(self, audio_data, pa_instance):
        interrupt_event.clear() # Clear interrupt flag so new thought can be processed

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
            
            garbage = ["thank you", "subtitles", "you", "copyright", "audio", "bye"]
            if len(text) > 2 and text.lower() not in garbage: 
                print(f">> USER: {text}")
                self.callback(text)
        except Exception as e:
            print(f"[!] STT Error: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- 4. STREAMING BRAIN ---
class LlamaRoaster:
    def __init__(self):
        self.model = OLLAMA_MODEL
        self.system_prompt = """
        You are a witty, cynical robot assistant.
        INSTRUCTIONS:
        1. Keep responses under 2 sentences.
        2. Roast the user based on context.
        3. OUTPUT FORMAT: Just the text of the roast. Do not include "roast:" or "tone:" labels.
        """
        self.history = [{"role": "system", "content": self.system_prompt}]

    def generate_stream(self, appearance, expression, user_speech):
        current_input = f"CONTEXT: User Look: {appearance}, User Emotion: {expression}\nUSER SAID: {user_speech}"
        self.history.append({"role": "user", "content": current_input})

        stream = ollama.chat(
            model=self.model, 
            messages=self.history, 
            stream=True, 
            options={"num_predict": 70, "temperature": 0.8}
        )

        full_response = ""
        buffer = ""
        
        for chunk in stream:
            if interrupt_event.is_set():
                print("--- LLM ABORTED ---")
                break
            
            content = chunk['message']['content']
            buffer += content
            full_response += content

            # Split sentences on . ! ?
            if re.search(r'[.!?](\s|$)', buffer):
                sentences = re.split(r'(?<=[.!?])\s+', buffer)
                for s in sentences[:-1]:
                    yield s.strip()
                buffer = sentences[-1]

        if buffer.strip() and not interrupt_event.is_set():
            yield buffer.strip()

        self.history.append({"role": "assistant", "content": full_response})
        if len(self.history) > 10: self.history = [self.history[0]] + self.history[-8:]

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

# --- MAIN ORCHESTRATOR ---
if __name__ == "__main__":
    vision = VisionSystem()
    roaster = LlamaRoaster()
    tts = KokoroTTS()
    player = AsyncAudioPlayer()
    
    ui_state = {"text": "Listening...", "color": (0, 255, 0)}

    def processing_thread(user_text):
        ui_state["text"] = "Thinking..."
        ui_state["color"] = (0, 255, 255) 

        _, _, emo, desc = vision.get_data()
        
        tone = "deadpan"
        if emo in ["happy", "surprise"]: tone = "mocking"
        if emo in ["angry", "sad"]: tone = "sarcastic"

        print(f"<< ROBOT THINKING (Context: {emo})...")
        
        try:
            for sentence in roaster.generate_stream(desc, emo, user_text):
                if interrupt_event.is_set():
                    break 
                
                print(f"<< SENTENCE: {sentence}")
                ui_state["text"] = sentence
                ui_state["color"] = (0, 0, 255)
                
                tts.synthesize_and_queue(sentence, tone)
        except Exception as e:
            print(f"Pipeline Error: {e}")

        ui_state["text"] = "Listening..."
        ui_state["color"] = (0, 255, 0)

    def on_user_speech(text):
        t = threading.Thread(target=processing_thread, args=(text,))
        t.start()

    ear = Ear(callback_function=on_user_speech, audio_player_ref=player)
    
    vision.start()
    ear.start()
    
    print("--- SYSTEM LIVE (GEMINI MODE) ---")
    print("--- WEAR HEADPHONES TO AVOID SELF-INTERRUPTION ---")

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
            # FIXED: Checking player.playing flag instead of stream active
            if not audio_queue.empty() or player.playing:
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
        cv2.destroyAllWindows()