import os
import io
import time
import base64
import numpy as np
import torch
import wave
import cv2
import soundfile as sf
import tempfile
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from deepface import DeepFace
import mlx_whisper
import ollama
from kokoro_onnx import Kokoro

# --- CONFIGURATION ---
MODEL_PATH = "kokoro-v1.0.onnx"
VOICES_PATH = "voices-v1.0.bin"
STT_MODEL = "mlx-community/whisper-large-v3-turbo-q4"
OLLAMA_MODEL = "llama3.2" 
VAD_THRESHOLD = 0.5

# Shared Memory for Vision
VISION_CONTEXT = {"desc": "A person", "emotion": "neutral"}

# --- 1. VAD SYSTEM ---
print("Loading VAD...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)

def check_vad_speech(audio_bytes):
    float_audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    tensor = torch.from_numpy(float_audio)
    speech_prob = vad_model(tensor, 16000).item()
    return speech_prob > VAD_THRESHOLD

# --- 2. VISION SYSTEM ---
class CloudVision:
    def __init__(self):
        print("Loading Vision Models...")
        self.device = "cpu"
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.last_desc_time = 0
        
        try: DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=['emotion'], enforce_detection=False, silent=True)
        except: pass

    def process_frame(self, frame_bytes):
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            res = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
            VISION_CONTEXT["emotion"] = res[0]['dominant_emotion']
        except: pass

        if time.time() - self.last_desc_time > 5:
            try:
                inputs = self.processor(Image.fromarray(rgb), return_tensors="pt").to(self.device)
                out = self.model.generate(**inputs, max_new_tokens=30)
                VISION_CONTEXT["desc"] = self.processor.decode(out[0], skip_special_tokens=True)
                self.last_desc_time = time.time()
            except: pass

# --- 3. SPEECH GENERATION (KOKORO) ---
class CloudKokoro:
    def __init__(self):
        print("Loading TTS...")
        try:
            self.kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
            
            v1 = self.kokoro.voices["af_bella"]
            v2 = self.kokoro.voices["am_adam"]
            self.sarcastic_mix = (v1 * 0.5) + (v2 * 0.5)
            
            self.tones = {
                "mocking": {"voice": "af_bella", "speed": 1.1},
                "sarcastic": {"voice": self.sarcastic_mix, "speed": 0.8},
                "deadpan": {"voice": "am_adam", "speed": 0.9},
                "playful": {"voice": "af_sarah", "speed": 1.1},
                "annoyed": {"voice": "am_michael", "speed": 0.8}
            }
        except Exception as e:
            print(f"TTS Load Error: {e}")
            print("Ensure kokoro-v1.0.onnx and voices-v1.0.bin are in this folder.")

    def generate_audio_bytes(self, text, tone="deadpan"):
        profile = self.tones.get(tone.lower(), self.tones["deadpan"])
        voice_data = self.sarcastic_mix if profile["voice"] == "mix_sarcasm" else profile["voice"]
        
        audio, sample_rate = self.kokoro.create(text, voice=voice_data, speed=profile["speed"], lang="en-us")
        
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

# --- 4. INTELLIGENCE (LLAMA) ---
class CloudBrain:
    def __init__(self):
        self.model = OLLAMA_MODEL
        
        self.system_prompt = """
        You are a highly intelligent, cynical robot assistant with eyes and ears.
        Your goal is to be conversational and helpful, but with a biting, witty, and observant personality.

        GUIDELINES:
        1. **Be Responsive:** If they ask a question, answer it. If they greet you, greet them back. 
        2. **Use Vision:** Constantly weave their appearance into your replies naturally.
        3. **Expression:** Choose a facial expression that matches your sentiment.
        4. **Brevity:** Keep responses SHORT (1-2 sentences max).

        OUTPUT FORMAT: 
        roast: "<your spoken response>" 
        tone: "<mocking|sarcastic|deadpan|playful|annoyed>"
        expression: "<neutral|happy|angry|surprise|suspicious|disgust>"
        """
        
        self.history = [{"role": "system", "content": self.system_prompt}]
        
        try: 
            ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}])
        except Exception as e: 
            print(f"OLLAMA ERROR: {e}")

    def think(self, user_text):
        if "reset" in user_text.lower() and "memory" in user_text.lower():
            self.history = [{"role": "system", "content": self.system_prompt}]
            return "Memory wiped. Who are you again?", "deadpan", "confused"

        current_input = f"""
        [Current Visual Status]
        - Appearance: {VISION_CONTEXT['desc']}
        - Facial Expression: {VISION_CONTEXT['emotion']}
        
        [User Speech]
        "{user_text}"
        """

        self.history.append({"role": "user", "content": current_input})

        try:
            res = ollama.chat(
                model=self.model, 
                messages=self.history,
                options={"num_predict": 100, "temperature": 0.8}
            )
            reply = res["message"]["content"]
            self.history.append({"role": "assistant", "content": reply})
            
            if len(self.history) > 15:
                self.history = [self.history[0]] + self.history[-10:]
            
            roast, tone, expr = reply, "deadpan", "neutral"
            
            for line in reply.split('\n'):
                line_lower = line.lower()
                if "roast:" in line_lower:
                    roast = line.split(':', 1)[1].replace('"','').strip()
                if "tone:" in line_lower:
                    tone = line.split(':', 1)[1].replace('"','').strip()
                if "expression:" in line_lower:
                    expr = line.split(':', 1)[1].replace('"','').strip()
            
            if roast == reply:
                roast = reply.replace("roast:", "").replace('"', "")

            return roast, tone, expr

        except Exception as e:
            print(f"Brain Error: {e}")
            return "My brain is disconnected.", "deadpan", "dizzy"

def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wf = wave.open(tmp.name, 'wb')
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(audio_bytes)
        wf.close()
        try:
            res = mlx_whisper.transcribe(tmp.name, path_or_hf_repo=STT_MODEL, language="en", verbose=False)
            text = res["text"].strip()
        except Exception as e:
            print(f"Whisper Error: {e}")
            text = ""
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
    return text

vision = CloudVision()
tts = CloudKokoro()
brain = CloudBrain()

print("AI ENGINE READY")