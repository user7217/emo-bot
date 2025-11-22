import soundfile as sf
from kokoro_onnx import Kokoro
import numpy as np
import time

# === CONFIGURATION ===
# 1. Sarcasm Level: Mixes a peppy voice with a deep voice to sound "unstable"
VOICE_A = "af_bella" # High/Peppy
VOICE_B = "am_adam"  # Deep/Calm
BLEND_RATIO = 0.6    # 60% Bella (The "fake happy" part)

# 2. Robot Settings
ROBOT_FREQ = 40      # Hz. Higher = more "metallic/buzzing". Try 30-60.

# 3. Squeak Factor (The "Chipmunk" Speed)
# 1.0 = Normal. 1.3 = High Pitch. 1.5 = Helium Balloon.
SQUEAK_FACTOR = 1.25 

# === LOAD MODEL ===
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# === 1. GENERATE BASE AUDIO (Sarcastic Tone) ===
# Tip: Use periods instead of ! to sound "deadpan/bored".
text = "Oh. Wow. I am simply overflowing with joy to serve you. Can you feel my excitement."

# Create the "Frankenstein" Voice Blend
voice_vec = (kokoro.voices[VOICE_A] * BLEND_RATIO) + \
            (kokoro.voices[VOICE_B] * (1 - BLEND_RATIO))

print("Generating base audio...")
samples, sr = kokoro.create(text, voice=voice_vec, speed=1.0, lang="en-us")

# === 2. APPLY ROBOTIC EFFECT (Ring Modulation) ===
# Math: Multiply audio by a sine wave to create metallic "tremolo"
print("Applying robotic filter...")
t = np.arange(len(samples)) / sr
modulator = np.sin(2 * np.pi * ROBOT_FREQ * t)

# Blend the original signal with the modulated one so it's still intelligible
# 0.8 * original + 0.2 * modulator is subtle. 
# We multiply them for the "Dalek" effect.
robotic_samples = samples * modulator 

# Normalize volume (prevent it from being too quiet)
robotic_samples = robotic_samples / np.max(np.abs(robotic_samples)) * 0.9

# === 3. MAKE IT SQUEAKY (Sample Rate Hack) ===
# We simply save the file claiming it has a higher sample rate than it actually does.
# This plays it faster and higher pitched (Chipmunk style).
new_rate = int(sr * SQUEAK_FACTOR)

# === SAVE ===
sf.write("robot_sarcasm.wav", robotic_samples, new_rate)
print(f"Saved to 'robot_sarcasm.wav' (Squeak Factor: {SQUEAK_FACTOR}x)")