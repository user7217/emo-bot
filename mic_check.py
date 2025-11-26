import pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()

print("\n--- AVAILABLE MICROPHONES ---")
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
default_device_index = 0

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        name = p.get_device_info_by_host_api_device_index(0, i).get('name')
        print(f"Index {i}: {name}")
        # Heuristic to find default
        if "Default" in name or "Built-in" in name:
            default_device_index = i

print(f"\nUsing Device Index: {default_device_index} (Change this in your code if wrong!)")
print("-----------------------------\n")

stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                input_device_index=default_device_index, frames_per_buffer=1024)

print("🎤 SPEAK NOW! (Press Ctrl+C to stop)")

try:
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Calculate Volume (RMS)
        volume = np.linalg.norm(audio_data) / 10
        
        # Visual Bar
        bars = "#" * int(volume / 50)
        print(f"Volume: {int(volume)} | {bars}")
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()