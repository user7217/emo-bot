import cv2
import torch
import time  # Import time library
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# --- 1. SETUP FLORENCE-2 (Same as before) ---
print("Loading Florence-2 model...")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

model_id = "microsoft/Florence-2-base"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def analyze_frame(cv2_frame):
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    prompt = "<CAPTION>" 

    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=prompt, 
        image_size=(pil_image.width, pil_image.height)
    )
    return parsed_answer['<CAPTION>']

# --- 2. OPENCV LOOP WITH TIMER ---
cap = cv2.VideoCapture(0)

# Initialize timer variables
last_analysis_time = time.time()
analysis_interval = 5  # 5 seconds

print(f"\n--- RUNNING: Auto-capturing every {analysis_interval} seconds ---")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Display the live feed continuously
    # Add a timer text to the screen so you know when it's coming
    time_left = analysis_interval - (time.time() - last_analysis_time)
    cv2.putText(frame, f"Next Scan: {max(0, time_left):.1f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Webcam Feed', frame)

    # 2. Check if 5 seconds have passed
    if time.time() - last_analysis_time > analysis_interval:
        print("\n[5s Timer Triggered] Analyzing frame...")
        
        # Run analysis (Note: The video feed will freeze briefly here while it thinks)
        start_process = time.time()
        description = analyze_frame(frame)
        end_process = time.time()
        
        print(f"--> RESULT: {description}")
        print(f"--> Took {end_process - start_process:.2f} seconds to process.")
        
        # Reset the timer
        last_analysis_time = time.time()

    # Quit logic
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()