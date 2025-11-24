import asyncio
import uvicorn
import time
import json
import base64
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# --- IMPORT AI MODULES ---
import ai_engine as ai

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 Client Connected")
    
    audio_buffer = bytearray()
    silence_start = time.time()
    is_speaking = False
    
    try:
        while True:
            message = await websocket.receive()
            
            # --- AUDIO FLOW ---
            if "bytes" in message:
                data = message["bytes"]
                audio_buffer.extend(data)
                
                # Check VAD using helper function
                if ai.check_vad_speech(data):
                    is_speaking = True
                    silence_start = time.time()
                
                # Speech ended logic
                elif is_speaking and (time.time() - silence_start > 0.6):
                    print("🗣️ Speech Ended, Processing...")
                    is_speaking = False
                    
                    # 1. Transcribe
                    text = ai.transcribe_audio(audio_buffer)
                    audio_buffer = bytearray() # Clear buffer
                    
                    if len(text) > 2:
                        print(f"User: {text}")
                        # 2. Think (Llama)
                        roast, tone, expr = ai.brain.think(text)
                        # 3. Speak (Kokoro)
                        wav_b64 = ai.tts.generate_audio_bytes(roast, tone)
                        
                        print(f"Robot: {roast}")
                        
                        # 4. Reply
                        await websocket.send_json({
                            "type": "response",
                            "text": roast,
                            "audio": wav_b64,
                            "expression": expr
                        })
            
            # --- VIDEO FLOW ---
            elif "text" in message:
                try:
                    payload = json.loads(message["text"])
                    if "image" in payload:
                        img_data = base64.b64decode(payload["image"].split(',')[1])
                        # Fire and forget (don't wait for result)
                        ai.vision.process_frame(img_data)
                except: pass

    except Exception as e:
        print(f"Client Disconnected: {e}")

# --- CLIENT UI ---
@app.get("/", response_class=HTMLResponse)
async def get():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RoastBot Mobile</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background: #000; color: #0f0; font-family: monospace; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
            #preview { width: 90%; max-width: 400px; border: 2px solid #333; border-radius: 10px; transform: scaleX(-1); }
            #face { font-size: 80px; margin: 20px; }
            #text { font-size: 20px; color: #fff; padding: 20px; min-height: 60px; }
            button { font-size: 20px; padding: 15px 30px; background: #0f0; border: none; font-weight: bold; cursor: pointer; }
        </style>
    </head>
    <body>
        <div id="face">🤖</div>
        <div id="text">Tap Start to Connect</div>
        <video id="preview" autoplay playsinline muted></video>
        <canvas id="hidden-canvas" style="display:none;"></canvas>
        <button onclick="init()">START ROBOT</button>

        <script>
            let ws;
            let audioCtx;
            let processor;
            
            async function init() {
                document.querySelector('button').style.display = 'none';
                document.getElementById('text').innerText = "Connecting...";

                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 320, facingMode: "user" }, 
                    audio: { echoCancellation: true, noiseSuppression: true } 
                });
                
                document.getElementById('preview').srcObject = stream;
                
                audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const source = audioCtx.createMediaStreamSource(stream);
                processor = audioCtx.createScriptProcessor(4096, 1, 1);
                source.connect(processor);
                processor.connect(audioCtx.destination);

                ws = new WebSocket("ws://" + window.location.host + "/ws");
                
                ws.onopen = () => {
                    document.getElementById('text').innerText = "Listening...";
                    const canvas = document.getElementById('hidden-canvas');
                    const ctx = canvas.getContext('2d');
                    
                    setInterval(() => {
                        canvas.width = 320; canvas.height = 240;
                        ctx.drawImage(document.getElementById('preview'), 0, 0, 320, 240);
                        ws.send(JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.5) }));
                    }, 500);
                };

                ws.onmessage = async (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === "response") {
                        document.getElementById('text').innerText = data.text;
                        const map = {"happy":"😊", "angry":"😠", "surprise":"😲", "suspicious":"🤨", "mocking":"😏"};
                        document.getElementById('face').innerText = map[data.expression] || "🤖";
                        
                        const audioData = atob(data.audio);
                        const arrayBuffer = new ArrayBuffer(audioData.length);
                        const view = new Uint8Array(arrayBuffer);
                        for (let i=0; i<audioData.length; i++) view[i] = audioData.charCodeAt(i);
                        const decodedAudio = await audioCtx.decodeAudioData(arrayBuffer);
                        const playSource = audioCtx.createBufferSource();
                        playSource.buffer = decodedAudio;
                        playSource.connect(audioCtx.destination);
                        playSource.start(0);
                    }
                };

                processor.onaudioprocess = (e) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        const input = e.inputBuffer.getChannelData(0);
                        const pcm = new Int16Array(input.length);
                        for (let i = 0; i < input.length; i++) pcm[i] = Math.max(-1, Math.min(1, input[i])) * 0x7FFF;
                        ws.send(pcm.buffer);
                    }
                };
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)