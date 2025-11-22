import asyncio
import websockets
import json

CLIENTS = set()

async def handler(websocket):
    """Register new connection and keep it open."""
    print(f"Device connected!")
    CLIENTS.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.remove(websocket)
        print("Device disconnected.")

async def input_loop():
    """Read keyboard input and send to ESP32."""
    print("Server Ready. Type message or /angry, /tensed, /sad")
    while True:
        text = await asyncio.to_thread(input)
        
        if not CLIENTS:
            print("No device connected.")
            continue

        # Determine if command or text
        if text.startswith("/"):
            data = {"emotion": text.replace("/", "")}
        else:
            data = {"text": text, "emotion": "angry"}
            
        # Broadcast to ESP32
        payload = json.dumps(data)
        for ws in CLIENTS:
            await ws.send(payload)

async def main():
    # 0.0.0.0 listens on ALL IP addresses your computer has
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await input_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")