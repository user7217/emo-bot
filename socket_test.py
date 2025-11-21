import socket
import threading
import time

# ESP32 SoftAP Defaults
ESP_IP = "192.168.4.1"
PORT = 8080

def listen_to_esp(sock):
    """Background thread to print messages coming FROM the ESP32."""
    try:
        while True:
            data = sock.recv(1024)
            if not data:
                break
            print(f"\n[ESP32]: {data.decode().strip()}")
            print("Command > ", end="", flush=True) # Re-print prompt
    except Exception as e:
        print(f"\nConnection lost: {e}")

def main():
    print(f"Connecting to ESP32 at {ESP_IP}:{PORT}...")
    print("Make sure your WiFi is connected to 'ESP32_FaceBot'")

    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((ESP_IP, PORT))
        print("Connected! Type 'sad', 'tensed', or 'angry'.")
        
        # Start listener thread
        t = threading.Thread(target=listen_to_esp, args=(client,), daemon=True)
        t.start()

        while True:
            msg = input("Command > ").lower().strip()
            if msg == "quit":
                break
            if msg:
                # Send command with newline
                try:
                    client.sendall((msg + "\n").encode())
                except OSError:
                    print("Socket error, attempting reconnect...")
                    break
                    
    except Exception as e:
        print(f"Could not connect: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()