from fastapi import FastAPI, HTTPException
import requests
import uvicorn

app = FastAPI()

UPLOAD_PC_TAILSCALE_IP = "thick-streets-double.loca.lt"

# 2. CHANGE "http://" to "https://" and REMOVE ":8000" entirely
UPLOAD_API_URL = f"https://{UPLOAD_PC_TAILSCALE_IP}/upload"

# This decorator tells FastAPI to run this function automatically as soon as the server boots up
@app.on_event("startup")
def send_video_automatically():
    print("--- SERVER STARTED: Attempting to send video automatically... ---")
    video_path = "./client_output_smooth.mp4" 
    
    try:
        with open(video_path, "rb") as video_file:
            files = {"file": (video_path, video_file, "video/mp4")}
            response = requests.post(UPLOAD_API_URL, files=files)
            
        if response.status_code == 200:
            print(f"SUCCESS: Video sent successfully! Server Response: {response.json()}")
        else:
            print(f"FAILED: Status {response.status_code}. Details: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to connect to Upload PC: {str(e)}")

# You can still keep the route here if you want to call it manually later
@app.post("/send-video-to-other-pc")
def send_video():
    # (Your original code remains here)
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)