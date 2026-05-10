from pydantic import BaseModel
from google import genai
import os
from httpcore import Response

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
import os
import io
import base64

def new_func():
    app = FastAPI(title="PCB Doctor AI")
    return app

app = new_func()
templates = Jinja2Templates(directory="templates")

try:
    gemini_client = genai.Client()
except Exception as e:
    print("WARNING: Could not initialize Gemini. Make sure GEMINI_API_KEY is set.")
    gemini_client = None

# Define the structure for incoming chat messages
class ChatMessage(BaseModel):
    user_input: str
    context: str = "" # We can use this later to pass YOLO detection results!
class DoctorRequest(BaseModel):
    defect_type: str
    board_type: str
    component_code: str = ""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

# Load Models (Make sure names match your sidebar!)
model_assembled = YOLO("qcba_inspector.pt")
model_bare = YOLO("best 1.pt") # Updated to your confirmed working model
model_components = YOLO("model_inspector.pt")

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")

@app.get("/scanner", response_class=HTMLResponse)
async def scanner(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/documentation", response_class=HTMLResponse)
async def read_docs(request: Request):
    return templates.TemplateResponse(request=request, name="documentation.html")

@app.get("/camera", response_class=HTMLResponse)
async def camera_page(request: Request):
    return templates.TemplateResponse(request=request, name="camera.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Returns a blank response so the browser stops throwing a 404
    return Response(content="", media_type="image/x-icon")

# --- AI LOGIC ---

# Endpoint that accepts a base64 frame snapshot from the browser (Webcam)
@app.post("/analyze-frame")
async def analyze_frame(
    frame_data: str = Form(...),        
    board_type: str = Form(default="assembled")
):
    print(f"\n--- CAMERA FRAME SCAN ---")
    print(f"DEBUG: Board type: '{board_type}'")

    try:
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "Could not decode frame"}

    except Exception as e:
        return {"status": "error", "message": f"Frame decode failed: {str(e)}"}

    # 🚨 FIX APPLIED: Removed cvtColor. YOLO automatically handles the BGR array.
    if board_type == "bare":
        # Pass 'img' directly, just like the test script
        results = model_bare.predict(source=img, imgsz=1024, conf=0.25)
        class_names = model_bare.names
    else:
        results = model_assembled.predict(source=img, imgsz=640, conf=0.25)
        class_names = model_assembled.names

    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > 0.25:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                defect_name = class_names[class_id]
                detections.append({
                    "defect_type": defect_name,
                    "confidence": round(conf, 2),
                    "bounding_box": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2)
                    }
                })

    print(f"DEBUG: Frame scan complete. Found {len(detections)} anomalies.")
    return {
        "status": "success",
        "board_analyzed": board_type,
        "defects_found": len(detections),
        "data": detections
    }


# Endpoint for File Uploads
@app.post("/analyze-pcba")
async def analyze_pcba(file: UploadFile = File(...), board_type: str = Form(default="assembled")):
    print(f"\n--- NEW FILE SCAN STARTED ---")
    print(f"DEBUG: Requested brain: '{board_type}'")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 🚨 FIX APPLIED: Removed cvtColor. 
    cv2.imwrite("debug_before_ai.jpg", img) # Still saving for debugging just in case

    if board_type == "bare":
        # Pass 'img' directly
        results = model_bare.predict(source=img, imgsz=1024, conf=0.25)
        class_names = model_bare.names
    else:
        results = model_assembled.predict(source=img, imgsz=640, conf=0.25)
        class_names = model_assembled.names

    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > 0.25:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                defect_name = class_names[class_id]
                detections.append({
                    "defect_type": defect_name,
                    "confidence": round(conf, 2),
                    "bounding_box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                })

    return {"status": "success", "board_analyzed": board_type, "defects_found": len(detections), "data": detections}

@app.post("/calibrate_endpoint")
async def save_calibration(file: UploadFile = File(...)):
    print("\n--- CALIBRATION FRAME RECEIVED ---")
    try:
        contents = await file.read()
        save_path = "baseline_environment.jpg"
        with open(save_path, "wb") as f:
            f.write(contents)
            
        print(f"DEBUG: Successfully saved to {save_path}")
        return {"status": "success", "message": "Baseline saved successfully"}
        
    except Exception as e:
        print(f"ERROR saving calibration: {e}")
        return {"status": "error", "message": str(e)}
    
# --- AI CHATBOT ROUTE ---
@app.post("/api/chat")
async def chat_assistant(message: ChatMessage):
    if not gemini_client:
        return {"reply": "Error: AI backend is offline. Missing API key."}

    # We inject system instructions to make it act like a PCB expert
    system_prompt = f"""You are 'PCB Doctor AI', an expert engineering assistant built into a PCB scanning web app. 
    Help the user troubleshoot circuit board manufacturing defects. Be concise and technical.
    
    Current UI Context: {message.context}
    
    User Question: {message.user_input}"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3-flash", 
            contents=system_prompt
        )
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"Error connecting to AI: {str(e)}"}    
    
@app.post("/ask-doctor")
async def ask_doctor_endpoint(req: DoctorRequest):
    if not gemini_client:
        return {"status": "error", "message": "AI backend offline. Missing API key."}

    # Prompt engineering specifically for this defect diagnosis
    prompt = f"""
    You are 'PCB Doctor AI', an expert hardware engineering assistant.
    The YOLO vision model just scanned a {req.board_type} circuit board.
    It detected the following defect: {req.defect_type}.
    
    Provide a concise, technical diagnosis. List 2 direct, actionable troubleshooting steps to resolve or verify this specific issue. 
    Keep formatting clean so it displays well on a web interface.
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3-flash", 
            contents=prompt
        )
        
        # Your JS uses innerHTML, so we convert markdown line breaks to HTML breaks
        formatted_advice = response.text.replace('\n', '<br>')
        
        return {
            "status": "success", 
            "defect": req.defect_type, 
            "advice": formatted_advice
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)