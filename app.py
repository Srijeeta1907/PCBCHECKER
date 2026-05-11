from pydantic import BaseModel
from groq import Groq          # pip install groq
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import Response

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

# ─────────────────────────────────────────────────────
#  Groq Client  (100% Free — 14,400 requests/day)
#  Get free key: https://console.groq.com/keys
# ─────────────────────────────────────────────────────
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("✅ Groq AI client initialized successfully.")
except Exception as e:
    print(f"WARNING: Could not initialize Groq client: {e}")
    groq_client = None

GROQ_MODEL = "llama-3.3-70b-versatile"   # Free, fast, very capable


# Define the structure for incoming chat messages
class ChatMessage(BaseModel):
    user_input: str
    context: str = ""

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

# Load Models
model_assembled  = YOLO("qcba_inspector.pt")
model_bare       = YOLO("best 1.pt")
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
    return Response(content="", media_type="image/x-icon")

# --- AI LOGIC ---

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

    if board_type == "bare":
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


@app.post("/analyze-pcba")
async def analyze_pcba(file: UploadFile = File(...), board_type: str = Form(default="assembled")):
    print(f"\n--- NEW FILE SCAN STARTED ---")
    print(f"DEBUG: Requested brain: '{board_type}'")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.imwrite("debug_before_ai.jpg", img)

    if board_type == "bare":
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
    if not groq_client:
        return {"reply": "Error: AI backend is offline. Missing GROQ_API_KEY in .env file."}

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are 'PCB Doctor AI', an expert engineering assistant built into a PCB scanning web app. "
                        "Help the user troubleshoot circuit board manufacturing defects. Be concise and technical.\n\n"
                        f"Current UI Context: {message.context}"
                    )
                },
                {
                    "role": "user",
                    "content": message.user_input
                }
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"reply": f"Error connecting to AI: {str(e)}"}


@app.post("/ask-doctor")
async def ask_doctor_endpoint(req: DoctorRequest):
    if not groq_client:
        return {"status": "error", "message": "AI backend offline. Missing GROQ_API_KEY in .env file."}

    prompt = f"""
    The YOLO vision model just scanned a {req.board_type} circuit board.
    It detected the following defect: {req.defect_type}.
    
    Provide a concise, technical diagnosis. List 2 direct, actionable troubleshooting steps to resolve or verify this specific issue. 
    Keep formatting clean so it displays well on a web interface.
    """

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": "You are 'PCB Doctor AI', an expert hardware engineering assistant specialising in PCB manufacturing defects."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        formatted_advice = response.choices[0].message.content.replace('\n', '<br>')

        return {
            "status": "success",
            "defect": req.defect_type,
            "advice": formatted_advice
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)