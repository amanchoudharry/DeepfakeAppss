"""
FastAPI server for deepfake detection.
Replace the mock inference in `predict_deepfake()` with your actual model.
"""

import base64
import io
import json
import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Keep heavyweight ML dependencies out of the startup path unless explicitly enabled.
ENABLE_DEEPFAKE_MODEL = os.getenv("ENABLE_DEEPFAKE_MODEL", "false").lower() == "true"
USE_DEEPFAKE_MODEL = False
cv2 = None
np = None
torch = None
nn = None
models = None
transforms = None

MODEL_DIR = Path(__file__).resolve().parent / "models"
FACE_PROTO = MODEL_DIR / "face_detector" / "deploy.prototxt"
FACE_MODEL = MODEL_DIR / "face_detector" / "res10_300x300_ssd_iter_140000.caffemodel"
MODEL_WEIGHTS = MODEL_DIR / "deepfake_model.pth"
FACE_CONF_THRESHOLD = 0.5

deepfake_model = None
face_net = None
preprocess = None
audio_model = None
audio_transform = None
DEVICE = None

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for feedback (replace with a DB in production)
feedback_store = []
FEEDBACK_FILE = Path("feedback.json")


# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    frame: str          # base64-encoded JPEG/PNG data-URL
    room_id: Optional[str] = None
    user_id: Optional[str] = None

class PredictAudioRequest(BaseModel):
    audio: str          # base64-encoded audio chunk
    room_id: Optional[str] = None
    user_id: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: str     # "real" | "fake"
    confidence: float   # 0.0 – 1.0
    reason: str
    processing_time_ms: float


class FeedbackRequest(BaseModel):
    room_id: Optional[str] = None
    accurate: bool
    comment: Optional[str] = None
    prediction_history: Optional[list] = None
    timestamp: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str


# ── Mock model (replace with real inference) ──────────────────────────────────

def load_deepfake_model() -> None:
    global deepfake_model, face_net, preprocess, audio_model, audio_transform
    global USE_DEEPFAKE_MODEL, DEVICE, cv2, np, torch, nn, models, transforms

    if not ENABLE_DEEPFAKE_MODEL:
        print("[deepfake] Model loading disabled. Set ENABLE_DEEPFAKE_MODEL=true to use real inference.")
        return

    try:
        import cv2 as cv2_module
        import numpy as np_module
        import torch as torch_module
        import torch.nn as nn_module
        from torchvision import models as torchvision_models
        from torchvision import transforms as torchvision_transforms

        cv2 = cv2_module
        np = np_module
        torch = torch_module
        nn = nn_module
        models = torchvision_models
        transforms = torchvision_transforms
        USE_DEEPFAKE_MODEL = True
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError as exc:
        print("[deepfake] ML dependencies are not installed. Using mock inference.", exc)
        return

    if not USE_DEEPFAKE_MODEL:
        print("[deepfake] Torch/OpenCV dependencies are not installed. Using mock inference.")
        return

    if not MODEL_WEIGHTS.exists() or not FACE_PROTO.exists() or not FACE_MODEL.exists():
        print("[deepfake] Missing model files in 'models/'. Using mock inference.")
        return

    try:
        try:
            model = models.resnet18(weights=None)
            am = models.resnet18(weights=None)
        except TypeError:
            model = models.resnet18(pretrained=False)
            am = models.resnet18(pretrained=False)

        model.fc = nn.Linear(model.fc.in_features, 2)
        state = torch.load(str(MODEL_WEIGHTS), map_location=DEVICE)
        model.load_state_dict(state)
        deepfake_model = model.to(DEVICE).eval()

        face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        am.fc = nn.Linear(am.fc.in_features, 2)
        AUDIO_WEIGHTS = MODEL_DIR / "audio_model.pth"
        if AUDIO_WEIGHTS.exists():
            am.load_state_dict(torch.load(str(AUDIO_WEIGHTS), map_location=DEVICE))
            audio_model = am.to(DEVICE).eval()
        audio_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        print("[deepfake] Models and face detector loaded successfully.")
    except Exception as exc:
        print("[deepfake] Failed to load model:", exc)
        deepfake_model = None
        face_net = None
        audio_model = None

load_deepfake_model()


def predict_deepfake(image_bytes: bytes) -> dict:
    if USE_DEEPFAKE_MODEL and deepfake_model is not None and face_net is not None:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image bytes")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < FACE_CONF_THRESHOLD:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            tensor = preprocess(face).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = deepfake_model(tensor)
                probs = torch.softmax(logits, dim=1)[0]

            fake_prob = float(probs[1].item())
            real_prob = float(probs[0].item())
            faces.append({
                "fake_prob": fake_prob,
                "real_prob": real_prob,
                "confidence": round(max(fake_prob, real_prob), 3),
            })

        if faces:
            best_face = max(faces, key=lambda item: item["fake_prob"])
            prediction = "fake" if best_face["fake_prob"] > 0.5 else "real"
            confidence = best_face["confidence"]
            reason = (
                f"Face-level fake probability {best_face['fake_prob']:.3f}"
                if prediction == "fake"
                else f"Face-level real probability {best_face['real_prob']:.3f}"
            )
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reason": reason,
            }

        return {
            "prediction": "real",
            "confidence": 0.35,
            "reason": "No face detected in the frame.",
        }

    # Fallback mock implementation when model dependencies are unavailable.
    time.sleep(0.05)
    fake_prob = random.betavariate(2, 5)
    real_reasons = [
        "Natural facial micro-movements detected",
        "Consistent lighting across facial landmarks",
        "Authentic eye-blinking patterns observed",
        "No temporal inconsistencies found",
        "Skin texture appears organic and unaltered",
    ]
    fake_reasons = [
        "Unnatural blending artifacts around facial boundaries",
        "Inconsistent lighting on nose and cheekbones",
        "Eye-blinking frequency anomaly detected",
        "Temporal flickering in hair region",
        "GAN-generated texture pattern identified",
        "Frequency domain anomalies detected",
    ]

    is_fake = fake_prob > 0.5
    confidence = fake_prob if is_fake else 1.0 - fake_prob
    return {
        "prediction": "fake" if is_fake else "real",
        "confidence": round(min(confidence + random.uniform(0, 0.15), 0.99), 3),
        "reason": random.choice(fake_reasons if is_fake else real_reasons),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/")
async def root():
    return {"status": "ok", "service": "Deepfake Detection API"}

def predict_audio(audio_bytes: bytes) -> dict:
    if USE_DEEPFAKE_MODEL and audio_model is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import librosa
            import librosa.display
            import soundfile as sf

            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        except Exception as e:
            return {"prediction": "real", "confidence": 0.5, "reason": "Audio format unsupported; skipping audio check"}
            
        if len(y) == 0:
            return {"prediction": "real", "confidence": 0.3, "reason": "Empty audio"}
            
        y = y[:22050*2]
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        plt.figure(figsize=(3,3))
        librosa.display.specshow(spec_db, sr=sr)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        img_array = np.frombuffer(buf.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return {"prediction": "real", "confidence": 0.3, "reason": "Failed to generate spectrogram image"}
            
        tensor = audio_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = audio_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
        fake_prob = float(probs[0].item())
        real_prob = float(probs[1].item())
        prediction = "fake" if fake_prob > 0.6 else "real"
        confidence = float(max(fake_prob, real_prob))
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": f"Audio anomaly prob: {fake_prob:.3f}" if prediction=="fake" else f"Audio natural prob: {real_prob:.3f}"
        }
    return {"prediction": "real", "confidence": 0.3, "reason": "Mock audio prediction"}

@app.post("/predict-audio", response_model=PredictResponse)
async def predict_audio_rt(req: PredictAudioRequest):
    start = time.perf_counter()
    try:
        header, encoded = req.audio.split(",", 1) if "," in req.audio else ("", req.audio)
        audio_bytes = base64.b64decode(encoded)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {e}")
        
    result = predict_audio(audio_bytes)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        reason=result["reason"],
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    start = time.perf_counter()

    # Decode base64 frame
    try:
        header, encoded = req.frame.split(",", 1) if "," in req.frame else ("", req.frame)
        image_bytes = base64.b64decode(encoded)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {e}")

    if len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="Frame too small")

    try:
        result = predict_deepfake(image_bytes)
    except Exception as e:
        print(f"Error in predict_deepfake: {e}")
        result = {"prediction": "real", "confidence": 0.0, "reason": f"Frame error: {e}"}

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        reason=result["reason"],
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    entry = {
        "id": f"fb_{int(time.time()*1000)}",
        "room_id": req.room_id,
        "accurate": req.accurate,
        "comment": req.comment,
        "prediction_history": req.prediction_history,
        "timestamp": req.timestamp or datetime.utcnow().isoformat(),
        "created_at": datetime.utcnow().isoformat(),
    }
    feedback_store.append(entry)

    # Persist to disk
    try:
        existing = json.loads(FEEDBACK_FILE.read_text()) if FEEDBACK_FILE.exists() else []
        existing.append(entry)
        FEEDBACK_FILE.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass

    return FeedbackResponse(
        status="success",
        message="Feedback recorded. Thank you!",
        feedback_id=entry["id"],
    )


@app.get("/feedback")
async def get_feedback():
    return {"total": len(feedback_store), "entries": feedback_store[-50:]}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
