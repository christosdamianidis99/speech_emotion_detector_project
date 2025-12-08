from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf

import tensorflow as tf  
import keras              

from utils.audio_utils import load_audio_from_bytes
from utils.features import audio_to_logmel_fixed

MODEL_PATH = "model/best_cnn_gru_ser.keras"
model = keras.models.load_model(MODEL_PATH) 

CLASS_NAMES = ["angry", "happy", "neutral", "sad"]

app = FastAPI(title="Speech Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    data = await file.read()

    y = load_audio_from_bytes(data)
    feat = audio_to_logmel_fixed(y)      
    feat = np.expand_dims(feat, axis=0)  

    probs = model.predict(feat)[0]     
    idx = int(np.argmax(probs))
    emotion = CLASS_NAMES[idx]
    prob = float(probs[idx])

    return {
        "emotion": emotion,
        "probability": prob,
        "all_probs": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
