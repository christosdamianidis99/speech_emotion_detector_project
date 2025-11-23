import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import io
import base64

# For mel-spectrogram rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
SR = 16000
N_MELS = 128
MAX_DURATION = 3.0     # seconds
MAX_SAMPLES = int(SR * MAX_DURATION)
MODEL_PATH = "model/best_cnn_gru_ser.keras"

EMOTIONS = ["angry", "happy", "neutral", "sad"]

# -------------------------------------------------------
# APP INIT
# -------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------
print("Loading TensorFlow model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# -------------------------------------------------------
# AUDIO PROCESSING
# -------------------------------------------------------
def load_audio(path, sr=SR):
    """Load audio, convert to mono if needed, normalize and trim/pad to MAX_DURATION."""
    y, orig_sr = librosa.load(path, sr=None, mono=True)

    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)

    # Trim/pad to exactly MAX_DURATION
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    else:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y


def audio_to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=512
    )
    logmel = librosa.power_to_db(mel).astype(np.float32)
    return logmel


def fix_length(logmel):
    """Force mel spectrogram to 128 × 300."""
    desired_frames = 300
    if logmel.shape[1] > desired_frames:
        logmel = logmel[:, :desired_frames]
    else:
        pad = desired_frames - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)))
    return logmel


# -------------------------------------------------------
# MEL IMAGE ENCODING
# -------------------------------------------------------
def mel_to_png_base64(mel):
    """Render mel spectrogram to PNG and return base64 string."""
    fig = plt.figure(figsize=(3, 2), dpi=100)
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches=0, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# -------------------------------------------------------
# API ENDPOINT
# -------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Store temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load audio
    y = load_audio(tmp_path)
    logmel = audio_to_logmel(y)
    logmel = fix_length(logmel)

    # Prepare for model
    x = logmel[np.newaxis, :, :, np.newaxis]  # (1, 128, 300, 1)

    # Predict
    probs = model.predict(x)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_label = EMOTIONS[predicted_idx]
    confidence = float(probs[predicted_idx])

    # mel spectrogram image
    mel_img_str = mel_to_png_base64(logmel)

    # JSON response
    return {
        "emotion": predicted_label,
        "probability": confidence,
        "all_probs": {e: float(p) for e, p in zip(EMOTIONS, probs)},
        "mel_image": mel_img_str
    }


# -------------------------------------------------------
# ROOT CHECK
# -------------------------------------------------------
@app.get("/")
def root():
    return {"message": "SER API is running"}


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
