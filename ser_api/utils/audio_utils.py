# ser_api/utils/audio_utils.py
from pathlib import Path
import numpy as np
import librosa

SR = 16000
N_MELS = 128
HOP_LENGTH = 160
WIN_LENGTH = 400
MAX_DURATION = 3.0
MAX_FRAMES = int(MAX_DURATION * SR / HOP_LENGTH)

def load_audio_from_bytes(data: bytes, sr: int = SR) -> np.ndarray:
    import io
    y, orig_sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y
