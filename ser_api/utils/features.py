# ser_api/utils/features.py
import numpy as np
import librosa
from .audio_utils import SR, N_MELS, HOP_LENGTH, WIN_LENGTH, MAX_FRAMES

def audio_to_logmel_fixed(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=1024,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=SR // 2,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # pad / truncate to MAX_FRAMES
    if logmel.shape[1] < MAX_FRAMES:
        pad = MAX_FRAMES - logmel.shape[1]
        logmel = np.pad(logmel, ((0,0),(0,pad)), mode="constant")
    else:
        logmel = logmel[:, :MAX_FRAMES]

    # add channel dimension
    logmel = logmel[..., np.newaxis].astype("float32")
    return logmel

def mel_to_png_base64(mel):
    """
    Convert a log-mel spectrogram (2D numpy array) to a PNG image
    encoded as a base64 string, so the Flutter client can display it.
    """
    fig = plt.figure(figsize=(3, 2), dpi=100)
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches=0, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")
