Speech Emotion Recognition Using CNN–BiGRU
Multi-Corpus Training, FastAPI Deployment, and Flutter Mobile Demo

This repository contains the full workflow for a Speech Emotion Recognition (SER) system developed as part of the Leiden University – Advanced Computing & Systems seminar.
It includes:

Three-corpus data integration: RAVDESS, CREMA-D, IEMOCAP

Unified preprocessing: label harmonisation, log-Mel spectrograms

Deep model: CNN + BiGRU hybrid architecture

Training notebooks with full pipeline

FastAPI inference server that reproduces preprocessing exactly

Flutter mobile application for real-time SER inference

The goal of this project is both research-focused evaluation and practical deployment, enabling a live demo of SER running on mobile.

Project Structure

speech_emotion_detector_project/
│
├── notebooks/                      # Step-by-step pipeline notebooks
│   ├── 01_dataset_access.ipynb
│   ├── 02_metadata_labels.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment_preparation.ipynb
│
├── ser_api/                        # FastAPI inference server
│   ├── main.py
│   ├── model/
│   │     └── best_cnn_gru_ser.keras
│   └── utils/
│         ├── audio_utils.py
│         └── features.py
│
├── flutter_app/ (optional)         # Flutter client if included
│
├── reports/                        # LaTeX final report, Overleaf-ready
│   ├── SER_Report.tex
│   ├── images/
│   │     ├── confusion_matrix.png
│   │     ├── training_validation_accuracy.png
│   │     ├── ravdess_waveform.png
│   │     ├── cremad_waveform.png
│   │     └── iemocap_waveform.png
│   └── pdf/
│
├── requirements.txt                # Python dependencies (generated manually)
└── README.md                       # This file


1. Summary of the Research Workflow
✔ Dataset Access

All .wav files from RAVDESS, CREMA-D, and IEMOCAP were loaded, inspected, normalised, counted, and verified.

✔ Label Harmonisation

Mapped each corpus to four unified classes:
angry, happy, neutral, sad
All incompatible emotion categories removed.

✔ Stratified Splitting

Unified dataset size: 12,159 utterances
Split into a balanced 70/15/15 train/validation/test partition.

✔ Feature Extraction

16 kHz mono

25ms window / 10ms hop

128 Mel bands

Log-Mel spectrograms

3 second fixed length (128×300)

✔ Model

CNN–BiGRU architecture:

3 convolutional blocks

Bidirectional GRU (128 units)

Dense layers + dropout

Softmax classification (4 classes)

✔ Results

67.4% test accuracy

Best F1 for angry/sad; lowest for happy

Full confusion matrix included in report



2. Deployment Architecture
✔ FastAPI Server (ser_api/)

Loads the trained CNN–BiGRU model

Reproduces preprocessing exactly (resampling, padding, log-Mel)

Exposes:POST /predict

Response:
{
  "emotion": "neutral",
  "probability": 0.74,
  "all_probs": {
    "angry": 0.05,
    "happy": 0.11,
    "neutral": 0.74,
    "sad": 0.10
  }
}

3. Flutter Mobile Client

The mobile client performs:

Records 3 seconds of speech

Plays back the recorded audio

Uploads WAV (16kHz mono) to FastAPI

Displays:

predicted emotion

confidence

probability distribution

inference time

entropy and margin (uncertainty indicators)

This allows real-time SER demos on a phone.


4. Running the Inference Server
Step 1 — Activate the environment
conda activate ser_env

Step 2 — Navigate to server folder
cd ser_api


Step 3 — Launch server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload


Expected output:
Uvicorn running on http://0.0.0.0:8000

5. requirements.txt

numpy
librosa
scikit-learn
tensorflow==2.15
keras==3.0.0
fastapi
uvicorn
soundfile
