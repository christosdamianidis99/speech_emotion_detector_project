# Speech Emotion Recognition Using CNN–BiGRU  
Multi-Corpus Training, FastAPI Deployment, and Flutter Mobile Demo  
Leiden University – Advanced Computing and Systems Seminar

This repository contains the complete workflow for a Speech Emotion Recognition (SER) system developed as part of the ACS seminar at Leiden University.  
It integrates three acted emotional-speech corpora, applies a unified preprocessing and feature extraction pipeline, trains a CNN–BiGRU model, evaluates it, and deploys it for real-time inference using FastAPI and a Flutter mobile application.

---

## Project Structure

speech_emotion_detector_project/
│
├── notebooks/
│ ├── 01_data_overview.ipynb
│ ├── 02_metadata_labels.ipynb
│ ├── 03_split_data.ipynb
│ ├── 04_feature_extraction.ipynb
│ ├── 05_train_cnn_gru.ipynb
│ └── 06_evaluation_and_export.ipynb
│
├── ser_api/
│ ├── main.py
│ ├── model/
│ │ └── best_cnn_gru_ser.keras
│ └── utils/
│ ├── audio_utils.py
│ └── features.py
│
├── flutter_app/
│
├── reports/
│ ├── SER_Report.tex
│ ├── images/
│ │ ├── confusion_matrix.png
│ │ ├── training_validation_accuracy.png
│ │ ├── ravdess_waveform.png
│ │ ├── cremad_waveform.png
│ │ └── iemocap_waveform.png
│ └── pdf/
│
├── requirements.txt
└── README.md


---

## 1. Research Workflow Summary

### Dataset Access

All WAV files from RAVDESS, CREMA-D, and IEMOCAP were recursively located, decoded, and analysed using Librosa.  
Sampling rate, duration, and waveform visualisations confirmed that the datasets were suitable for processing.

### Label Harmonisation

All corpora were mapped to four unified emotion labels:

angry, happy, neutral, sad

Labels inconsistent with this scheme (fear, disgust, surprise, frustration, etc.) were removed.

After filtering, 12,159 utterances remained.

### Stratified Splitting

A stratified 70/15/15 split was applied to preserve class proportions:

Train: 8511  
Validation: 1824  
Test: 1824  

### Acoustic Feature Extraction

All audio was converted to a uniform representation:

- 16 kHz mono  
- 25 ms window, 10 ms hop  
- 128 Mel bands  
- log-scaled Mel spectrograms  
- fixed length 3 seconds (128 × 300 tensor)

This representation matches typical setups in SER literature.

### Model Architecture (CNN–BiGRU)

The network consists of:

- Three convolutional blocks (3 × 3 kernels, 32→64→128 filters)
- Batch Normalization, ReLU, MaxPooling
- Temporal reshaping
- Bidirectional GRU (128 units)
- Dense + Dropout
- Softmax output layer (4 classes)

The model was trained using Adam (1e–3), batch size 32, and early stopping.

### Evaluation Results

Test accuracy: 67.4 percent  
Macro-F1: 0.68  

Highest performance was observed for angry and sad; happy was the most confusable class, consistent with findings in prior SER research.

A confusion matrix and per-class metrics are included in the final report.

---

## 2. Deployment Architecture

### FastAPI Inference Server

The server performs:

1. Resampling to 16 kHz  
2. Log-Mel feature extraction identical to training  
3. Model inference  
4. Returning predicted class and full probability distribution

Endpoint: POST /predict


Response structure:

```json
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

The label order matches the training configuration.

Flutter Mobile Client

The mobile application:

Records a 3-second speech clip

Plays the recording back

Sends a 16 kHz mono WAV file to the FastAPI server

Displays:

predicted emotion

confidence score

probability distribution

inference time

entropy and margin (simple interpretability indicators)

This enables an interactive real-time SER demo.

3. Running the FastAPI Inference Server
Activate the environment:

conda activate ser_env


Navigate to the server directory:

cd ser_api


Start the server:

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload


Server will be available at:

http://localhost:8000


Interactive API docs:

http://localhost:8000/docs

4. Requirements

The required packages (see requirements.txt):

numpy
librosa
scikit-learn
tensorflow==2.15
keras==3.0.0
fastapi
uvicorn
soundfile
pydantic
python-multipart

5. Acknowledgements

This work was completed as part of the Advanced Computing and Systems seminar at Leiden University.

Datasets:

RAVDESS

CREMA-D

IEMOCAP

Model design informed by:

Satt et al. (Interspeech 2017)

Neumann and Vu (Interspeech 2017)

Latif et al. (IEEE TAC 2019)

Atmaja and Akagi (Sensors 2020)






