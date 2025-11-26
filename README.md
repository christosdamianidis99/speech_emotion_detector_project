# Speech Emotion Recognition Model and Flutter App

This project aims to recognize human emotions (happy, sad, angry, neutral) from speech using a deep learning model (CRNN) trained in Python with Keras/TensorFlow. The repository also contains a Flutter application for live emotion detection using the trained model via a Python Flask server.

## Project Structure

* `EmotionRecognition_Final.ipynb`: Jupyter Notebook containing the data processing, model training, evaluation, and export steps.
* `server.py`: Python Flask server script to load the trained Keras model and provide predictions via an API endpoint.
* `unbiased_app_model.keras`: The final, trained Keras model file used by the server.
* `requirements.txt`: (We will create this next) List of Python dependencies.
* `flutter_app/`: (Placeholder for later) Directory containing the Flutter mobile application code.

## Setup (Python Backend)

1.  Clone the repository: `git clone ...`
2.  Navigate to the repository folder: `cd your-repository-name`
3.  Create a Conda environment: `conda create --name ser_final python=3.9 -y`
4.  Activate the environment: `conda activate ser_final`
5.  Install dependencies: `pip install -r requirements.txt`
6.  Run the server: `python server.py`

## Flutter Application (`emotion_detector_app/`)

A mobile application built with Flutter that records audio, sends it to the Python backend server, and displays the predicted emotion.

### Setup (Flutter App - Local Development)

1.  Ensure you have the Flutter SDK installed.
2.  Navigate to the app directory: `cd emotion_detector_app`
3.  Install dependencies: `flutter pub get`
4.  **Important:** Ensure the Python server (`server.py`) is running locally.
5.  Connect an Android device via USB with USB Debugging enabled.
6.  Set up ADB port forwarding: `adb reverse tcp:5000 tcp:5000`
7.  Ensure the `serverUrl` constant in `lib/main.dart` is set to `http://127.0.0.1:5000/predict`.
8.  Run the app: `flutter run`
