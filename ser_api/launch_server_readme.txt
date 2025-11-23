separate terminal: adb reverse tcp:8000 tcp:8000


cd C:\Code\speech_emotion_detector_project\ser_api

conda activate ser_env

pip install uvicorn fastapi python-multipart


python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
