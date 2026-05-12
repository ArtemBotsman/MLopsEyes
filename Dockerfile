FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY open_eyes_classifier.py .
COPY eye_cnn_best_val_final.pth .

ENTRYPOINT ["python", "open_eyes_classifier.py"]
