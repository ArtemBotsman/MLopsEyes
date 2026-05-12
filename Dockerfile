FROM python:3.10-slim

WORKDIR /app

# CPU-only PyTorch (no nvidia-* wheels) + longer timeouts for slow networks.
# Local `pip install -r requirements.txt` unchanged; image matches CPU inference.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --default-timeout=1000 "Pillow>=8.0.0"

COPY open_eyes_classifier.py .
COPY eye_cnn_best_val_final.pth .

ENTRYPOINT ["python", "open_eyes_classifier.py"]
