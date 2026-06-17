FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BASKETBALL_HOST=0.0.0.0 \
    BASKETBALL_PORT=8765

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-service.txt ./requirements-service.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-service.txt

COPY app ./app
COPY utils ./utils
COPY examples ./examples
COPY scripts ./scripts
COPY train.py train_mac.py dataset.py ./

RUN mkdir -p /app/model_checkpoints /app/analysis_outputs /app/output_videos /app/service_inputs

EXPOSE 8765

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
