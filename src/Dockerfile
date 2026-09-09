FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# --no-cache-dir : removes the files saved in the cache that have been used to download the libs (to free up some space)
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data/conso/real_time_conso/

COPY artifacts/model_artifacts/Ridge_2026-07-30_23-24-53/Ridge_models.joblib /app/artifacts/model_artifacts/Ridge_2026-07-30_23-24-53/Ridge_models.joblib

COPY src/scheduler.py .
COPY src/predict.py .
COPY src/raw_preprocessing.py .
COPY src/feature_engineering.py .

CMD ["python", "-u", "scheduler.py"]