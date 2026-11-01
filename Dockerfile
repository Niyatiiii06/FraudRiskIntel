FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY fraud/ fraud/
COPY models/fraud_model.joblib models/fraud_model.joblib

EXPOSE 8000
CMD ["uvicorn", "fraud.api:app", "--host", "0.0.0.0", "--port", "8000"]
