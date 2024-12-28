import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from prometheus_client import Counter, Gauge, start_http_server
import time
import traceback

# Comprehensive Prometheus metrics
class PrometheusMetrics:
    def __init__(self):
        self.API_REQUEST_COUNT = Counter("api_request_count", "Total API requests")
        self.API_REQUEST_ERRORS = Counter("api_request_errors", "Total API request errors")
        self.API_LATENCY = Gauge("api_latency_seconds", "API request latency")
        self.INPUT_VALIDATION_ERRORS = Counter("input_validation_errors", "Input validation errors")
        self.PREDICTION_COUNT = Counter("prediction_count", "Successful predictions")
        self.PREDICTION_ERROR_COUNT = Counter("prediction_error_count", "Prediction errors")
        self.INPUT_FEATURE_COUNT = Gauge("input_feature_count", "Number of input features")

metrics = PrometheusMetrics()
start_http_server(8002)  # Metrics endpoint

class PredictionInput(BaseModel):
    features: list[list[float]]

class PredictionOutput(BaseModel):
    predictions: list[float]

# Load pre-trained LSTM model and scalers
model = load_model('D:/Work/MLops/course-project-Fai-zanAli/models/lstm_model.h5')
data = pd.read_csv('D:/Work/MLops/course-project-Fai-zanAli/data/raw/environmental_data.csv')
data = data.drop(columns=['weather', 'timestamp', 'aqi'])

imputer = SimpleImputer(strategy='mean')
imputer.fit(data)

scaler = StandardScaler()
scaler.fit(imputer.transform(data))

app = FastAPI(title="Air Quality LSTM Predictor")

@app.post("/predict", response_model=PredictionOutput)
async def predict_aqi(input_data: PredictionInput):
    try:
        metrics.API_REQUEST_COUNT.inc()
        start_time = time.time()

        metrics.INPUT_FEATURE_COUNT.set(len(input_data.features[0]))

        if len(input_data.features[0]) != data.shape[1]:
            metrics.INPUT_VALIDATION_ERRORS.inc()
            raise HTTPException(status_code=400, detail=f"Input must have {data.shape[1]} features")

        input_array = np.array(input_data.features)
        input_imputed = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imputed)

        input_reshaped = np.repeat(input_scaled, 10, axis=0).reshape(1, 10, input_scaled.shape[1])
        predictions = model.predict(input_reshaped).flatten()

        metrics.PREDICTION_COUNT.inc(len(predictions))
        latency = time.time() - start_time
        metrics.API_LATENCY.set(latency)

        return {"predictions": predictions.tolist()}

    except HTTPException:
        raise
    except Exception as e:
        metrics.API_REQUEST_ERRORS.inc()
        metrics.PREDICTION_ERROR_COUNT.inc()
        
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)