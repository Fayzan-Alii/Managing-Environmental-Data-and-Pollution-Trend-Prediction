import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.keras
import mlflow.sklearn



# Load and prepare the data
data = pd.read_csv('D:/Work/MLops/course-project-Fai-zanAli/data/raw/environmental_data.csv')
data = data.drop(columns=['weather'])  # Drop categorical columns
data['timestamp'] = pd.to_datetime(data['timestamp'])  # Convert timestamp
data.set_index('timestamp', inplace=True)  # Set timestamp as index

# Handle missing values (impute with mean)
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)


# Scale features
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)

# Split data into train and test sets
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[:train_size], data_scaled[train_size:]

# ARIMA model
model_arima = ARIMA(train['aqi'], order=(5, 0, 1))
model_arima_fit = model_arima.fit()

# Make ARIMA predictions
arima_predictions = model_arima_fit.forecast(steps=len(test))

# ARIMA evaluation
arima_rmse = math.sqrt(mean_squared_error(test['aqi'], arima_predictions))
arima_mae = mean_absolute_error(test['aqi'], arima_predictions)

# Log ARIMA metrics in MLflow
with mlflow.start_run() as arima_run:
    mlflow.log_metric("ARIMA RMSE", arima_rmse)
    mlflow.log_metric("ARIMA MAE", arima_mae)

# Prepare data for LSTM
def create_lstm_data(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :-1])  # Features (all except target column 'aqi')
        y.append(data[i+lookback, -1])  # Target (AQI)
    return np.array(X), np.array(y)

lookback = 10
X, y = create_lstm_data(data_scaled.values)

# Split LSTM data into training and test sets
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(64, return_sequences=False, input_shape=(lookback, X.shape[2])))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))  # Output layer
model_lstm.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the LSTM model
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# LSTM predictions
lstm_predictions = model_lstm.predict(X_test)

# LSTM evaluation
lstm_rmse = math.sqrt(mean_squared_error(y_test, lstm_predictions))
lstm_mae = mean_absolute_error(y_test, lstm_predictions)

# Log LSTM metrics in MLflow
with mlflow.start_run() as lstm_run:
    mlflow.log_metric("LSTM RMSE", lstm_rmse)
    mlflow.log_metric("LSTM MAE", lstm_mae)
    mlflow.keras.log_model(model_lstm, "lstm_model")

model_lstm.save('models/lstm_model.h5')

# Compare models based on RMSE and MAE
if arima_rmse < lstm_rmse:
    print("ARIMA performs better based on RMSE.")
else:
    print("LSTM performs better based on RMSE.")

if arima_mae < lstm_mae:
    print("ARIMA performs better based on MAE.")
else:
    print("LSTM performs better based on MAE.")


