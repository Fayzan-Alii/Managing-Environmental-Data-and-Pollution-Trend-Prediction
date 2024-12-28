# Managing Environmental Data and Pollution Trend Prediction

## Project Overview
This project aims to manage live environmental data streams, predict pollution trends, and alert high-risk pollution days. It leverages modern data versioning techniques with DVC, machine learning models for time-series analysis, and monitoring tools for real-time system performance.

Key outcomes include efficient data collection, robust version control, accurate pollution predictions, and a fully deployed API for real-time predictions.

## Features
- **Automated Data Management**: Collect and version live environmental data using APIs and DVC.
- **Time-Series Pollution Prediction**: Build ARIMA and LSTM models to forecast air quality levels and AQI trends.
- **Experiment Tracking**: Use MLflow to log and optimize model training.
- **Deployment and Monitoring**: Deploy the best model via FastAPI and monitor performance using Grafana and Prometheus.

## Technologies Used
- **Programming Languages**: Python
- **Libraries/Tools**: DVC, MLflow, FastAPI, Grafana, Prometheus
- **APIs for Data Collection**: OpenWeatherMap, AirVisual, EPA AirNow

## Dataset
Live data is fetched from publicly available APIs, including:
- **OpenWeatherMap**: Weather and Air Quality Data

- Follow the instructions in the project report.