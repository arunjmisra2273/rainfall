```markdown
# IMD Sub-Seasonal Rainfall Forecaster

A production-ready machine learning pipeline and API that predicts 14-day sub-seasonal rainfall across India's 36 meteorological subdivisions, trained on 125 years of historical Indian Meteorological Department (IMD) gridded data (1901–2025).

## System Architecture

The pipeline processes massive 3D `.grd` files, performs exact geographic masking using native IMD coordinate reference systems, and trains a highly optimized global gradient boosting model that outperforms historical 125-year climatology baselines. 

The codebase is structured into strictly ordered, modular Python scripts executing the end-to-end data pipeline:
* **Step 1-4:** Data retrieval, spatial projection (EPSG:32644 to WGS84), spatial masking, and EDA.
* **Step 5-6:** Time-series feature engineering (autoregressive lags, volatility windows) and baseline validation.
* **Step 7:** Model shootout (Scikit-Learn vs. LightGBM vs. XGBoost) and artifact serialization.
* **Step 8:** FastAPI deployment.

## The API
Because real-time IMD grid updates often lag by a year, the FastAPI application features a hybrid architecture for both testing and live inference. This ensures a streamlined, intuitive interface for portfolio reviewers while supporting actual production usage.

### 1. Live Forecast Endpoint (`POST /predict/live`)
For real-time predictions (2026+). Requires the client to inject current weather data.
```json
{
  "subdivision": "KERALA",
  "day_of_year": 200,
  "past_14d_rain": 250.5,
  "past_30d_rain": 510.2,
  "past_60d_rain": 890.0,
  "volatility_14d": 12.4,
  "lag_14": 15.2,
  "lag_21": 8.1
}

```

### 2. Historical Auto-Lookup Endpoint (`POST /predict/historical`)

For validating historical events up to 2025. The server automatically reconstructs the lagging weather conditions from the internal database.

```json
{
  "subdivision": "KERALA",
  "year": 2024,
  "day_of_year": 200
}

```

## Quickstart

1. Install dependencies:
`pip install -r requirements.txt`
2. Generate the data pipeline and model (requires `data/` and `models/` directories):
`python step_7_train_global_model.py`
3. Launch the API:
`python main.py`
4. Access the interactive Swagger UI at `http://localhost:8000/docs`

```