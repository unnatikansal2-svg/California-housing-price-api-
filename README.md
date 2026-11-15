# 🏡 California Housing Price Prediction API

> A machine-learning REST API that predicts median house prices in California from census block-group data, built with **FastAPI** and a **Random Forest** model.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project trains a `RandomForestRegressor` on the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) (20,640 records) and serves it through a FastAPI application.

You can get a prediction in two ways:

- **Single prediction** – send the 8 features of one block group as JSON.
- **Batch prediction** – upload a CSV and download it back with a `predicted_price_usd` column added.

FastAPI also generates interactive documentation automatically, so you can try everything from your browser.

## ✨ Features

- Random Forest model (100 trees) trained on an 80/20 train–test split
- Input validation with Pydantic (invalid values return a clear error)
- Single-house prediction endpoint (`/predict`)
- CSV batch prediction endpoint with column and data validation (`/predict-file`)
- Health-check endpoint that reports the model's features and average error
- Auto-generated Swagger UI at `/docs`

## 🧰 Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| API | FastAPI, Uvicorn, Pydantic |
| ML | scikit-learn (Random Forest) |
| Data | pandas, NumPy |
| Model storage | joblib |

## 📁 Project Structure

```
california-housing-price-api/
├── main.py              # FastAPI application (API endpoints)
├── train.py             # Trains the model and saves it with joblib
├── explore.py           # Quick look at the dataset (shape, head, describe)
├── requirements.txt     # Python dependencies
├── sample_data/
│   ├── california_housing_api_test_data.csv   # Example input for /predict-file
│   └── predictions.csv                        # Example output from /predict-file
├── README.md
├── LICENSE
└── .gitignore
```

> The trained model files (`housing_model.joblib`, `housing_model_columns.joblib`) are **not stored in the repo** — the model is about 145 MB, which is above GitHub's 100 MB file limit. You create them by running `train.py`.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/california-housing-price-api.git
cd california-housing-price-api
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

This downloads the dataset, trains the model, prints the Mean Absolute Error and R² score, and saves `housing_model.joblib` and `housing_model_columns.joblib` in the project folder. (The first run needs an internet connection to download the dataset.)

### 4. Start the API

```bash
uvicorn main:app --reload
```

The API is now running at **http://127.0.0.1:8000**.
Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

## 🔌 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Welcome message and API status |
| `GET` | `/health` | Model name, feature list and average error |
| `POST` | `/predict` | Predict the price for a single block group (JSON) |
| `POST` | `/predict-file` | Predict prices for many rows by uploading a CSV |

### `POST /predict`

**Request**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322,
    "AveOccup": 2.56,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

**Response**

```json
{
  "predicted_price": "$423,820",
  "confidence_range": "$413,820 - $433,820"
}
```

> ℹ️ `confidence_range` is a simple fixed band of ±$10,000 around the prediction. It is **not** a statistical confidence interval.

### `POST /predict-file`

Upload a CSV that contains the 8 feature columns (see below). The API returns a downloadable `predictions.csv` with an extra `predicted_price_usd` column.

```bash
curl -X POST "http://127.0.0.1:8000/predict-file" \
  -F "file=@sample_data/california_housing_api_test_data.csv" \
  -o predictions.csv
```

Example output (first rows):

```csv
MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,predicted_price_usd
5.2,25.0,5.4,1.2,1200.0,3.0,34.05,-118.25,"$251,752"
3.8,18.0,4.8,1.05,950.0,2.7,37.77,-122.42,"$245,537"
6.1,32.0,6.2,1.1,1450.0,2.9,36.6,-121.9,"$343,563"
```

The upload is rejected with a `400` error if the file is not a `.csv`, a required column is missing, a value is empty or non-numeric, or a value is out of range.

## 📊 Input Features

| Feature | Description |
| --- | --- |
| `MedInc` | Median income in the block group (in tens of thousands of dollars) |
| `HouseAge` | Median house age in the block group (years) |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Population of the block group |
| `AveOccup` | Average number of household members |
| `Latitude` | Latitude of the block group |
| `Longitude` | Longitude of the block group |

The model predicts the **median house value** of the block group. (Internally the dataset stores this in units of $100,000; the API converts it to dollars for you.)

## 📈 Model Performance

- **Algorithm:** Random Forest Regressor (`n_estimators=100`, `random_state=42`)
- **Split:** 80% training / 20% testing
- **Mean Absolute Error:** about **$32,789** (the value shown by the `/health` endpoint)

Run `python train.py` to see the exact MAE and R² score on your machine.

## ⚠️ Limitations

- The California Housing dataset is based on **1990 census data**, so predictions reflect 1990-era prices, not today's market.
- The model is trained on California only — predictions for other regions are not meaningful.
- Each prediction is for a *block group* (a neighbourhood-sized area), not an individual house.
- The `avg_error` shown in `/health` is a fixed value written in `main.py`; update it if you retrain with different settings.

## 🔮 Future Improvements

- [ ] Add a `Dockerfile` for one-command deployment
- [ ] Add automated tests (pytest) for the endpoints
- [ ] Compare other models (Gradient Boosting, XGBoost) and tune hyperparameters
- [ ] Return real prediction intervals instead of a fixed ±$10,000 band
- [ ] Deploy to a cloud platform (Render, Railway, etc.)

## 👩‍💻 Author

**Unnati**
B.Tech CSE (Data Science), Maharaja Agrasen Institute of Technology, Delhi

## 📄 License

This project is licensed under the [MIT License](LICENSE).
