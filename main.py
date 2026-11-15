import io
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# Initialize FastAPI
app = FastAPI()


# Load trained model and feature columns
model = joblib.load("housing_model.joblib")
features = joblib.load("housing_model_columns.joblib")


# Pydantic model for input validation
class HousingFeatures(BaseModel):

    MedInc: float = Field(gt=0, description="Median Income in block group")
    HouseAge: float = Field(gt=0, description="Median House Age in block group")
    AveRooms: float = Field(gt=0, description="Average Rooms per Household")
    AveBedrms: float = Field(gt=0, description="Average Bedrooms per Household")
    Population: float = Field(gt=0, description="Population in block group")
    AveOccup: float = Field(gt=0, description="Average Occupancy per Household")
    Latitude: float = Field(gt=-90, lt=90, description="Latitude of block group")
    Longitude: float = Field(gt=-180, lt=180, description="Longitude of block group")


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "California Housing Prediction",
        "status": "running",
        "endpoint": "Send POST request to /predict"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "Random Forest Regressor",
        "features": features,
        "avg_error": "$32,789"
    }


# Single house prediction
@app.post("/predict")
def predict(house: HousingFeatures):

    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        input_data = input_data[features]

        predicted = model.predict(input_data)[0]

        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "confidence_range": (
                f"${(predicted - 0.1) * 100000:,.0f} - "
                f"${(predicted + 0.1) * 100000:,.0f}"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# CSV batch prediction
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):

    # Check file extension
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file"
        )

    try:
        # Read uploaded file
        contents = await file.read()

        df = pd.read_csv(io.BytesIO(contents))

        # Required columns
        required_columns = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]

        # Check missing columns
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"These columns are missing: {missing_columns}"
            )

        # Validate numeric data
        input_df = df[required_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        if input_df.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="CSV contains missing or non-numeric feature values"
            )

        if (input_df <= 0).any().any():
            # Latitude and longitude can be negative
            positive_columns = [
                "MedInc", "HouseAge", "AveRooms",
                "AveBedrms", "Population", "AveOccup"
            ]

            if (input_df[positive_columns] <= 0).any().any():
                raise HTTPException(
                    status_code=400,
                    detail="Positive features must be greater than zero"
                )

        if not input_df["Latitude"].between(-90, 90).all():
            raise HTTPException(
                status_code=400,
                detail="Latitude must be between -90 and 90"
            )

        if not input_df["Longitude"].between(-180, 180).all():
            raise HTTPException(
                status_code=400,
                detail="Longitude must be between -180 and 180"
            )

        # Arrange columns in training order
        input_df = input_df[features]

        # Generate predictions
        predictions = model.predict(input_df)

        # Convert predictions into USD
        df["predicted_price_usd"] = predictions * 100000

        # Format predictions
        df["predicted_price_usd"] = df[
            "predicted_price_usd"
        ].apply(lambda x: f"${x:,.0f}")

        # Convert DataFrame into CSV
        output = df.to_csv(index=False)

        # Return downloadable CSV
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predictions.csv"
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    finally:
        await file.close()
