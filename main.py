
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="California House Price Prediction API",
    description=(
        "Predict California house values using a trained "
        "Random Forest model. Supports single and CSV predictions."
    ),
)

MODEL_PATH = Path("house_model.joblib")
FEATURES_PATH = Path("house_features.joblib")

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    raise RuntimeError(
        "Model files not found. Ensure both .joblib files "
        "are in the application directory."
    )

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

REQUIRED_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

# Ensure saved feature names match the API inputs.
if set(features) != set(REQUIRED_COLUMNS):
    raise RuntimeError(
        f"Saved model features do not match API inputs: {features}"
    )

#validating imports

class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median neighborhood income")
    HouseAge: float = Field(gt=0, le=52, description="Median house age")
    AveRooms: float = Field(gt=0, description="Average rooms per household")
    AveBedrms: float = Field(gt=0, description="Average bedrooms per household")
    Population: float = Field(gt=0, description="Neighborhood population")
    AveOccup: float = Field(gt=0, description="Average household occupancy")
    Latitude: float = Field(ge=32, le=42, description="California latitude")
    Longitude: float = Field(ge=-125, le=-114, description="California longitude")


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate uploaded data and arrange model features."""

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV contains no data rows."
        )

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_columns}"
        )

    data = df[features].copy()

    # Converting values into numbers; rejecting invalid text and missing values.
    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data.isnull().any().any():
        raise HTTPException(
            status_code=400,
            detail="Input columns contain missing or non-numeric values."
        )

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise HTTPException(
            status_code=400,
            detail="Input values must be finite numbers."
        )

    positive_columns = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup"
    ]

    for col in positive_columns:
        if (data[col] <= 0).any():
            raise HTTPException(
                status_code=400,
                detail=f"All values in {col} must be greater than zero."
            )

    if (data["HouseAge"] > 52).any():
        raise HTTPException(
            status_code=400,
            detail="HouseAge must not exceed 52."
        )

    if not data["Latitude"].between(32, 42).all():
        raise HTTPException(
            status_code=400,
            detail="Latitude must be between 32 and 42."
        )

    if not data["Longitude"].between(-125, -114).all():
        raise HTTPException(
            status_code=400,
            detail="Longitude must be between -125 and -114."
        )

    return data


def predict_prices(data: pd.DataFrame) -> np.ndarray:
    """Return predicted values in dollars."""

    predictions = np.asarray(model.predict(data), dtype=float)

    if not np.isfinite(predictions).all():
        raise ValueError("The model returned invalid predictions.")

    return predictions * 100000


@app.get("/")
def home():
    return {
        "message": "California House Price Prediction API",
        "status": "running",
        "endpoints": {
            "single_prediction": "POST /predict",
            "csv_prediction": "POST /predict-file",
            "documentation": "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": type(model).__name__,
        "features": features,
        "csv_predictions": True
    }

@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([house.model_dump()])
        input_data = input_data[features]

        price_usd = float(predict_prices(input_data)[0])

        return {
            "predicted_price_usd": round(price_usd, 2),
            "predicted_price": f"${price_usd:,.2f}",
            "currency": "USD",
            "model": type(model).__name__
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        ) from e


# using file 

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    filename = file.filename or ""

    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Parsing the  CSV bytes correctly.
        df = pd.read_csv(io.BytesIO(contents))
        input_data = validate_dataframe(df)

        prices = predict_prices(input_data)

        result = df.copy()
        result["predicted_price_usd"] = np.round(prices, 2)
        result["predicted_price"] = [
            f"${price:,.2f}" for price in prices
        ]

        output = io.StringIO()
        result.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    'attachment; filename="house_predictions.csv"'
            }
        )

    except HTTPException:
        raise

    except pd.errors.EmptyDataError as e:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV contains no readable data."
        ) from e

    except (UnicodeDecodeError, pd.errors.ParserError) as e:
        raise HTTPException(
            status_code=400,
            detail="Could not read the CSV. Check its encoding and format."
        ) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"CSV prediction failed: {str(e)}"
        ) from e

    finally:
        await file.close()
