# 🏡 California House Price Prediction

A Machine Learning project that predicts California house prices using the **Random Forest Regression** algorithm. The project combines a trained ML model, a **FastAPI backend** for prediction APIs, and a **Streamlit web application** for interactive predictions and batch CSV processing.

##  Project Overview

This project demonstrates the complete workflow of a machine learning application, from model training and evaluation to API development and user-facing deployment.

Users can enter property details to estimate a house price or upload a CSV file to generate predictions for multiple properties.

## ✨ Features

- **Machine Learning:** Uses `RandomForestRegressor` from Scikit-learn.
- **Interactive Web App:** Predict house prices through a Streamlit interface.
- **REST API:** FastAPI endpoints for individual predictions.
- **Batch Predictions:** Upload a CSV file to predict prices for multiple properties.
- **Input Validation:** Validates prediction inputs using Pydantic.
- **Model Persistence:** Saves the trained model and feature metadata using Joblib.
- **Model Evaluation:** Evaluates model performance using Mean Absolute Error (MAE) and R² score.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data handling and numerical operations |
| Scikit-learn | Model training and evaluation |
| Random Forest Regression | Price prediction |
| FastAPI | Backend REST API |
| Streamlit | Interactive web interface |
| Pydantic | API input validation |
| Joblib | Model serialization |
| Uvicorn | API server |

##  Machine Learning Workflow

1. Load the California Housing dataset.
2. Separate the input features and target variable.
3. Split the dataset into training and testing sets.
4. Train a `RandomForestRegressor`.
5. Evaluate the model using MAE and R².
6. Save the trained model and feature names using Joblib.
7. Load the saved model in the FastAPI backend and Streamlit application.
8. Generate predictions for individual properties or uploaded CSV files.

##  Input Features

The model uses eight input features:

| Feature | Description |
|---|---|
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

**Target:** Median house value from the California Housing dataset.

> **Note:** The original Scikit-learn dataset expresses target values in units of $100,000. If the application multiplies predictions by 100,000, the displayed result is in US dollars.

## 📁 Project Structure

```text
california-house-price-prediction/
├── app.py                    # Streamlit application
├── main.py                   # FastAPI backend
├── train_model.py            # Model training script (if included)
├── house_model.joblib        # Saved trained model
├── house_features.joblib     # Saved feature metadata
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from Git
└── README.md                 # Project documentation
```

*Adjust the structure to match the actual files in your repository. Include `train_model.py` only if you have created it.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd california-house-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure that `requirements.txt` includes the libraries required by your application, such as:

```text
fastapi
uvicorn[standard]
pandas
numpy
scikit-learn
joblib
python-multipart
streamlit
```

### 4. Prepare the trained model

Make sure these files are available in the project directory:

```text
house_model.joblib
house_features.joblib
```

If they are not included in the repository, run your model training script to generate them, or obtain them from your project's documented model-generation process.

### 5. Run the FastAPI backend

```bash
python -m uvicorn main:app --reload
```

Open the following URLs:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 6. Run the Streamlit application

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open it in your browser to use the application.

##  API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API welcome or status response |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict a single house price |
| `POST` | `/predict-file` | Process a CSV file and return predictions |

## Example: Single Prediction

Send a `POST` request to `/predict` with JSON data containing all eight required features.

```json
{
  "MedInc": 5.2,
  "HouseAge": 25.0,
  "AveRooms": 5.5,
  "AveBedrms": 1.1,
  "Population": 1200.0,
  "AveOccup": 3.0,
  "Latitude": 37.5,
  "Longitude": -122.2
}
```

The API returns a predicted house value. The exact response format depends on the implementation in `main.py`.

## 📈 Model Evaluation

The model is evaluated using:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **R² Score:** Measures how well the model explains variation in the target values.

Add your actual evaluation results here after running the training script. Do not report unverified accuracy or performance numbers.

##  Best Practices

- Keep your virtual environment out of Git by adding `.venv/` to `.gitignore`.
- Do not commit API keys, passwords, or other secrets.
- Keep the model file and feature metadata consistent with the training code.
- Validate incoming data before making predictions.
- Use a reproducible train-test split and record the evaluation results.

##  Future Improvements

- Deploy the API and web interface to cloud hosting.
- Add automated tests for API endpoints and model predictions.
- Improve model performance through hyperparameter tuning and cross-validation.
- Add model monitoring and structured logging.
- Add CI/CD automation for testing and deployment.

##  Author

**MD Azhar Sohail**

- GitHub: https://github.com/mdazharsohail452-oss
- LinkedIn: https://www.linkedin.com/in/md-azhar-sohail-167aa02a4

---

*Built to demonstrate practical Machine Learning, API development, and full-stack application integration.*
