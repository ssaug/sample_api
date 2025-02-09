from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Initialize FastAPI
app = FastAPI()

# Model path
MODEL_PATH = "./model/iris_classifier.pkl"

# Load the trained model if it exists
clf = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        clf = pickle.load(file)

# Pydantic model for input validation
class PredictionInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

# Route for checking API status
@app.get("/get_status")
async def get_status():
    return {
        "status": "API is running",
        "training_data_split": "70%",
        "test_data_split": "30%"
    }

# Route for making predictions
@app.post("/prediction")
async def prediction(data: PredictionInput):
    try:
        if clf is None:
            raise HTTPException(status_code=400, detail="Model not loaded. Train the model first using /training")

        # Extract input features
        input_data = np.array([[data.SepalLengthCm, data.SepalWidthCm, data.PetalLengthCm, data.PetalWidthCm]])

        # Make prediction
        prediction = clf.predict(input_data)
        return {"predicted_class": int(prediction[0])}  # Convert NumPy int to Python int for JSON serialization

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route for retraining the model
@app.get("/training")
async def training():
    try:
        global clf
        message = train_model()

        # Reload the trained model
        with open(MODEL_PATH, "rb") as file:
            clf = pickle.load(file)

        return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
