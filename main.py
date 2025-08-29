from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd  # Added pandas for DataFrame

models = {}

# Load model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_pipeline = joblib.load("addiction_score_pipeline.tuned.joblib")
    models["addiction_pipeline"] = model_pipeline
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Pydantic model for request body
class AddictionRequest(BaseModel):
    Age: int
    Gender: str
    Academic_Level: str
    Avg_Daily_Usage_Hours: float
    Most_Used_Platform: str
    Affects_Academic_Performance: str
    Sleep_Hours_Per_Night: float
    Mental_Health_Score: int
    Relationship_Status: str
    Conflicts_Over_Social_Media: int

# Function to categorize addiction score
def categorize_addiction(score: float) -> str:
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Moderate"
    else:
        return "High"

@app.post("/predict")
async def predict_addiction(data: AddictionRequest):
    model = models.get("addiction_pipeline")
    if not model:
        return {"status_code": 500, "message": "Model not loaded!"}

    # Convert incoming JSON to a DataFrame with a single row
    input_data = pd.DataFrame([data.dict()])

    # Predict
    prediction = model.predict(input_data)[0]
    category = categorize_addiction(prediction)

    return {
        "status_code": 200,
        "predicted_score": round(float(prediction), 2),
        "category": category
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
