from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.predict import load_model, make_prediction

# ============================================================
# 1. Request schema (respeta claves con espacios)
# ============================================================

class ModelRequest(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: str = Field(alias="Saving accounts")
    Checking_account: str = Field(alias="Checking account")
    Credit_amount: float = Field(alias="Credit amount")
    Duration: int
    Purpose: str

    model_config = {
        "populate_by_name": True,
        "extra": "ignore"
    }


# ============================================================
# 2. Response schema (SOLO 2 CAMPOS)
# ============================================================

class ModelResponse(BaseModel):
    prediction_label: Literal["good risk", "bad risk"]
    probability: float


# ============================================================
# 3. FastAPI app
# ============================================================

app = FastAPI(
    title="Credit Risk Model API",
    version="1.0.0"
)

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model_path = os.path.join(os.path.dirname(__file__), "models", "credit_risk_model.joblib")
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print(f"Warning: Model not found at {model_path}")

@app.get("/")
async def root():
    return {
        "message": "Credit Risk Model API is running",
        "endpoints": {
            "prediction": "/call_model (POST)",
            "documentation": "/docs"
        }
    }

# ============================================================
# 4. Endpoint único solicitado
# ============================================================

@app.post("/call_model", response_model=ModelResponse)
async def call_model(request: ModelRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        # User input data (respecting the spaces in keys for the model)
        # We use .model_dump() with by_alias=True to get the keys with spaces
        input_data = request.model_dump(by_alias=True)
        
        # Inference
        prediction, probability = make_prediction(model, input_data)

        return ModelResponse(
            prediction_label=prediction.lower(),
            probability=round(probability, 2)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
