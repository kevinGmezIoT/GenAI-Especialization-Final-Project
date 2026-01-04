from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import asyncio
import os

# Enable LangChain/LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.predict import load_model, make_prediction
from models.enrich_inference import generate_inference_description
from monitoring.logger import log_inference
from monitoring.run_monitoring import run_monitoring_suite
from langsmith import traceable

@traceable(name="Credit Risk Assessment")
def run_assessment_logic(model, input_data):
    # 1. Generate Description via LangChain
    description = generate_inference_description(input_data)
    
    # 2. ML Inference
    prediction, probability = make_prediction(model, input_data)
    
    return prediction, probability, description

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
    description: str


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
            "monitoring": "/monitoring (GET)",
            "documentation": "/docs"
        }
    }

@app.get("/monitoring")
async def get_monitoring_report():
    """
    Returns the current data and prediction drift report.
    """
    report = run_monitoring_suite()
    return report

# ============================================================
# 4. Endpoint único solicitado
# ============================================================

@app.post("/call_model", response_model=ModelResponse)
async def call_model(request: ModelRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        # User input data (respecting the spaces in keys for the model)
        input_data = request.model_dump(by_alias=True)
        
        # Unified flow with LangSmith tracing
        prediction, probability, description = run_assessment_logic(model, input_data)

        # 3. Log for Monitoring (Drift detection)
        log_inference(input_data, prediction)

        return ModelResponse(
            prediction_label=prediction.lower().strip(),
            probability=round(probability, 2),
            description=description
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
