
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from .inference import predict, load_artifacts, MODEL_PATH
import os
import logging

app = FastAPI(title='Symptom Diagnosis API', version='1.0.0')


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(..., description='List of symptom strings')

    @validator('symptoms')
    def symptoms_must_not_be_empty(cls, v):
        if not v or not isinstance(v, list) or not any(s.strip() for s in v):
            raise ValueError('symptoms list cannot be empty')
        return v

class TopDisease(BaseModel):
    disease: str
    confidence: Optional[float]

class PredictResponse(BaseModel):
    disease: Optional[str]
    confidence: Optional[float]
    severity: str
    stage: str
    input_symptoms: List[str]
    unknown_symptoms: List[str]
    known_symptoms_fraction: float
    model_accuracy: float
    top_diseases: List[TopDisease]

@app.on_event('startup')
async def verify_model():
    if not MODEL_PATH.exists():
        raise RuntimeError('Model not found. Train the model first.')
    load_artifacts()

@app.post('/predict', response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest, top_n: int = Query(1, ge=1, le=5, description="Number of top diseases to return")):
    try:
        result = predict(req.symptoms, top_n=top_n)
        return result
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.get('/ready')
async def ready():
    # Check if model is loaded
    try:
        load_artifacts()
        return {'ready': True}
    except Exception:
        return {'ready': False}

# For local dev convenience
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
