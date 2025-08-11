# Symptom Diagnosis Model
# Training data: https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-datase
Pipeline to train a classifier mapping sets of symptoms to likely diseases, and serve predictions via a REST API (FastAPI) with Docker.

> DISCLAIMER: This is an ML pipeline using heuristic severity/stage logic and not a medically validated diagnostic tool.

## Project Layout
```
├── data/
│   └── DiseaseAndSymptoms.csv
├── src/
│   ├── train.py          # training script (creates models/ directory)
│   ├── inference.py      # prediction utilities + heuristic severity/stage
│   └── api.py            # FastAPI app exposing /predict and /health
├── models/               # (created after training) model + meta
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Training
### Local (no Docker)
```powershell
pip install -r requirements.txt
python .\src\train.py
```
Artifacts:
```
models/
  ├── disease_model.joblib   (RandomForest + MultiLabelBinarizer)
  └── meta.json              (metrics & symptom vocabulary)
```

### Inside Docker
```powershell
docker build -t diagnosis-api .
docker run --rm -p 8000:8000 diagnosis-api
```
Force retrain:
```powershell
docker run --rm -e FORCE_RETRAIN=1 -p 8000:8000 diagnosis-api
```

## Serving the API
### With docker-compose
docker compose up --build -d
```powershell
docker compose up --build -d
```
Force retrain with compose (one-off):
```powershell
FORCE_RETRAIN=1 docker compose up --build -d
Then visit http://localhost:8000/docs for interactive Swagger UI.

### Direct (no Docker)
```powershell
uvicorn src.api:app --reload --port 8000
```

## Request Example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms":["itching","skin rash","vomiting"]}'
```
Response (example):
```json
{
  "disease": "Fungal infection",
  "confidence": 0.82,
  "severity": "Medium",
  "stage": "Progressed",
  "input_symptoms": ["itching","skin_rash","vomiting"],
  "known_symptoms_fraction": 0.67,
  "model_accuracy": 0.94
}
```


## Severity & Stage Mappings (Medically Informed)
- **Severity**: Each symptom is assigned a severity score (1 = mild/common, 2 = significant, 3 = severe/life-threatening) based on clinical relevance and medical guidelines. The total severity score for a case is mapped to Low, Medium, or High.
- **Stage**: The number of provided symptoms is used to estimate disease stage: Early (few symptoms), Progressed (moderate), Advanced (many symptoms). This is a common approach in clinical triage and is adjustable for your use case.

See `src/inference.py` for the full list of symptoms and their severity scores.