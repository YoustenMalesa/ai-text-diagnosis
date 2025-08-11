# Symptom Diagnosis Model

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
Training now happens automatically at container startup if the model is missing (or if you set `FORCE_RETRAIN=1`).
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

## Severity & Stage Heuristics
- Severity: weighted sum of symptom severities; mapped to Low / Medium / High.
- Stage: number of provided symptoms (Early / Progressed / Advanced).
These are placeholders; refine with domain expertise.

## Extending
1. Add proper medical severity & stage mappings based on validated guidelines.
2. Add unit tests (e.g., for normalization, prediction output schema).
3. Add versioning for models (timestamped filenames + registry).
4. Add monitoring (request logging, prediction drift checks).

## Security & Production Notes
- Add authentication (API key, OAuth, or reverse proxy) before public exposure.
- Use HTTPS in production (terminate TLS at proxy or container).
- Rate limiting and input validation (already ensures non-empty list).
- Set resource limits in docker-compose for memory/CPU.
- Monitor logs (stdout) and add alerting for errors.
- Use health (`/health`) and readiness (`/ready`) endpoints for orchestration.
- Model version and training time are included in metadata for traceability.
- Consider a confidence threshold and return multiple top diagnoses (see `top_diseases`).
- Integrate structured logging (JSON) for observability and compliance.
- Add unit/integration tests and CI/CD for automated deployment.

## License
Provided as-is for educational purposes.
