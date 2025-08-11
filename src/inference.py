from pathlib import Path
import joblib
import json
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger("diagnosis.inference")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

MODEL_PATH = Path('models/disease_model.joblib')
META_PATH = Path('models/meta.json')

SEVERITY_RULES = {
    # Medically-informed severity: 3=severe, 2=moderate, 1=mild/common
    'abdominal_pain': 2,
    'abnormal_menstruation': 1,
    'acidity': 1,
    'acute_liver_failure': 3,
    'altered_sensorium': 3,
    'anxiety': 1,
    'back_pain': 1,
    'belly_pain': 2,
    'blackheads': 1,
    'bladder_discomfort': 1,
    'blister': 1,
    'blood_in_sputum': 3,
    'bloody_stool': 3,
    'blurred_and_distorted_vision': 2,
    'burning_micturition': 2,
    'chest_pain': 3,
    'chills': 1,
    'coma': 3,
    'congestion': 1,
    'constipation': 1,
    'continuous_feel_of_urine': 1,
    'continuous_sneezing': 1,
    'cramps': 1,
    'dark_urine': 2,
    'dehydration': 3,
    'depression': 2,
    'diarrhoea': 2,
    'dischromic_patches': 1,
    'dizziness': 2,
    'distention_of_abdomen': 2,
    'drying_and_tingling_lips': 1,
    'enlarged_thyroid': 2,
    'excessive_hunger': 1,
    'extra_marital_contacts': 1,
    'family_history': 1,
    'fast_heart_rate': 2,
    'fatigue': 1,
    'foul_smell_of_urine': 1,
    'fluid_overload': 3,
    'headache': 1,
    'high_fever': 2,
    'hip_joint_pain': 2,
    'history_of_alcohol_consumption': 1,
    'increased_appetite': 1,
    'indigestion': 1,
    'inflammatory_nails': 1,
    'internal_itching': 1,
    'irregular_sugar_level': 2,
    'irritability': 1,
    'irritation_in_anus': 1,
    'itching': 1,
    'joint_pain': 2,
    'knee_pain': 2,
    'lack_of_concentration': 1,
    'lethargy': 1,
    'loss_of_appetite': 2,
    'loss_of_balance': 2,
    'loss_of_smell': 1,
    'malaise': 1,
    'mild_fever': 1,
    'movement_stiffness': 1,
    'mucoid_sputum': 1,
    'muscle_pain': 1,
    'muscle_wasting': 2,
    'muscle_weakness': 2,
    'nausea': 1,
    'neck_pain': 1,
    'nodal_skin_eruptions': 1,
    'obesity': 1,
    'pain_during_bowel_movements': 2,
    'pain_in_anal_region': 2,
    'pain_behind_the_eyes': 2,
    'painful_walking': 2,
    'palpitations': 2,
    'patches_in_throat': 2,
    'passage_of_gases': 1,
    'phlegm': 1,
    'polyuria': 2,
    'prominent_veins_on_calf': 1,
    'puffy_face_and_eyes': 1,
    'pus_filled_pimples': 1,
    'red_sore_around_nose': 1,
    'red_spots_over_body': 1,
    'receiving_blood_transfusion': 2,
    'receiving_unsterile_injections': 2,
    'restlessness': 1,
    'runny_nose': 1,
    'rusty_sputum': 2,
    'scurring': 1,
    'shivering': 1,
    'silver_like_dusting': 1,
    'sinus_pressure': 1,
    'skin_peeling': 1,
    'skin_rash': 1,
    'slurred_speech': 2,
    'small_dents_in_nails': 1,
    'spinning_movements': 2,
    'spotting_urination': 1,
    'stiff_neck': 2,
    'stomach_bleeding': 3,
    'stomach_pain': 2,
    'sunken_eyes': 2,
    'sweating': 1,
    'swelled_lymph_nodes': 1,
    'swelling_joints': 2,
    'swelling_of_stomach': 2,
    'swollen_blood_vessels': 1,
    'swollen_extremeties': 2,
    'swollen_legs': 2,
    'throat_irritation': 1,
    'toxic_look_(typhos)': 3,
    'ulcers_on_tongue': 2,
    'unsteadiness': 2,
    'visual_disturbances': 2,
    'vomiting': 2,
    'watering_from_eyes': 1,
    'weakness_in_limbs': 2,
    'weight_gain': 1,
    'weight_loss': 2,
    'yellow_crust_ooze': 1,
    'yellow_urine': 1,
    'yellowish_skin': 2,
    'yellowing_of_eyes': 2,
}


STAGE_RULES = {
    # naive stage mapping - more symptoms => later stage
    'early': (0, 2),
    'progressed': (3, 5),
    'advanced': (6, 1000)
}


def load_artifacts():
    logger.info("Loading model artifacts from %s", MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    logger.info("Loaded model with accuracy %.3f", meta.get('accuracy', 0))
    return bundle['model'], bundle['mlb'], meta


def predict(symptoms: List[str], top_n: int = 1) -> Dict:
    model, mlb, meta = load_artifacts()
    vocab = set(mlb.classes_)
    # normalize incoming symptoms to match training representation
    cleaned = []
    unknown = []
    for s in symptoms:
        s_norm = s.strip().lower().replace(' ', '_')
        if s_norm in vocab:
            cleaned.append(s_norm)
        else:
            unknown.append(s_norm)

    logger.info("Received symptoms: %s | Cleaned: %s | Unknown: %s", symptoms, cleaned, unknown)

    # Handle empty after filtering
    if not cleaned:
        logger.warning("No known symptoms provided. Returning null diagnosis.")
        return {
            'disease': None,
            'confidence': None,
            'severity': 'Low',
            'stage': 'Early',
            'input_symptoms': [],
            'unknown_symptoms': unknown,
            'known_symptoms_fraction': 0.0,
            'model_accuracy': meta.get('accuracy'),
            'top_diseases': []
        }

    X = mlb.transform([cleaned])
    proba = None
    top_diseases = []
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        top_indices = np.argsort(proba)[::-1][:top_n]
        for idx in top_indices:
            top_diseases.append({
                'disease': model.classes_[idx],
                'confidence': float(proba[idx])
            })
        disease = model.classes_[top_indices[0]]
        confidence = float(proba[top_indices[0]])
    else:
        disease = model.predict(X)[0]
        confidence = None
        top_diseases.append({'disease': disease, 'confidence': None})

    severity_score = sum(SEVERITY_RULES.get(s, 1) for s in cleaned)
    if severity_score <= 3:
        severity = 'Low'
    elif severity_score <= 7:
        severity = 'Medium'
    else:
        severity = 'High'

    count = len(cleaned)
    stage = 'Early'
    for label, (lo, hi) in STAGE_RULES.items():
        if lo <= count <= hi:
            stage = label.title()
            break

    logger.info("Diagnosis: %s | Confidence: %s | Severity: %s | Stage: %s", disease, confidence, severity, stage)

    return {
        'disease': disease,
        'confidence': confidence,
        'severity': severity,
        'stage': stage,
        'input_symptoms': cleaned,
        'unknown_symptoms': unknown,
        'known_symptoms_fraction': float((X.sum() / len(cleaned)) if cleaned else 0.0),
        'model_accuracy': meta.get('accuracy'),
        'top_diseases': top_diseases
    }

if __name__ == '__main__':
    print(predict(['itching','skin rash']))
