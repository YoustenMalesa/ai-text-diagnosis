
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import datetime

DATA_PATH = Path('data/DiseaseAndSymptoms.csv')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'disease_model.joblib'
META_PATH = MODEL_DIR / 'meta.json'

SEVERITY_MAP = {
    # Example severity heuristics (counts of symptoms or domain knowledge). Placeholder mapping.
    # In real scenario this would come from medical domain knowledge.
}

STAGE_MAP = {
    # Placeholder for potential future use
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names (strip spaces) and values (strip, lower, replace spaces with underscores inside symptoms)
    symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]
    # Clean disease names
    df['Disease'] = df['Disease'].str.strip()
    # Clean symptoms
    for c in symptom_cols:
        df[c] = (df[c].astype(str)
                        .str.strip()
                        .str.replace(' ', '_')
                        .str.replace('__', '_')
                        .replace({'nan': np.nan, '': np.nan}))
    return df


def build_training_rows(df: pd.DataFrame):
    symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]
    records = []
    for _, row in df.iterrows():
        symptoms = [s for s in row[symptom_cols].tolist() if isinstance(s, str) and s]
        records.append({'disease': row['Disease'], 'symptoms': symptoms})
    train_df = pd.DataFrame(records)
    return train_df


def train_model(train_df: pd.DataFrame):
    # Build vocabulary
    all_symptoms = sorted({s for lst in train_df['symptoms'] for s in lst})
    mlb = MultiLabelBinarizer(classes=all_symptoms)
    X = mlb.fit_transform(train_df['symptoms'])
    y = train_df['disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    meta = {
        'symptom_vocabulary': all_symptoms,
        'train_size': int(X_train.shape[0]),
        'test_size': int(X_test.shape[0]),
        'accuracy': acc,
        'classification_report': report,
        'model_version': datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S'),
        'trained_at_utc': datetime.datetime.utcnow().isoformat() + 'Z'
    }

    joblib.dump({'model': clf, 'mlb': mlb}, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f'Model saved to {MODEL_PATH}. Accuracy: {acc:.4f}')


def main():
    df = load_data(DATA_PATH)
    train_df = build_training_rows(df)
    train_model(train_df)


if __name__ == '__main__':
    main()
