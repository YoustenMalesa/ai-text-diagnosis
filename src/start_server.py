import os
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

MODEL_PATH = Path('models/disease_model.joblib')
META_PATH = Path('models/meta.json')


def ensure_model():
    force = os.getenv('FORCE_RETRAIN', '0') == '1'
    if force or not MODEL_PATH.exists() or not META_PATH.exists():
        logging.info('Training model (force=%s, model_exists=%s)...', force, MODEL_PATH.exists())
        # Lazy import to avoid cost if already trained
        from . import train
        train.main()
    else:
        logging.info('Model already present, skipping training.')


def main():
    ensure_model()
    import uvicorn
    uvicorn.run('src.api:app', host='0.0.0.0', port=int(os.getenv('PORT', '8000')))


if __name__ == '__main__':
    main()
