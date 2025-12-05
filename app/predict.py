# SUML Delivery project series
# By Hryhorii Hrymailo s27157

import joblib
import numpy as np
from pathlib import Path


MODEL_PATH = Path(__file__).parent / "model.joblib"
_model = joblib.load(MODEL_PATH)

def predict(features: list[float]) -> str:
    data = np.array(features).reshape(1, -1)
    prediction = _model.predict(data)[0]
    return prediction
