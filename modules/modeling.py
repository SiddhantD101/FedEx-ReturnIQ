import joblib
from pathlib import Path

MODEL_PATH = Path("models/returniq_model.pkl")

def load_model(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception as e:
        print("⚠️ Model load error:", e)
        return None
