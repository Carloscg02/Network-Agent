import joblib
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "rf_anomaly_detector.joblib"

class AnomalyDetector:
    def __init__(self, path_model=model_path):
        self.detector = joblib.load(path_model)
        self.model = self.detector["model"]
        self.scaler = self.detector["scaler"]
        self.features = self.scaler.feature_names_in_

    def predict(self, df):
        # Asegurarse de seleccionar las columnas correctas
        X = df[self.features]
        X_scaled_np = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled_np, columns = self.features)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = self.model.predict(X_scaled)
        return probs, preds
