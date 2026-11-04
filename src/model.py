"""Production model loading and prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import joblib
import numpy as np
import pandas as pd

try:
    from .preprocessing import load_preprocessing_artifacts, transform_job_posting
except ImportError:
    from preprocessing import load_preprocessing_artifacts, transform_job_posting


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


def load_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    """Load the saved production model. This function never trains a model."""
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Saved model artifact was not found: {path}")
    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise TypeError(f"Saved artifact at {path} is not a prediction model.")
    classes = getattr(model, "classes_", None)
    if classes is not None and set(classes) != {0, 1}:
        raise ValueError(
            f"Saved model must contain binary classes 0 and 1; found {classes!r}."
        )
    return model


def predict_job_posting(
    job_posting: Union[Mapping[str, Any], pd.DataFrame, pd.Series],
    model: Optional[Any] = None,
    preprocessing_artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Preprocess one posting and return its prediction and model score."""
    if model is None:
        model = load_model()
    matrix = transform_job_posting(job_posting, artifacts=preprocessing_artifacts)
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise ValueError(
            f"Preprocessing produced {matrix.shape[1]} features, but the loaded "
            f"model expects {expected_features}. Verify that model and preprocessing "
            "artifacts come from the same training run."
        )
    if matrix.shape[0] != 1:
        raise ValueError("Expected exactly one transformed job posting.")

    prediction = model.predict(matrix)[0]
    if prediction not in (0, 1):
        raise ValueError(f"Model returned an unsupported class label: {prediction!r}.")
    prediction = int(prediction)

    decision_score = None
    if hasattr(model, "decision_function"):
        decision_score = float(np.asarray(model.decision_function(matrix)).reshape(-1)[0])
        if not np.isfinite(decision_score):
            raise ValueError("Model returned a non-finite decision margin.")

    return {
        "prediction": prediction,
        "label": "Potentially Fraudulent Job" if prediction == 1 else "Legitimate Job",
        "decision_score": decision_score,
        "probability": None,
        "feature_count": int(matrix.shape[1]),
        "feature_vector": matrix,
    }


__all__ = ["MODEL_PATH", "load_model", "predict_job_posting"]
