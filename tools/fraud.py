"""XGBoost fraud scoring with SHAP feature contributions."""
import os
from pathlib import Path
from typing import Any

_model: Any = None
_explainer: Any = None
_loaded_path: str | None = None
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_model() -> tuple[Any, Any]:
    global _model, _explainer, _loaded_path
    model_path = Path(os.environ.get("FRAUD_MODEL_PATH", "models/fraud_xgb.json"))
    if not model_path.is_absolute():
        model_path = _PROJECT_ROOT / model_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Fraud model not found at {model_path}. Train one with scripts/train_fraud_model.py."
        )
    if _model is None or _loaded_path != str(model_path):
        try:
            import shap
            from xgboost import XGBClassifier
        except ImportError as exc:
            missing = exc.name or "xgboost or shap"
            raise RuntimeError(
                "Fraud inference dependencies are unavailable in this MCP "
                f"server's Python environment (missing {missing!r}). Install them "
                "there with `python -m pip install -r requirements-ml.txt`."
            ) from exc
        model = XGBClassifier()
        model.load_model(str(model_path))
        _model = model
        _explainer = shap.TreeExplainer(model)
        _loaded_path = str(model_path)
    return _model, _explainer


def score_transaction(amount: float, hour: int, country: str,
                      is_new_device: bool = False) -> dict:
    """Return a calibrated model score and the five strongest SHAP contributions."""
    model, explainer = _load_model()
    feature_names = ["amount", "hour", "foreign_country", "is_new_device"]
    values = {
        "amount": float(amount),
        "hour": float(hour),
        "foreign_country": float(country.strip().upper() not in ("IN", "INDIA")),
        "is_new_device": float(is_new_device),
    }
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        model_features = getattr(model.get_booster(), "feature_names", None)
    if model_features is not None and len(model_features) > 0:
        feature_names = list(model_features)
        missing = [name for name in feature_names if name not in values]
        if missing:
            raise ValueError(f"Fraud model expects unsupported features: {missing}")
    row = [[values[name] for name in feature_names]]
    probabilities = model.predict_proba(row)[0]
    positive_index = list(model.classes_).index(1) if hasattr(model, "classes_") else 1
    score = float(probabilities[positive_index])

    contributions = explainer.shap_values(row)
    if isinstance(contributions, list) and len(contributions) == len(model.classes_):
        contributions = contributions[positive_index]
    contributions = contributions[0]
    top_indexes = sorted(range(len(contributions)), key=lambda index: abs(float(contributions[index])), reverse=True)[:5]
    reasons = [
        {
            "feature": feature_names[index],
            "contribution": round(float(contributions[index]), 6),
            "direction": "increases risk" if contributions[index] > 0 else "decreases risk",
        }
        for index in top_indexes
    ]

    high_threshold = float(os.environ.get("FRAUD_HIGH_THRESHOLD", "0.7"))
    medium_threshold = float(os.environ.get("FRAUD_MEDIUM_THRESHOLD", "0.3"))
    if not 0 <= medium_threshold < high_threshold <= 1:
        raise ValueError("Fraud thresholds must satisfy 0 <= medium < high <= 1.")
    label = "HIGH" if score >= high_threshold else "MEDIUM" if score >= medium_threshold else "LOW"
    return {"risk_score": round(score, 6), "label": label, "reasons": reasons}
