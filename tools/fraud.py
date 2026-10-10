"""Fraud Risk Analyzer tool.

DEMO VERSION: simple rule-based scoring so the protocol works end to end.
TODO: replace score_transaction() body with your real trained model
      (e.g. joblib.load("models/fraud_model.pkl").predict_proba(...)).
"""


def score_transaction(amount: float, hour: int, country: str,
                      is_new_device: bool = False) -> dict:
    score = 0.0
    reasons = []

    if amount > 50000:
        score += 0.4
        reasons.append("very high amount")
    elif amount > 10000:
        score += 0.2
        reasons.append("high amount")

    if hour < 5 or hour > 23:
        score += 0.2
        reasons.append("unusual hour")

    if country.strip().upper() not in ("IN", "INDIA"):
        score += 0.2
        reasons.append("foreign country")

    if is_new_device:
        score += 0.2
        reasons.append("new device")

    score = min(score, 1.0)
    label = "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.3 else "LOW"
    return {"risk_score": round(score, 2), "risk_level": label, "reasons": reasons}
