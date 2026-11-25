"""Train the project's binary XGBoost fraud model from a labeled CSV."""
import argparse
from pathlib import Path


FEATURES = ["amount", "hour", "foreign_country", "is_new_device"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="CSV containing the four model features and a label")
    parser.add_argument("--target", default="label", help="Binary target column (default: label)")
    parser.add_argument("--output", type=Path, default=Path("models/fraud_xgb.json"))
    args = parser.parse_args()

    try:
        import pandas as pd
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit("Install the optional ML dependencies with: pip install -r requirements-ml.txt") from exc

    data = pd.read_csv(args.dataset)
    missing = [column for column in FEATURES + [args.target] if column not in data.columns]
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {', '.join(missing)}")
    if data.empty:
        raise SystemExit("Training dataset must contain at least one row.")
    if data[FEATURES + [args.target]].isna().any().any():
        raise SystemExit("Training data cannot contain missing feature or target values.")
    labels = data[args.target]
    if set(labels.unique()) != {0, 1}:
        raise SystemExit("Target must contain both classes encoded as 0 (legitimate) and 1 (fraud).")

    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(data[FEATURES], labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))
    print(f"Saved model to {args.output}; training rows: {len(data)}")


if __name__ == "__main__":
    main()