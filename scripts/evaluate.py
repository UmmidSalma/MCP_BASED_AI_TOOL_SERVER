"""Evaluate a saved fraud model or benchmark one locally callable tool."""
import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_fraud(dataset: Path, target: str) -> dict:
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    from tools.fraud import _load_model

    features = ["amount", "hour", "foreign_country", "is_new_device"]
    data = pd.read_csv(dataset)
    missing = [column for column in features + [target] if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("Evaluation dataset must contain at least one row.")
    if data[features + [target]].isna().any().any():
        raise ValueError("Evaluation data cannot contain missing feature or target values.")
    test_x = data[features]
    test_y = data[target]
    if set(test_y.unique()) != {0, 1}:
        raise ValueError("Evaluation target must contain both binary classes 0 and 1.")
    model, _ = _load_model()
    predicted = model.predict(test_x)
    probabilities = model.predict_proba(test_x)[:, list(model.classes_).index(1)]
    return {
        "rows": len(test_y),
        "accuracy": float(accuracy_score(test_y, predicted)),
        "f1": float(f1_score(test_y, predicted)),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
    }


def benchmark_tool(tool_name: str, arguments: dict, iterations: int) -> dict:
    from server import call_tool

    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        call_tool(tool_name, arguments)
        durations.append((time.perf_counter() - started) * 1000)
    ordered = sorted(durations)
    return {
        "tool": tool_name,
        "iterations": iterations,
        "mean_ms": round(statistics.mean(durations), 3),
        "median_ms": round(statistics.median(durations), 3),
        "p95_ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target", default="label")
    parser.add_argument("--tool", default="ping")
    parser.add_argument("--arguments", default="{}", help="JSON object for the selected tool")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least one")
    rate_limit = max(1, int(os.environ.get("RATE_LIMIT_CALLS", "60")))
    if args.iterations > rate_limit:
        parser.error(
            f"--iterations cannot exceed RATE_LIMIT_CALLS ({rate_limit}); "
            "raise the limit explicitly for a longer benchmark"
        )
    try:
        tool_arguments = json.loads(args.arguments)
    except json.JSONDecodeError as exc:
        parser.error(f"--arguments must be valid JSON: {exc}")
    if not isinstance(tool_arguments, dict):
        parser.error("--arguments must be a JSON object")
    results = {"latency": benchmark_tool(args.tool, tool_arguments, args.iterations)}
    if args.dataset:
        results["fraud_model"] = evaluate_fraud(args.dataset, args.target)
    results["environment"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    output = PROJECT_ROOT / "docs" / "evaluation-results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report = PROJECT_ROOT / "docs" / "evaluation.md"
    report_text = report.read_text(encoding="utf-8")
    start_marker, end_marker = "<!-- evaluation-results:start -->", "<!-- evaluation-results:end -->"
    generated = ["## Latest Measurements", "", f"Run: {results['environment']['timestamp_utc']}", ""]
    latency = results["latency"]
    generated.extend([
        f"- Tool: `{latency['tool']}` ({latency['iterations']} calls)",
        f"- Latency: mean {latency['mean_ms']} ms, median {latency['median_ms']} ms, p95 {latency['p95_ms']} ms",
    ])
    if "fraud_model" in results:
        metrics = results["fraud_model"]
        generated.extend([
            f"- Fraud holdout: {metrics['rows']} rows",
            f"- Accuracy: {metrics['accuracy']:.4f}; F1: {metrics['f1']:.4f}; ROC-AUC: {metrics['roc_auc']:.4f}",
        ])
    generated.extend([f"- Environment: Python {results['environment']['python']}, `{results['environment']['platform']}`", ""])
    if start_marker in report_text and end_marker in report_text:
        before, remainder = report_text.split(start_marker, 1)
        _, after = remainder.split(end_marker, 1)
        report_text = before + start_marker + "\n" + "\n".join(generated) + end_marker + after
    report.write_text(report_text, encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()