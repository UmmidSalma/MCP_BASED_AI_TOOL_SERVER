"""Generate clearly synthetic fraud data for demonstrating the local model pipeline."""
import argparse
import csv
import random
from pathlib import Path


FIELDS = ["amount", "hour", "foreign_country", "is_new_device", "label"]


def generate_rows(row_count: int, seed: int) -> list[dict[str, int | float]]:
    """Create reproducible simulated transactions with intentionally synthetic labels."""
    if row_count < 1:
        raise ValueError("row_count must be at least one")

    rng = random.Random(seed)
    rows = []
    for _ in range(row_count):
        amount = min(rng.lognormvariate(7.2, 1.1), 1_000_000)
        hour = rng.randrange(24)
        foreign_country = int(rng.random() < 0.18)
        is_new_device = int(rng.random() < 0.14)

        probability = 0.015
        probability += 0.10 if amount >= 12_000 else 0
        probability += 0.15 if amount >= 40_000 else 0
        probability += 0.12 if hour < 5 else 0
        probability += 0.13 if foreign_country else 0
        probability += 0.16 if is_new_device else 0
        probability += 0.18 if foreign_country and is_new_device else 0
        probability += 0.10 if amount >= 40_000 and is_new_device else 0

        rows.append({
            "amount": round(amount, 2),
            "hour": hour,
            "foreign_country": foreign_country,
            "is_new_device": is_new_device,
            "label": int(rng.random() < min(probability, 0.95)),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic-fraud"))
    parser.add_argument("--train-rows", type=int, default=10_000)
    parser.add_argument("--holdout-rows", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_rows < 1 or args.holdout_rows < 1:
        parser.error("--train-rows and --holdout-rows must both be positive")

    train_path = args.output_dir / "train.csv"
    holdout_path = args.output_dir / "holdout.csv"
    write_csv(train_path, generate_rows(args.train_rows, args.seed))
    write_csv(holdout_path, generate_rows(args.holdout_rows, args.seed + 1))
    print(f"Wrote {args.train_rows} synthetic training rows to {train_path}")
    print(f"Wrote {args.holdout_rows} synthetic holdout rows to {holdout_path}")
    print("These simulated labels are for software demonstration only, not fraud research or deployment.")


if __name__ == "__main__":
    main()
