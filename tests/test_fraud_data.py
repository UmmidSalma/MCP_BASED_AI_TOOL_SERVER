import csv

import pytest

from scripts.generate_synthetic_fraud_data import FIELDS, generate_rows, write_csv


def test_synthetic_rows_are_reproducible_and_have_expected_schema():
    rows = generate_rows(100, 42)

    assert rows == generate_rows(100, 42)
    assert set(rows[0]) == set(FIELDS)
    assert {row["label"] for row in rows} == {0, 1}
    assert all(0 <= row["hour"] <= 23 for row in rows)


def test_synthetic_generator_requires_positive_row_count():
    with pytest.raises(ValueError, match="at least one"):
        generate_rows(0, 42)


def test_write_csv_writes_expected_header_and_rows(tmp_path):
    path = tmp_path / "train.csv"
    rows = generate_rows(10, 42)

    write_csv(path, rows)

    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == FIELDS
        assert len(list(reader)) == 10
