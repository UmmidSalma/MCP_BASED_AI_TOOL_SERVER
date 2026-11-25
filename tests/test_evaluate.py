import sys
from types import ModuleType

import pytest

from scripts import evaluate


def test_fraud_evaluation_uses_entire_supplied_holdout(monkeypatch, tmp_path):
    class Series(list):
        def dropna(self):
            return self

        def unique(self):
            return list(set(self))

    class Frame:
        columns = ["amount", "hour", "foreign_country", "is_new_device", "label"]
        empty = False

        def __init__(self):
            self.rows = [
                {"amount": 10, "hour": 1, "foreign_country": 0, "is_new_device": 0, "label": 0},
                {"amount": 20, "hour": 2, "foreign_country": 1, "is_new_device": 1, "label": 1},
            ]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, column):
            if isinstance(column, list):
                return self
            return Series(row[column] for row in self.rows)

        class MissingValues:
            def any(self):
                return type("AnyResult", (), {"any": lambda _self: False})()

        def isna(self):
            return self.MissingValues()

        def any(self):
            return self.MissingFrame()

    class Model:
        classes_ = [0, 1]

        def predict(self, features):
            assert len(features) == 2
            return [0, 1]

        def predict_proba(self, features):
            assert len(features) == 2

            class ProbabilityMatrix:
                values = [[0.9, 0.1], [0.1, 0.9]]

                def __getitem__(self, selection):
                    rows, column = selection
                    return [self.values[index][column] for index in range(len(self.values))[rows]]

            return ProbabilityMatrix()

    pandas = ModuleType("pandas")
    pandas.read_csv = lambda _path: Frame()
    sklearn = ModuleType("sklearn")
    metrics = ModuleType("sklearn.metrics")
    metrics.accuracy_score = lambda truth, predicted: sum(a == b for a, b in zip(truth, predicted)) / len(truth)
    metrics.f1_score = lambda _truth, _predicted: 1.0
    metrics.roc_auc_score = lambda _truth, _scores: 1.0
    sklearn.metrics = metrics
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics)
    from tools import fraud

    monkeypatch.setattr(fraud, "_load_model", lambda: (Model(), None))

    result = evaluate.evaluate_fraud(tmp_path / "holdout.csv", "label")

    assert result == {"rows": 2, "accuracy": 1.0, "f1": 1.0, "roc_auc": 1.0}


def test_benchmark_rejects_non_object_arguments():
    with pytest.raises(ValueError, match="JSON object"):
        evaluate.benchmark_tool("ping", [], 1)


def test_benchmark_rejects_zero_iterations():
    with pytest.raises(ValueError, match="at least one"):
        evaluate.benchmark_tool("ping", {}, 0)
