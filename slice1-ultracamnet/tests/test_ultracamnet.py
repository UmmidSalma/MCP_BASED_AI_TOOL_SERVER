from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_tool_path = Path(__file__).resolve().parents[1] / "tools" / "ultracamnet.py"
_spec = spec_from_file_location("slice_ultracamnet", _tool_path)
assert _spec is not None and _spec.loader is not None
_slice_tool = module_from_spec(_spec)
_spec.loader.exec_module(_slice_tool)


def test_ultracamnet_predict_returns_placeholder():
    result = _slice_tool.predict_image("not-used-by-the-stub")

    assert result == {
        "prediction": "stub",
        "confidence": 0.0,
        "note": "Placeholder output. Model not connected.",
        "disclaimer": "Research demo only. Not a medical diagnosis.",
    }
