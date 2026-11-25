import asyncio
import base64
import io
import sys
import types

import pytest
Image = pytest.importorskip("PIL.Image")

from tools import ultracamnet
from security import validation


def _encoded_image():
    image = Image.new("RGB", (12, 10), (50, 100, 150))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeProcessor:
    def __call__(self, images, return_tensors):
        assert images.mode == "RGB"
        assert return_tensors == "pt"
        return FakeBatch(pixel_values="fake pixels")

    def decode(self, _tokens, skip_special_tokens):
        assert skip_special_tokens is True
        return "  a sample image caption  "


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def generate(self, pixel_values, max_new_tokens):
        self.calls += 1
        assert pixel_values == "fake pixels"
        assert max_new_tokens == 50
        return [[1]]


@pytest.fixture
def fake_caption_model(monkeypatch, tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    monkeypatch.setenv("IMAGE_ROOT", str(image_root))
    model = FakeModel()
    monkeypatch.setattr(
        ultracamnet,
        "_load_model",
        lambda: (FakeProcessor(), model),
    )
    return model, image_root


def _write_image(path):
    path.write_bytes(base64.b64decode(_encoded_image()))
    return path


def test_invalid_base64_returns_error():
    result = asyncio.run(ultracamnet.predict_image(image_base64="not base64!"))

    assert result == {
        "error": "image_base64 must contain valid standard base64 data."
    }


def test_non_image_bytes_return_error():
    image_base64 = base64.b64encode(b"not an image").decode("ascii")

    result = asyncio.run(ultracamnet.predict_image(image_base64=image_base64))

    assert result == {"error": "The supplied data is not a supported image."}


def test_oversized_input_returns_error():
    image_base64 = base64.b64encode(b"x" * (5 * 1024 * 1024 + 1)).decode("ascii")

    result = asyncio.run(ultracamnet.predict_image(image_base64=image_base64))

    assert result == {"error": "Image exceeds the 5 MB size limit."}


def test_valid_image_returns_caption(fake_caption_model):
    model, _ = fake_caption_model

    result = asyncio.run(
        ultracamnet.predict_image(image_base64=_encoded_image())
    )

    assert result == {"caption": "a sample image caption"}
    assert model.calls == 1


def test_valid_image_path_returns_caption(fake_caption_model):
    model, image_root = fake_caption_model
    _write_image(image_root / "test.png")

    result = asyncio.run(ultracamnet.predict_image(image_path="test.png"))

    assert result == {"caption": "a sample image caption"}
    assert model.calls == 1


def test_relative_image_root_resolves_from_project_not_working_directory(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    image_root = project_root / "images"
    working_directory = tmp_path / "working"
    image_root.mkdir(parents=True)
    working_directory.mkdir()
    image_path = _write_image(image_root / "test.png")
    monkeypatch.setattr(validation, "_PROJECT_ROOT", project_root)
    monkeypatch.setenv("IMAGE_ROOT", "images")
    monkeypatch.chdir(working_directory)

    assert validation.validate_image_file("test.png") == image_path


def test_path_outside_image_root_is_rejected(fake_caption_model, tmp_path):
    _write_image(tmp_path / "outside.png")

    result = asyncio.run(
        ultracamnet.predict_image(image_path=str(tmp_path / "outside.png"))
    )

    assert "error" in result
    assert "inside the configured IMAGE_ROOT" in result["error"]


def test_path_traversal_is_rejected(fake_caption_model, tmp_path):
    _, image_root = fake_caption_model
    _write_image(tmp_path / "outside.png")

    result = asyncio.run(ultracamnet.predict_image(image_path="..\\outside.png"))

    assert "error" in result
    assert "inside the configured IMAGE_ROOT" in result["error"]


def test_nonexistent_image_path_returns_clear_error(fake_caption_model):
    result = asyncio.run(ultracamnet.predict_image(image_path="missing.png"))

    assert result == {"error": "Image file does not exist."}


def test_unsupported_image_extension_is_rejected(fake_caption_model):
    _, image_root = fake_caption_model
    _write_image(image_root / "test.gif")

    result = asyncio.run(ultracamnet.predict_image(image_path="test.gif"))

    assert "error" in result
    assert "Unsupported image type" in result["error"]


def test_oversized_image_path_is_rejected_before_reading(fake_caption_model):
    _, image_root = fake_caption_model
    (image_root / "large.png").write_bytes(b"x" * (5 * 1024 * 1024 + 1))

    result = asyncio.run(ultracamnet.predict_image(image_path="large.png"))

    assert result == {"error": "Image exceeds the 5 MB size limit."}


def test_both_image_inputs_are_rejected(fake_caption_model):
    result = asyncio.run(
        ultracamnet.predict_image(
            image_path="test.png", image_base64=_encoded_image()
        )
    )

    assert result == {"error": "Provide exactly one of image_path or image_base64."}


def test_neither_image_input_is_rejected():
    result = asyncio.run(ultracamnet.predict_image())

    assert result == {"error": "Provide exactly one of image_path or image_base64."}


def test_missing_transformers_returns_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    image_base64 = _encoded_image()

    result = asyncio.run(ultracamnet.predict_image(image_base64=image_base64))

    assert "error" in result
    assert "missing 'transformers'" in result["error"]
    assert "python -m pip install -r requirements-ml.txt" in result["error"]


def test_model_and_processor_are_loaded_once(monkeypatch):
    load_calls = []
    model = FakeModel()

    class FakeProcessorClass:
        @staticmethod
        def from_pretrained(model_name):
            load_calls.append(("processor", model_name))
            return FakeProcessor()

    class FakeModelClass:
        @staticmethod
        def from_pretrained(model_name):
            load_calls.append(("model", model_name))
            return model

    model.to = lambda device: model
    model.eval = lambda: model
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BlipForConditionalGeneration = FakeModelClass
    fake_transformers.BlipProcessor = FakeProcessorClass
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("IMAGE_CAPTION_MODEL", "local/test-model")
    monkeypatch.setattr(ultracamnet, "_model", None)
    monkeypatch.setattr(ultracamnet, "_processor", None)
    monkeypatch.setattr(ultracamnet, "_loaded_model_name", None)

    first = ultracamnet._load_model()
    second = ultracamnet._load_model()

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert load_calls == [
        ("processor", "local/test-model"),
        ("model", "local/test-model"),
    ]
